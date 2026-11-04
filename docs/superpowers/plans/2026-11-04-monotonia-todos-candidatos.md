# Adendo ao plano — monotonia em todos os candidatos

> Complementa `docs/superpowers/plans/2026-11-03-eficiencia-energetica.md` (mesmas Global Constraints e "Fluxo por tarefa"). Executar depois da Tarefa 9 e antes da Tarefa 14.

**Spec:** `docs/superpowers/specs/2026-11-03-eficiencia-energetica-design.md` §7.3 (decisão do usuário de 2026-11-04).

**Por quê:** no treino rápido só o XGBoost passou em `checar_monotonia`; linear, Random Forest e as redes reprovaram, então nunca poderiam vencer. O usuário escolheu tornar todos monotônicos e ampliar a busca das redes.

### Task 15: Candidatos monotônicos por construção

**Files:**
- Create: `src/eficiencia/modelos/restricoes.py`
- Modify: `src/eficiencia/modelos/xgb.py`, `src/eficiencia/modelos/linear.py`, `src/eficiencia/modelos/floresta.py`, `src/eficiencia/modelos/rede.py`
- Test: `tests/test_monotonia_candidatos.py`

**Interfaces:**
- Produces: `restricoes.sinais_monotonia(nomes: list[str], alvo: str) -> np.ndarray` (int, mesmo tamanho de `nomes`): `+1` nas colunas de `MONOTONAS` para alvos `mj_km*`, `-1` para alvos `kml_*`, `0` nas demais. `XGBoost` passa a usar esta função (sem mudar o comportamento).
- `Linear`: mesma interface pública e mesmo `sugerir` (alpha log [1e-3, 1e2]); o estimador passa a ser ridge com **limites de sinal**: `scipy.optimize.lsq_linear` sobre o sistema aumentado `[X; sqrt(alpha)·I]`, `[y_centrado; 0]`, com limites `[0, ∞)` onde o sinal é `+1`, `(−∞, 0]` onde é `−1`, livre onde é `0`; intercepto = média de y (X já padronizado).
- `Floresta`: `RandomForestRegressor(monotonic_cst=sinais_monotonia(nomes, alvo))`, resto igual.
- Redes (`RedePorAlvo` e `RedeMulti`): nova arquitetura monotônica no MLP compartilhado. Para cada saída j:
  `y_j = g_j(x_livre) + sinal_j · Σ_k softplus(s_jk(x_livre)) · m_k(x_mono)`
  onde `x_mono` = colunas de `MONOTONAS`, `x_livre` = demais colunas, `g` e `s` são MLPs livres, e `m` é um MLP com **pesos não negativos** (peso efetivo = `softplus(w)`) e ativações não decrescentes (ex.: ReLU/tanh), logo cada `m_k` é não decrescente em cada entrada de `x_mono`. `sinal_j = +1` para `mj_km*`, `−1` para `kml_*`. Como a padronização das entradas e dos alvos (desvio > 0) e o `exp` preservam a ordem, a monotonia vale nas unidades originais em qualquer etapa do treino.
- Busca das redes ampliada em `_sugerir_mlp`: `n_camadas` ∈ [1, 5], `largura` ∈ {32, 64, 128, 256, 512}; demais hiperparâmetros como antes (dropout, lr, weight_decay, tamanho_lote). `PARAMS_RAPIDOS` inalterados.

- [ ] **Step 1: Testes que falham**

```python
@pytest.mark.parametrize("nome", ["linear", "floresta", "xgboost", "rede_por_alvo", "rede_multi"])
def test_candidato_monotono_por_construcao(nome):
    df = carregar_base()
    cls = CANDIDATOS[nome]
    m = cls(**cls.PARAMS_RAPIDOS).fit(df[ENTRADAS], df[ALVOS])
    assert checar_monotonia(m, df) == []

def test_rede_monotona_mesmo_sem_treino():
    df = carregar_base()
    m = CANDIDATOS["rede_multi"](max_epocas=0).fit(df[ENTRADAS], df[ALVOS])
    assert checar_monotonia(m, df) == []

def test_sinais_monotonia():
    nomes = ["massa_om_kg", "potencia_kw", "f2_n_kmh2", "categoria=G"]
    assert list(sinais_monotonia(nomes, "mj_km_urbano")) == [1, 0, 1, 0]
    assert list(sinais_monotonia(nomes, "kml_gasolina_estrada")) == [-1, 0, -1, 0]

def test_treino_rapido_todos_aprovados(tmp_path):
    m = treinar(rapido=True, pasta_modelo=tmp_path / "m", pasta_relatorios=tmp_path / "r")
    assert all(c["monotonia"] == "aprovado" for c in m["candidatos"].values())
```
Run: `uv run pytest tests/test_monotonia_candidatos.py -v` → Expected: FAIL (import de `restricoes`; linear/floresta/redes reprovam).

- [ ] **Step 2: Implementar** `restricoes.py`, depois `xgb.py` (só passar a usar a função), `linear.py`, `floresta.py`, `rede.py`.
- [ ] **Step 3: Testes passam** — `uv run pytest -q` → PASS (todo o conjunto, incluindo `test_modelos_rede.py`, `test_treino.py`). Registrar no relatório o `erro_mj` da CV rápida de cada candidato antes × depois.
- [ ] **Step 4: Commit** — "Todos os candidatos monotônicos por construção e busca maior das redes".
