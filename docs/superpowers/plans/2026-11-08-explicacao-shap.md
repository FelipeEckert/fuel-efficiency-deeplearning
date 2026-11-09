# Explicação SHAP das previsões — Plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mostrar na página Veículo quanto cada entrada (massa, F0, potência, propulsão…) empurrou o MJ/km previsto para cima ou para baixo.

**Architecture:**
- Um módulo novo, `src/eficiencia/explicar.py`, usa o SHAP embutido no XGBoost (`Booster.predict(..., pred_contribs=True)`), sem dependência nova.
- As contribuições saem em log(alvo), porque o modelo treina em log. Elas são agregadas de volta às colunas de `ENTRADAS`: o one-hot é somado por categórica.
- Cada efeito vira um percentual multiplicativo, `exp(φ) − 1`, em relação ao "veículo médio" da base, que é `exp(base)`.
- A página Veículo ganha uma seção "Por que este valor?" com um gráfico de barras horizontais (plotly) por ciclo.

**Tech Stack:** Python 3.14, xgboost 3.4, pandas, Streamlit, plotly, pytest (`uv run pytest`).

**Spec:** não há documento de spec. O pedido do usuário foi: "adicione o SHAP" ao app, para explicar as previsões do XGBoost publicado.

## Global Constraints

- Texto da interface, docstrings e comentários em pt-BR.
- Nenhuma dependência nova: o SHAP vem do próprio xgboost.
- Só o XGBoost tem explicação. Com outro algoritmo publicado (por exemplo `linear`), a página mostra uma legenda e não quebra.
- Números com vírgula decimal: use `eficiencia.formato.numero` ou o `_fixo` que já existe na página.
- Rótulos das entradas vêm de `eficiencia.esquema.ROTULOS`.
- Ruff com `line-length = 110`.
- Não tocar em `.claude/` nem em `docs/superpowers/plans/2026-11-03-eficiencia-energetica.md`.

## Review Focus

1. Categóricas: a soma das colunas one-hot de uma categórica precisa virar uma linha só (ex.: `propulsao`), nunca `propulsao=Elétrico`.
2. Fechamento: `base + Σ contribuições` precisa bater com `log(previsão)` do mesmo alvo (tolerância 1e-4). O teste da Tarefa 1 cobre isso.
3. Pacote com vencedor não-XGBoost (o `pacote_rapido` dos testes pode ser `linear`): a página não pode levantar exceção.
4. Alvo sem previsão (ex.: `kml_etanol_*` num elétrico): fora do escopo. Só os alvos `mj_km_urbano` e `mj_km_estrada` são explicados.

---

### Task 1: Módulo `explicar`

**Files:**
- Create: `src/eficiencia/explicar.py`
- Test: `tests/test_explicar.py`

**Interfaces:**
- Consumes: `Candidato` com `.prep` (`Preprocessador`, que tem `.nomes` e `.transform(X) -> np.ndarray`) e `.estimadores: dict[str, XGBRegressor]`, de `eficiencia/modelos/base.py`. Também `ENTRADAS` e `CATEGORICAS`, de `eficiencia/esquema.py`.
- Produces:
  - `explicavel(modelo) -> bool`: True quando todos os estimadores têm `get_booster`, ou seja, são XGBoost.
  - `explicar(modelo, veiculo: dict, alvo: str) -> tuple[float, pd.Series]`, que devolve `(base_log, contribuicoes_log)`:
    - `contribuicoes_log` tem índice = `ENTRADAS`, na mesma ordem, e valores em log(alvo);
    - `base_log` é o termo de viés (a última coluna de `pred_contribs`).

- [ ] **Step 1: Escrever os testes que falham** em `tests/test_explicar.py`:

```python
import numpy as np

from eficiencia.esquema import ALVOS, ENTRADAS, carregar_base
from eficiencia.explicar import explicar, explicavel
from eficiencia.modelos import CANDIDATOS


def _modelo(nome):
    df = carregar_base()
    cls = CANDIDATOS[nome]
    return cls(**cls.PARAMS_RAPIDOS).fit(df[ENTRADAS], df[ALVOS]), df


def test_contribuicoes_fecham_com_a_previsao():
    m, df = _modelo("xgboost")
    veiculo = df[ENTRADAS].iloc[0].to_dict()
    for alvo in ("mj_km_urbano", "mj_km_estrada"):
        base, contrib = explicar(m, veiculo, alvo)
        assert list(contrib.index) == ENTRADAS  # uma linha por entrada; one-hot já agregado
        previsto = m.predict(df[ENTRADAS].iloc[[0]])[alvo].iloc[0]
        assert np.isclose(base + contrib.sum(), np.log(previsto), atol=1e-4)


def test_explicavel_so_xgboost():
    assert explicavel(_modelo("xgboost")[0])
    assert not explicavel(_modelo("linear")[0])
```

- [ ] **Step 2: Rodar e ver falhar**

Rode `uv run pytest tests/test_explicar.py -v`. O esperado é FALHAR com `ModuleNotFoundError: eficiencia.explicar`.

- [ ] **Step 3: Implementar `src/eficiencia/explicar.py`**

O módulo tem docstring de uma linha em pt-BR. Abordagem:
- `Xt = modelo.prep.transform(pd.DataFrame([veiculo]))`;
- `phi = modelo.estimadores[alvo].get_booster().predict(xgboost.DMatrix(Xt), pred_contribs=True)[0]`;
- `base = phi[-1]`;
- some `phi[:-1]` por entrada. A coluna `f"{c}={v}"` vai para `c`, e a numérica vai para ela mesma. Use `nome.split("=")[0]`: os nomes numéricos não têm "=".
- Devolva `pd.Series(...).reindex(ENTRADAS, fill_value=0.0)`.

Para `explicavel`, basta `all(hasattr(e, "get_booster") for e in modelo.estimadores.values())`. Proteja com `getattr(modelo, "estimadores", {})` e exija o dicionário não vazio.

- [ ] **Step 4: Rodar e ver passar**

Rode `uv run pytest tests/test_explicar.py -v`. O esperado é que os 2 testes PASSEM. Rode também `uv run ruff check src tests` e `uv run ruff format --check src tests`.

- [ ] **Step 5: Commit**

```bash
git add src/eficiencia/explicar.py tests/test_explicar.py
git commit -m "Explicação SHAP por entrada com o pred_contribs do XGBoost"
```

### Task 2: Seção "Por que este valor?" na página Veículo

**Files:**
- Modify: `app/pages/1_Veiculo.py`
- Test: `tests/test_app_veiculo.py`

**Interfaces:**
- Consumes: `explicar` e `explicavel`, da Tarefa 1. Também `ROTULOS`, de `eficiencia.esquema`, e `pacote.modelo`.
- Produces: nada que outra tarefa use.

**O que a seção mostra.** Ela entra logo depois de `_resultado(original)` e antes de `_e_se`:

- **Função:** `_explicacao(pacote, veiculo: dict) -> None`, com subtítulo "Por que este valor?".
- **Pacote não explicável:** `st.caption("A explicação por entrada está disponível só para o modelo XGBoost.")` e retorna.
- **Ciclo:** `st.radio("Ciclo", ["urbano", "estrada"], horizontal=True, key="shap_ciclo")`. O alvo é `f"mj_km_{ciclo}"`.
- **Legenda:** `"Veículo médio da base: {exp(base):,2} MJ/km. Cada barra é o quanto a entrada aumenta (+) ou reduz (−) o consumo em relação a ele; os efeitos se multiplicam."`, com vírgula decimal.
- **Gráfico:** `go.Bar` horizontal com `x = (exp(contrib) − 1) × 100` em %, `y = ROTULOS.get(entrada, entrada)`.
  - Mostra só as 8 entradas de maior |efeito|, ordenadas para que a maior fique no topo.
  - Cor `"#c0392b"` para efeito positivo (gasta mais) e `"#27ae60"` para negativo.
  - Use `xaxis_title="efeito no MJ/km (%)"`, `separators=",."`, `height=360` e `margin={"t": 30}`.
- **Escala do eixo:** a escala log mostra efeitos pequenos e grandes juntos, mas não precisa; deixe linear.

- [ ] **Step 1: Escrever o teste que falha** em `tests/test_app_veiculo.py`. Reaproveite `_pagina()`:

```python
def test_explicacao_sem_erro_nos_dois_ciclos(pacote_rapido):
    at = _pagina()
    assert any(s.value == "Por que este valor?" for s in at.subheader)
    if at.radio(key="shap_ciclo") is not None:
        at.radio(key="shap_ciclo").set_value("estrada").run()
    assert not at.exception
```

  O `pacote_rapido` pode ter vencedor `linear`. Nesse caso não há radio, e o teste só confere o subtítulo e a ausência de exceção. Se `at.radio(key=...)` levantar KeyError quando o widget não existe, troque por `[r for r in at.radio if r.key == "shap_ciclo"]`.

- [ ] **Step 2: Rodar e ver falhar**

Rode `uv run pytest tests/test_app_veiculo.py -v -k explicacao`. O esperado é FALHAR, porque o subtítulo não existe.

- [ ] **Step 3: Implementar `_explicacao` em `app/pages/1_Veiculo.py`** e chamá-la depois de `_resultado(original)`.

- [ ] **Step 4: Rodar e ver passar**

Rode `uv run pytest tests/test_app_veiculo.py tests/test_explicar.py -v`. O esperado é que tudo PASSE. Rode também o ruff, como na Tarefa 1.

- [ ] **Step 5: Commit**

```bash
git add app/pages/1_Veiculo.py tests/test_app_veiculo.py
git commit -m "Página Veículo: seção Por que este valor? com o SHAP por entrada"
```
