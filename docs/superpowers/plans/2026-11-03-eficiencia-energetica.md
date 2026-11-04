# Estimador de eficiência energética — plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> Execução combinada com o usuário: implementadores **Haiku 5.5**, revisores **Sonnet 5.5**, orquestrador **Opus 5.5**.

**Goal:** Motor Python que treina e compara 5 modelos de regressão sobre a base PBEV limpa e um app Streamlit que estima MJ/km e km/l (com incerteza, avisos de faixa, cenários e portfólio) para veículos futuros.

**Architecture:** Pacote `eficiencia` (em `src/`) com módulos pequenos: esquema/física → pré-processamento + candidatos (interface comum) → validação aninhada com Optuna → pós-modelo (monotonia, incerteza, faixas) → treino/artefatos → `prever()` e `media_portfolio()`. O app Streamlit (`app/`) só chama o pacote.

**Tech Stack:** Python 3.14, uv, pandas, numpy, scikit-learn, xgboost, torch (CPU), optuna, optuna-dashboard, joblib, streamlit, plotly (app), matplotlib (relatório), openpyxl; dev: pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-11-03-eficiencia-energetica-design.md` — ler junto com este plano. Dados: `data/README.md`.

## Global Constraints

- Python `>=3.14`; ambiente e dependências só via `uv` (`pyproject.toml` + `uv.lock` commitados).
- `torch` sempre do índice CPU `https://download.pytorch.org/whl/cpu` (senão o CI Linux baixa CUDA).
- Nomes de módulos, funções, colunas, mensagens e interface em **português** (sem acento em identificadores Python; acentos ok em textos).
- Entradas do modelo = exatamente `esquema.ENTRADAS`. **Nunca** usar `marca`, `modelo`, `area_m2`, `lcvm`, `nome_homologado`, `grupo_cv`, `versoes`, `n_versoes` nem qualquer coluna `kml_*`/`mj_km*` como entrada.
- Validação sempre por grupo (`grupo_cv`); nenhum grupo em treino e teste da mesma dobra.
- Tudo determinístico com `semente` (padrão 0).
- Treino roda na CPU; previsão no app instantânea.
- Desenvolvimento em Windows 11; CI em Ubuntu. Caminhos com `pathlib`, nunca separador fixo.
- Números exibidos ao usuário em formato brasileiro (vírgula decimal).

## Fluxo por tarefa (vale para todas)

1. Partir da `main` atualizada: `git switch main && git pull && git switch -c tarefa-NN-<nome>`.
2. TDD conforme os passos.
3. Antes do commit final: `uv run ruff check . && uv run ruff format --check . && uv run pytest -q` — tudo verde.
4. Commits pequenos, mensagem em português, terminando com `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
5. O orquestrador faz `git push -u origin <branch>` e `gh pr create` (corpo terminando com `🤖 Generated with [Claude Code](https://claude.com/claude-code)`). **O merge é do usuário.** Nunca commitar direto na `main` (exceto o commit inicial da Tarefa 1).

Dependências entre tarefas: 1 → 2 → 3 → 4 → {5, 6} ; 7 pode rodar em paralelo com 4–6 (só precisa da 3) ; {5, 6, 7} → 8 → 9 → 10 → {11, 12, 13} → 14.

## Review Focus

1. **Planilha enviada pelo usuário com vírgula decimal, texto em coluna numérica, coluna faltando ou sobrando** → erros por linha ("linha 3: F2 não é número"), nunca exceção. Teste na Tarefa 9 (`test_validar_texto_e_virgula`) e na 10 (`test_ler_planilha_colunas_faltando`).
2. **Combinação nunca vista** (ex.: Plug-in a diesel, categoria que não existe para a propulsão) → previsão sai, com aviso; combustíveis que não se aplicam ficam vazios, não com número inventado. Teste na Tarefa 9 (`test_combinacao_inedita_avisa`).
3. **Linha de portfólio com volume 0, negativo ou vazio; sem MJ/km conhecido e sem especificações; volume total 0** → linha excluída com mensagem; média vazia com erro, sem divisão por zero. Teste na Tarefa 10 (`test_volumes_invalidos`).
4. **Valores fisicamente impossíveis** (massa ≤ 0, F2 negativo, potência 0) → rejeitados na entrada com mensagem, antes do modelo. Teste na Tarefa 9 (`test_limites_fisicos`).
5. **Incoerência propulsão × campos** (Elétrico com combustível gasolina ou cilindrada > 0; Combustão com bateria > 0) → erro claro para combustível incompatível; cilindrada de elétrico e bateria de combustão são normalizadas para 0. Teste na Tarefa 9 (`test_coerencia_propulsao`).

---

### Task 1: Projeto, git, repositório e CI

**Files:**
- Create: `pyproject.toml`, `uv.lock`, `.gitignore`, `README.md`, `.github/workflows/ci.yml`
- Create: `src/eficiencia/__init__.py`, `src/eficiencia/dados.py` (conteúdo movido de `scripts/limpar_pbev.py`)
- Delete: `scripts/limpar_pbev.py`
- Test: `tests/test_dados.py`

**Interfaces:**
- Produces: `eficiencia.dados.limpar(caminho: Path = ENTRADA) -> tuple[pd.DataFrame, dict]`, `eficiencia.dados.salvar(df, saida: Path = SAIDA) -> None`, `eficiencia.dados.RAIZ: Path` (raiz do repositório), constantes `ENTRADA`, `SAIDA`, `COMPLEMENTO_BATERIAS`. Comando `uv run eficiencia-limpar`.

- [ ] **Step 1: Commit inicial do estado atual na `main`**

```bash
git init -b main
git add data docs scripts .claude/skills
git commit -m "Estado inicial: dados, limpeza, design e skills"
```

- [ ] **Step 2: Criar o repositório privado e enviar a `main`**

```bash
gh repo create FelipeEckert/fuel-efficiency-deeplearning --private --source . --push
```
Expected: URL do repositório impressa. Depois: `git switch -c tarefa-01-setup`.

- [ ] **Step 3: `pyproject.toml`**

Projeto `eficiencia`, `requires-python = ">=3.14"`, layout `src/`, build backend `hatchling`. Dependências: pandas, numpy, scikit-learn, xgboost, torch, optuna, optuna-dashboard, joblib, streamlit, plotly, matplotlib, openpyxl. Grupo `dev`: pytest, ruff. Script: `eficiencia-limpar = "eficiencia.dados:main"`. Torch pelo índice CPU:

```toml
[tool.uv.sources]
torch = { index = "pytorch-cpu" }

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.ruff]
line-length = 110

[tool.pytest.ini_options]
testpaths = ["tests"]
```
Run: `uv sync` → Expected: instala sem erro e cria `uv.lock`.

- [ ] **Step 4: Escrever `tests/test_dados.py` (falha: módulo não existe)**

Transformar cada `assert` de `checar()` e dos exemplos de parser em testes, mais:

```python
def test_regenerar_igual_ao_commitado(tmp_path):
    df, rel = limpar()
    salvar(df, tmp_path / "x.csv")
    novo = pd.read_csv(tmp_path / "x.csv", dtype={"lcvm": str})
    atual = pd.read_csv(SAIDA, dtype={"lcvm": str})
    pd.testing.assert_frame_equal(novo, atual)

def test_contagens():
    df, rel = limpar()
    assert len(df) == 510
    assert df["grupo_cv"].nunique() == 258
    assert rel["F2 convertido de N/(m/s)² para N/(km/h)²"] == 10
    assert int(df["bateria_estimada"].sum()) == 17
```
Run: `uv run pytest tests/test_dados.py -v` → Expected: FAIL (`ModuleNotFoundError: eficiencia`).

- [ ] **Step 5: Mover `scripts/limpar_pbev.py` para `src/eficiencia/dados.py`**

`RAIZ = Path(__file__).resolve().parents[2]`. `checar()` continua sendo chamada por `main()` antes de salvar. `main()` = antigo bloco `__main__`. Apagar `scripts/limpar_pbev.py` e a pasta `scripts/` se ficar vazia. Atualizar `data/README.md`: comando vira `uv run eficiencia-limpar`.

- [ ] **Step 6: Rodar testes** — `uv run pytest -q` → Expected: PASS.

- [ ] **Step 7: `.gitignore`, `README.md`, CI**

`.gitignore`: `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `optuna/`, `*.db`. `README.md` raiz: o que é o projeto (2 linhas), `uv sync`, `uv run eficiencia-limpar`, link para a spec e para `data/README.md` (comandos de treino e app entram nas tarefas 8 e 13).

`.github/workflows/ci.yml`: em `pull_request` e `push` para `main`; um job com id **`ci`** (é o nome exigido na proteção da `main`); `ubuntu-latest`; `actions/checkout@v4`; `astral-sh/setup-uv@v6`; `uv sync --locked`; `uv run ruff check .`; `uv run ruff format --check .`; `uv run pytest -q`.

- [ ] **Step 8: ruff limpo** — `uv run ruff check . && uv run ruff format --check .` → Expected: sem erros (rodar `ruff format .` antes, se preciso).

- [ ] **Step 9: Commit, PR e proteção da `main`**

```bash
git add -A && git commit -m "Projeto uv, pacote eficiencia, testes da limpeza e CI"
git push -u origin tarefa-01-setup && gh pr create --fill
```
Esperar o CI verde. Tentar proteger a `main` exigindo o check do CI:
```bash
gh api -X PUT repos/FelipeEckert/fuel-efficiency-deeplearning/branches/main/protection \
  -F required_status_checks[strict]=true -F "required_status_checks[contexts][]=ci" \
  -F enforce_admins=false -F required_pull_request_reviews=null -F restrictions=null
```
Se retornar 403 (plano gratuito), registrar no PR e avisar o usuário: vale a convenção do "Fluxo por tarefa".

---

### Task 2: Esquema e física

**Files:**
- Create: `src/eficiencia/esquema.py`, `src/eficiencia/fisica.py`
- Test: `tests/test_esquema.py`, `tests/test_fisica.py`

**Interfaces:**
- Produces (`esquema`):
  - `CATEGORICAS = ["propulsao", "combustivel", "categoria", "categoria_proconve", "transmissao", "direcao"]`
  - `NUMERICAS = ["massa_om_kg", "classe_inercia_kg", "pbt_kg", "f0_n", "f2_n_kmh2", "potencia_kw", "cilindrada_cm3", "n_marchas", "bateria_kwh", "bateria_estimada"]`
  - `ENTRADAS = CATEGORICAS + NUMERICAS`
  - `ALVOS = ["mj_km_urbano", "mj_km_estrada", "kml_gasolina_urbano", "kml_gasolina_estrada", "kml_etanol_urbano", "kml_etanol_estrada"]`
  - `MONOTONAS = ["massa_om_kg", "classe_inercia_kg", "f0_n", "f2_n_kmh2"]`
  - `VALORES: dict[str, list[str]]` — valores válidos de cada categórica (os da base limpa)
  - `ROTULOS: dict[str, str]` — rótulo pt-BR de cada entrada e saída. Fixos (usados em mensagens testadas): `"f2_n_kmh2": "F2"`, `"f0_n": "F0"`, `"massa_om_kg": "Massa em ordem de marcha (kg)"`, `"categoria": "Categoria"`, `"combustivel": "Combustível"`, `"propulsao": "Propulsão"`, `"potencia_kw": "Potência (kW)"`. Os demais a critério.
  - `CAMINHO_BASE: Path`; `carregar_base(caminho: Path = CAMINHO_BASE) -> pd.DataFrame` (lê com `dtype={"lcvm": str}`)
- Produces (`fisica`):
  - `classe_inercia(massa_om_kg: float) -> float`
  - `combinar_mj(urbano, estrada)` → `0.55*u + 0.45*e` (escalar, array ou Series)
  - `combinar_kml(urbano, estrada)` → `1 / (0.55/u + 0.45/e)`
  - `FATOR_ELETRICO = 20.29`; `kml_eletrico_eq(mj_km)` → `FATOR_ELETRICO / mj_km`
  - `fatores_diesel(df: pd.DataFrame) -> tuple[float, float]` (medianas de `kml_diesel_* × mj_km_*`, urbano e estrada)
  - `kml_diesel(mj_km, fator: float)` → `fator / mj_km`
  - `combustiveis_aplicaveis(propulsao: str, combustivel: str) -> set[str]` ⊆ `{"gasolina", "etanol", "eletrico_eq", "diesel"}`

- [ ] **Step 1: Testes que falham**

```python
def test_entradas_sem_colunas_proibidas():
    proibidas = {"marca", "modelo", "area_m2", "lcvm", "nome_homologado", "grupo_cv", "versoes", "n_versoes"}
    assert not proibidas & set(ENTRADAS)
    assert not [c for c in ENTRADAS if c.startswith(("kml_", "mj_km"))]

def test_base_tem_todas_as_colunas():
    df = carregar_base()
    assert set(ENTRADAS + ALVOS + ["grupo_cv", "mj_km"]) <= set(df.columns)
    for c in CATEGORICAS:
        assert set(df[c].dropna()) <= set(VALORES[c])
    assert set(ROTULOS) >= set(ENTRADAS)

@pytest.mark.parametrize("massa,classe", [(1490, 1644), (1650, 1758), (300, 454), (3000, 2495)])
def test_classe_inercia(massa, classe):
    assert classe_inercia(massa) == classe

def test_combinar():
    assert combinar_mj(2.23326400123257, 1.598742623945293) == pytest.approx(1.947729381453296, abs=1e-9)
    assert round(combinar_kml(10.3, 13.5), 1) == 11.5
    assert kml_eletrico_eq(0.5) == pytest.approx(40.58)

def test_fatores_diesel():
    urb, est = fatores_diesel(carregar_base())
    assert urb == pytest.approx(27.96, abs=0.05) and est == pytest.approx(25.57, abs=0.05)

@pytest.mark.parametrize("prop,comb,esperado", [
    ("Combustão", "flex", {"gasolina", "etanol"}), ("Combustão", "gasolina", {"gasolina"}),
    ("Plug-in", "gasolina", {"gasolina", "eletrico_eq"}), ("Plug-in", "flex", {"gasolina", "etanol", "eletrico_eq"}),
    ("Elétrico", "eletrico", {"eletrico_eq"}), ("Híbrido", "diesel", {"diesel"})])
def test_combustiveis_aplicaveis(prop, comb, esperado):
    assert combustiveis_aplicaveis(prop, comb) == esperado
```
Run: `uv run pytest tests/test_esquema.py tests/test_fisica.py -v` → Expected: FAIL (import).

- [ ] **Step 2: Implementar `esquema.py` e `fisica.py`**

`classe_inercia`: tabela da spec §8.1 (mínimo da faixa → classe), massa de ensaio = massa + 136, classe da maior faixa com mínimo ≤ massa de ensaio (`bisect`). Ignorar a linha "2608→2495".

- [ ] **Step 3: Testes passam** — `uv run pytest -q` → Expected: PASS.

- [ ] **Step 4: Commit** — `git commit -am "Esquema de colunas e funções físicas"` (adicionar arquivos novos).

---

### Task 3: Pré-processamento, interface de candidato e regressão linear

**Files:**
- Create: `src/eficiencia/preprocess.py`, `src/eficiencia/modelos/__init__.py`, `src/eficiencia/modelos/base.py`, `src/eficiencia/modelos/linear.py`
- Test: `tests/test_preprocess.py`, `tests/test_modelos.py`

**Interfaces:**
- Consumes: `esquema.ENTRADAS/CATEGORICAS/NUMERICAS/ALVOS`.
- Produces:
  - `Preprocessador().fit(X: pd.DataFrame) -> Preprocessador`; `.transform(X) -> np.ndarray` (float32); `.nomes: list[str]` (nome de cada coluna gerada; numéricas mantêm o nome original, one-hot = `"<coluna>=<valor>"`). Regras: `n_marchas` vazio → 0; `bateria_estimada` → 0/1; categoria desconhecida → todas as colunas one-hot 0; numéricas padronizadas (média/desvio do fit).
  - `class Candidato` (base abstrata em `base.py`): `nome: str`; `PARAMS_RAPIDOS: dict`; `__init__(self, semente: int = 0, **params)`; `@staticmethod sugerir(trial: optuna.Trial) -> dict`; `fit(X: pd.DataFrame, Y: pd.DataFrame) -> Self` (X = colunas `ENTRADAS`; Y = colunas `ALVOS`, NaN onde não se aplica); `predict(X) -> pd.DataFrame` (colunas `ALVOS`, unidades originais, mesmo índice de X).
  - `class CandidatoPorAlvo(Candidato)`: cria um `Preprocessador` no `fit`, treina um estimador por alvo só nas linhas com alvo presente, **em log do alvo**, e devolve `exp` na previsão. Subclasses implementam `_novo_estimador(self, alvo: str, nomes: list[str])` → objeto com `fit(X, y)`/`predict(X)` (numpy).
  - `modelos.CANDIDATOS: dict[str, type[Candidato]]` — esta tarefa registra `"linear"`.
  - `Linear` (`nome = "linear"`): `sklearn.linear_model.Ridge(alpha)`; `sugerir`: `alpha` log-uniforme em [1e-3, 1e2].

- [ ] **Step 1: Testes que falham**

```python
def test_preprocessador_categoria_nova_e_marchas_vazias():
    df = carregar_base()
    p = Preprocessador().fit(df[ENTRADAS])
    novo = df[ENTRADAS].iloc[[0]].copy()
    novo["categoria"] = "INEXISTENTE"; novo["n_marchas"] = np.nan
    x = p.transform(novo)
    assert np.isfinite(x).all()
    assert all(x[0, i] == 0 for i, n in enumerate(p.nomes) if n.startswith("categoria="))

def test_linear_fit_predict_formato():
    df = carregar_base()
    m = CANDIDATOS["linear"](alpha=1.0).fit(df[ENTRADAS], df[ALVOS])
    prev = m.predict(df[ENTRADAS].iloc[:5])
    assert list(prev.columns) == ALVOS and (prev.index == df.index[:5]).all()
    assert (prev > 0).all().all()

def test_por_alvo_ignora_alvo_ausente():
    df = carregar_base()
    m = CANDIDATOS["linear"](alpha=1.0).fit(df[ENTRADAS], df[ALVOS])  # etanol só em 168 linhas
    assert np.isfinite(m.predict(df[ENTRADAS])["kml_etanol_urbano"]).all()
```
Run: `uv run pytest tests/test_preprocess.py tests/test_modelos.py -v` → Expected: FAIL (import).

- [ ] **Step 2: Implementar** os quatro arquivos com as interfaces acima.
- [ ] **Step 3: Testes passam** — `uv run pytest -q` → PASS.
- [ ] **Step 4: Commit** — "Pré-processamento, interface de candidato e regressão linear".

---

### Task 4: Validação aninhada com Optuna

**Files:**
- Create: `src/eficiencia/validacao.py`
- Test: `tests/test_validacao.py`

**Interfaces:**
- Consumes: `modelos.CANDIDATOS`, `fisica.combinar_mj`, `esquema`.
- Produces:
  - `erro_percentual(real, previsto) -> np.ndarray` → `|previsto/real − 1|`
  - `dobras_por_grupo(grupos: pd.Series, n: int, semente: int = 0) -> list[tuple[np.ndarray, np.ndarray]]` (índices posicionais treino/teste; grupos embaralhados com a semente)
  - `erro_mj_combinado(df: pd.DataFrame, previsto: pd.DataFrame) -> np.ndarray` (combina urbano/estrada previstos com `combinar_mj` e compara com `df["mj_km"]`)
  - `buscar_params(nome: str, df: pd.DataFrame, n_tentativas: int, storage: str | None, nome_estudo: str, n_dobras: int = 3, rapido: bool = False, semente: int = 0) -> dict` — estudo Optuna (minimizar média de `erro_mj_combinado` na CV por grupo interna), `load_if_exists=True`; cada chamada **adiciona** `n_tentativas`; poda com `MedianPruner` reportando o erro por dobra; com `rapido=True` os params de `PARAMS_RAPIDOS` sobrescrevem os sugeridos.
  - `@dataclass ResultadoCV`: `nome: str`, `oof: pd.DataFrame` (colunas `ALVOS`, índice de df), `params_por_dobra: list[dict]`, `erro_mj: float`, `erro_por_propulsao: dict[str, float]`, `erro_por_alvo: dict[str, float]`, `n_tentativas: int` (total acumulado nos estudos)
  - `cv_aninhada(nome: str, df: pd.DataFrame, n_tentativas: int, storage: str | None, n_externas: int = 5, n_internas: int = 3, rapido: bool = False, semente: int = 0) -> ResultadoCV` — nome de estudo por dobra externa: `f"{nome}-{hash8(df)}-ext{k}"` (hash dos dados: base nova = estudo novo).

- [ ] **Step 1: Testes que falham**

```python
def test_dobras_sem_vazamento():
    df = carregar_base()
    dobras = dobras_por_grupo(df["grupo_cv"], 5)
    testes = np.concatenate([te for _, te in dobras])
    assert sorted(testes) == list(range(len(df)))
    for tr, te in dobras:
        assert not set(df["grupo_cv"].iloc[tr]) & set(df["grupo_cv"].iloc[te])

def test_cv_aninhada_linear():
    r = cv_aninhada("linear", carregar_base(), n_tentativas=3, storage=None, n_externas=2, n_internas=2)
    assert r.oof.notna()["mj_km_urbano"].all()
    assert r.erro_mj < 0.10                      # referência medida: 7,7 %
    assert set(r.erro_por_propulsao) == {"Combustão", "Híbrido", "Plug-in", "Elétrico"}

def test_estudo_retomavel(tmp_path):
    storage = f"sqlite:///{tmp_path / 'e.db'}"
    df = carregar_base()
    buscar_params("linear", df, 2, storage, "t", n_dobras=2)
    buscar_params("linear", df, 2, storage, "t", n_dobras=2)
    assert len(optuna.load_study(study_name="t", storage=storage).trials) == 4
```
Run: `uv run pytest tests/test_validacao.py -v` → FAIL (import).

- [ ] **Step 2: Implementar `validacao.py`.** Em cada dobra externa: `buscar_params` só com o treino da dobra (dobras internas por grupo), treinar com os melhores params, prever o teste → `oof`.
- [ ] **Step 3: Testes passam** — PASS.
- [ ] **Step 4: Commit** — "Validação aninhada por grupo com Optuna retomável".

---

### Task 5: Random Forest e XGBoost monotônico

**Files:**
- Create: `src/eficiencia/modelos/floresta.py`, `src/eficiencia/modelos/xgb.py`
- Modify: `src/eficiencia/modelos/__init__.py` (registrar `"floresta"`, `"xgboost"`)
- Test: `tests/test_modelos_arvores.py`

**Interfaces:**
- Consumes: `CandidatoPorAlvo`, `Preprocessador.nomes`, `esquema.MONOTONAS`.
- Produces: `Floresta` (`nome="floresta"`, `RandomForestRegressor`; `sugerir` = grade pequena via `trial.suggest_categorical`: `n_estimators` ∈ {200, 500}, `max_features` ∈ {0.3, 0.6, 1.0}, `min_samples_leaf` ∈ {1, 2, 4}); `XGBoost` (`nome="xgboost"`, `XGBRegressor` com `monotone_constraints` = tupla do tamanho de `Preprocessador.nomes`: nas colunas cujo nome está em `MONOTONAS`, `+1` para alvos `mj_km_*` e `-1` para alvos `kml_*`; `0` nas demais). `PARAMS_RAPIDOS`: `{"n_estimators": 20}` em ambos.

- [ ] **Step 1: Testes que falham**

```python
def test_xgboost_monotono_em_f2():
    df = carregar_base()
    m = CANDIDATOS["xgboost"](**CANDIDATOS["xgboost"].PARAMS_RAPIDOS).fit(df[ENTRADAS], df[ALVOS])
    base = df[ENTRADAS].iloc[[0]]
    grade = pd.concat([base.assign(f2_n_kmh2=v) for v in np.linspace(0.02, 0.12, 15)], ignore_index=True)
    p = m.predict(grade)
    assert (np.diff(p["mj_km_urbano"]) >= -1e-9).all()
    assert (np.diff(p["kml_gasolina_urbano"]) <= 1e-9).all()

@pytest.mark.parametrize("nome", ["floresta", "xgboost"])
def test_cv_rapida(nome):
    r = cv_aninhada(nome, carregar_base(), n_tentativas=2, storage=None, n_externas=2, n_internas=2, rapido=True)
    assert r.erro_mj < 0.12
```
Run → FAIL (chave não registrada).

- [ ] **Step 2: Implementar** os dois candidatos. `XGBoost.sugerir`: `n_estimators` [200, 1500], `max_depth` [2, 6], `learning_rate` log [0.01, 0.3], `subsample` [0.6, 1], `colsample_bytree` [0.5, 1], `min_child_weight` [1, 10], `reg_lambda` log [1e-3, 10].
- [ ] **Step 3: Testes passam** — PASS.
- [ ] **Step 4: Commit** — "Random Forest e XGBoost com restrição monotônica".

---

### Task 6: Redes neurais (por alvo e multi-saída)

**Files:**
- Create: `src/eficiencia/modelos/rede.py`
- Modify: `src/eficiencia/modelos/__init__.py` (registrar `"rede_por_alvo"`, `"rede_multi"`)
- Test: `tests/test_modelos_rede.py`

**Interfaces:**
- Consumes: `Candidato`, `CandidatoPorAlvo`, `Preprocessador`.
- Produces: `RedePorAlvo(CandidatoPorAlvo)` (`nome="rede_por_alvo"`; o estimador é um MLP PyTorch com interface `fit/predict` numpy); `RedeMulti(Candidato)` (`nome="rede_multi"`; uma rede com 6 saídas sobre log dos alvos padronizados, **perda MSE mascarada** — alvo NaN não contribui). Ambas: CPU, `torch.manual_seed(semente)`; parada antecipada usando 15 % aleatórios do treino recebido como validação interna (o candidato não recebe grupos; o erro reportado continua vindo da CV por grupo); parâmetro `max_epocas` (padrão 500). `PARAMS_RAPIDOS = {"max_epocas": 5}`. Arquitetura e espaço de busca: a critério do implementador (sugestão: 1–3 camadas ocultas de 16–256, dropout 0–0,3, `lr` log [1e-4, 1e-2], `weight_decay` log [1e-6, 1e-2]).

- [ ] **Step 1: Testes que falham**

```python
def test_rede_multi_alvos_ausentes_nao_viram_nan():
    df = carregar_base()
    m = CANDIDATOS["rede_multi"](**CANDIDATOS["rede_multi"].PARAMS_RAPIDOS).fit(df[ENTRADAS], df[ALVOS])
    p = m.predict(df[ENTRADAS])
    assert np.isfinite(p.values).all() and (p.values > 0).all()

def test_rede_deterministica():
    df = carregar_base()
    a = CANDIDATOS["rede_por_alvo"](semente=1, max_epocas=5).fit(df[ENTRADAS], df[ALVOS]).predict(df[ENTRADAS])
    b = CANDIDATOS["rede_por_alvo"](semente=1, max_epocas=5).fit(df[ENTRADAS], df[ALVOS]).predict(df[ENTRADAS])
    pd.testing.assert_frame_equal(a, b)

@pytest.mark.parametrize("nome", ["rede_por_alvo", "rede_multi"])
def test_cv_rapida_rede(nome):
    r = cv_aninhada(nome, carregar_base(), n_tentativas=2, storage=None, n_externas=2, n_internas=2, rapido=True)
    assert np.isfinite(r.erro_mj)
```
Run → FAIL.

- [ ] **Step 2: Implementar `rede.py`** (MLP compartilhado entre as duas classes).
- [ ] **Step 3: Testes passam** — PASS.
- [ ] **Step 4: Commit** — "Redes neurais PyTorch por alvo e multi-saída".

---

### Task 7: Monotonia, incerteza e faixas

**Files:**
- Create: `src/eficiencia/monotonia.py`, `src/eficiencia/incerteza.py`, `src/eficiencia/faixa.py`, `src/eficiencia/formato.py`
- Test: `tests/test_monotonia.py`, `tests/test_incerteza.py`, `tests/test_faixa.py`

**Interfaces:**
- Consumes: `Candidato.predict`, `esquema.MONOTONAS/ALVOS/ROTULOS/NUMERICAS/CATEGORICAS`, `fisica.combinar_*`.
- Produces:
  - `formato.numero(v: float, casas: int | None = None) -> str` — vírgula decimal, sem zeros sobrando (`0.025 → "0,025"`, `1644.0 → "1644"`).
  - `monotonia.checar_monotonia(modelo, df: pd.DataFrame, n_veiculos: int = 30, fatores=(0.8, 0.9, 1.0, 1.1, 1.2), semente: int = 0) -> list[str]` — para veículos sorteados da base, varia cada coluna de `MONOTONAS` (as outras fixas); MJ/km (urbano/estrada) não pode cair e `kml_*` não pode subir (tolerância 1e-9 relativa). Lista vazia = passou; cada violação vira uma mensagem.
  - `incerteza.SAIDAS_COM_FAIXA = ALVOS + ["mj_km", "kml_gasolina", "kml_etanol"]`
  - `incerteza.calcular_quantis(df, oof: pd.DataFrame, nivel: float = 0.9) -> dict[str, dict[str, float]]` — propulsão → saída → q (erro relativo). Quantil conformal com correção de amostra finita: posição `ceil((n+1)·nivel)` na lista ordenada de erros (limitada a n). Combinados calculados de oof urbano/estrada.
  - `incerteza.aplicar(previsto: float, q: float) -> tuple[float, float]` → `(previsto·(1−q), previsto·(1+q))`
  - `faixa.calcular_faixas(df) -> dict` — `{"numericas": {propulsao: {coluna: (min, max)}}, "categoricas": {propulsao: {coluna: set[str]}}}`
  - `faixa.avisos(veiculo: Mapping, faixas: dict) -> list[str]`

- [ ] **Step 1: Testes que falham**

```python
class Falso:  # MJ/km cai com a massa → deve reprovar
    def predict(self, X):
        v = 3 - X["massa_om_kg"].to_numpy() / 1000
        return pd.DataFrame({a: v for a in ALVOS}, index=X.index)

def test_monotonia_reprova_modelo_invertido():
    assert checar_monotonia(Falso(), carregar_base())

def test_quantil_conformal_exato():
    df = pd.DataFrame({"propulsao": ["Combustão"] * 10, "mj_km_urbano": 1.0, "mj_km_estrada": 1.0, "mj_km": 1.0})
    for a in ALVOS[2:]:
        df[a] = np.nan
    oof = pd.DataFrame({"mj_km_urbano": 1 + np.arange(1, 11) / 100, "mj_km_estrada": 1 + np.arange(1, 11) / 100})
    assert calcular_quantis(df, oof)["Combustão"]["mj_km_urbano"] == pytest.approx(0.10)

def test_avisos_texto():
    faixas = {"numericas": {"Elétrico": {"f2_n_kmh2": (0.029, 0.08)}}, "categoricas": {"Elétrico": {"categoria": {"SUVC"}}}}
    v = {"propulsao": "Elétrico", "f2_n_kmh2": 0.025, "categoria": "PIC"}
    assert avisos(v, faixas) == ["F2 = 0,025 abaixo de qualquer Elétrico da base (mínimo 0,029)",
                                 "Categoria = PIC nunca vista em Elétrico na base"]
    assert avisos({"propulsao": "Elétrico", "f2_n_kmh2": 0.09, "categoria": "SUVC"}, faixas) == [
        "F2 = 0,09 acima de qualquer Elétrico da base (máximo 0,08)"]
```
Run → FAIL.

- [ ] **Step 2: Implementar** os quatro módulos.
- [ ] **Step 3: Testes passam** — PASS.
- [ ] **Step 4: Commit** — "Teste de monotonia, incerteza conformal e avisos de faixa".

---

### Task 8: Treino, seleção do vencedor, artefatos e relatório

**Files:**
- Create: `src/eficiencia/artefatos.py`, `src/eficiencia/treino.py`, `tests/conftest.py`
- Modify: `pyproject.toml` (script `eficiencia-treinar = "eficiencia.treino:main"`), `README.md` (comandos de treino e `uv run optuna-dashboard sqlite:///optuna/estudos.db`)
- Test: `tests/test_treino.py`, `tests/test_artefato_versionado.py`

**Interfaces:**
- Consumes: `cv_aninhada`, `buscar_params`, `CANDIDATOS`, `checar_monotonia`, `calcular_quantis`, `calcular_faixas`, `fatores_diesel`.
- Produces:
  - `@dataclass Pacote` (`artefatos.py`): `algoritmo: str`, `modelo: Candidato`, `quantis: dict`, `faixas: dict`, `fatores_diesel: tuple[float, float]`, `referencias: dict[str, dict[str, float]]` (categoria → mediana de `f0_n` e `f2_n_kmh2`), `metadados: dict` (algoritmo, params, data ISO, hash do CSV, versões de bibliotecas, tentativas por candidato).
  - `artefatos.PASTA_MODELO = RAIZ / "modelos/vencedor"`; `salvar(pacote, pasta=PASTA_MODELO)` → `pacote.joblib` + `metadados.json`; `carregar(pasta: Path | None = None) -> Pacote` (sem argumento: variável de ambiente `EFICIENCIA_MODELO`, senão `PASTA_MODELO`).
  - `treino.PASTA_RELATORIOS = RAIZ / "relatorios"`
  - `treino.treinar(n_tentativas: int = 50, candidatos: list[str] | None = None, rapido: bool = False, storage: str | None = "sqlite:///optuna/estudos.db", pasta_modelo: Path = PASTA_MODELO, pasta_relatorios: Path = PASTA_RELATORIOS) -> dict` — devolve o conteúdo de `metricas.json`. `rapido=True`: 2 tentativas, 2×2 dobras, `PARAMS_RAPIDOS`, `storage=None`.
  - CLI: `uv run eficiencia-treinar [--tentativas N] [--candidatos linear,xgboost] [--rapido]`.
  - `metricas.json`: `{"vencedor": str, "candidatos": {nome: {"erro_mj": float, "erro_por_propulsao": {...}, "erro_por_alvo": {...}, "n_tentativas": int, "monotonia": "aprovado" | "reprovado", "violacoes": [str, ...]}}}`
  - `relatorio.md` + `figuras/` (matplotlib): tabela de erros por candidato × propulsão, previsto × real do vencedor, tentativas acumuladas.
  - Fixtures de sessão em `conftest.py`:
    - `treino_rapido` → `SimpleNamespace(pacote, metricas, pasta_modelo, pasta_relatorios)`: roda `treinar(rapido=True, candidatos=["linear", "xgboost"], pasta_modelo=tmp/"modelo", pasta_relatorios=tmp/"relatorios")` uma vez por sessão e define as variáveis de ambiente `EFICIENCIA_MODELO` e `EFICIENCIA_RELATORIOS` para essas pastas.
    - `pacote_rapido` → `treino_rapido.pacote`.

Algoritmo de seleção (spec §7.3):
```text
para cada candidato: r = cv_aninhada(...)                      # erro reportado
para cada candidato: p = buscar_params(nome, df_todo, ..., nome_estudo=f"{nome}-{hash8}-final")
                     m = Candidato(**p).fit(df_todo)           # modelo final
                     violacoes = checar_monotonia(m, df_todo)
vencedor = menor r.erro_mj entre os sem violação
se nenhum passar: erro explícito (não salva artefato)
quantis = calcular_quantis(df_todo, r_vencedor.oof)
```

- [ ] **Step 1: Testes que falham**

```python
def test_treino_rapido(treino_rapido):
    t = treino_rapido
    assert t.pacote.algoritmo == t.metricas["vencedor"] in {"linear", "xgboost"}
    assert t.metricas["candidatos"][t.metricas["vencedor"]]["monotonia"] == "aprovado"
    assert set(t.pacote.quantis) == {"Combustão", "Híbrido", "Plug-in", "Elétrico"}
    assert (t.pasta_relatorios / "metricas.json").exists() and (t.pasta_relatorios / "relatorio.md").exists()
    assert carregar(t.pasta_modelo).algoritmo == t.pacote.algoritmo
```

`tests/test_artefato_versionado.py`:
```python
@pytest.mark.skipif(not (PASTA_MODELO / "pacote.joblib").exists(), reason="modelo ainda não treinado")
def test_artefato_versionado_monotono():
    p = carregar(PASTA_MODELO)
    assert checar_monotonia(p.modelo, carregar_base()) == []
```
Run → FAIL.

- [ ] **Step 2: Implementar `artefatos.py` e `treino.py`.**
- [ ] **Step 3: Testes passam** — PASS. Rodar também `uv run eficiencia-treinar --rapido` e conferir que `relatorios/` e `modelos/vencedor/` foram gerados; **não commitar** esses arquivos nesta tarefa (apagar depois).
- [ ] **Step 4: Commit** — "Treino com seleção do vencedor, artefatos e relatório".

---

### Task 9: Validação de entrada e `prever()`

**Files:**
- Create: `src/eficiencia/entrada.py`, `src/eficiencia/prever.py`
- Test: `tests/test_entrada.py`, `tests/test_prever.py`

**Interfaces:**
- Consumes: `Pacote`, `carregar`, `incerteza.aplicar`, `faixa.avisos`, `fisica.*`, `esquema.*`.
- Produces:
  - `entrada.LIMITES_FISICOS = {"massa_om_kg": (300, 10000), "classe_inercia_kg": (300, 10000), "pbt_kg": (300, 15000), "f0_n": (1, 2000), "f2_n_kmh2": (0.001, 1), "potencia_kw": (1, 2000), "cilindrada_cm3": (0, 10000), "n_marchas": (0, 12), "bateria_kwh": (0, 300)}`
  - `entrada.validar(df: pd.DataFrame, primeira_linha: int = 1) -> tuple[pd.DataFrame, list[str]]` — normaliza (texto aparado; categóricas comparadas sem diferença de maiúsculas/acentos e devolvidas no valor canônico de `VALORES`; números aceitam `"1,5"`); coerência: Combustão → `bateria_kwh = 0`; Elétrico → `cilindrada_cm3 = 0` e exige `combustivel == "eletrico"`; `combustivel == "eletrico"` exige propulsão Elétrico; `bateria_estimada = False` sempre; `n_marchas` pode ficar vazio. Obrigatórios: todas as `ENTRADAS` exceto `n_marchas` e `bateria_estimada`; `bateria_kwh` obrigatório se propulsão ≠ Combustão. Devolve só as linhas válidas (índice original) e mensagens no formato `"linha {n}: {ROTULO} {problema}"`.
  - `prever.SAIDAS = ["mj_km", "mj_km_urbano", "mj_km_estrada"] + [f"kml_{c}{s}" for c in ("gasolina", "etanol", "eletrico_eq", "diesel") for s in ("", "_urbano", "_estrada")]`
  - `prever.prever(veiculos: pd.DataFrame | Mapping, pacote: Pacote | None = None) -> pd.DataFrame` — valida (erro de validação → `ValueError` com todas as mensagens); para cada saída: coluna valor, `_min`, `_max`; combustíveis fora de `combustiveis_aplicaveis` = NaN; coluna `avisos: list[str]`. Combinados: `mj_km = combinar_mj(urbano, estrada)`; todo `kml_*` combinado = `combinar_kml(urbano, estrada)` (harmônica); `kml_diesel_*` = `kml_diesel(mj, fator)` com `pacote.fatores_diesel`. Faixas: alvos modelados e combinados por `aplicar` com o q da propulsão; `kml_eletrico_eq*` e `kml_diesel*` derivados do MJ/km e da faixa dele (mín. de km/l = fator / MJ máx.).

- [ ] **Step 1: Testes que falham** (`pacote_rapido` da Tarefa 8)

```python
def veiculo(**kw):  # A3 Sedan da base como ponto de partida
    v = carregar_base()[ENTRADAS].iloc[0].to_dict(); v.update(kw); return v

def test_prever_saidas_e_faixa(pacote_rapido):
    r = prever(veiculo(), pacote_rapido).iloc[0]
    assert r["mj_km_min"] < r["mj_km"] < r["mj_km_max"]
    assert r["mj_km"] == pytest.approx(0.55 * r["mj_km_urbano"] + 0.45 * r["mj_km_estrada"])
    assert np.isnan(r["kml_etanol"]) and np.isnan(r["kml_diesel"])   # A3 é gasolina

def test_eletrico_km_l_equivalente(pacote_rapido):
    df = carregar_base(); v = df[df.propulsao == "Elétrico"][ENTRADAS].iloc[0].to_dict()
    r = prever(v, pacote_rapido).iloc[0]
    assert r["kml_eletrico_eq_urbano"] == pytest.approx(20.29 / r["mj_km_urbano"])
    assert np.isnan(r["kml_gasolina"])

def test_validar_texto_e_virgula():
    df = pd.DataFrame([veiculo(f2_n_kmh2="0,035", massa_om_kg="abc", categoria=" g ")])
    ok, erros = validar(df)
    assert erros == ["linha 1: Massa em ordem de marcha (kg) não é número"] and ok.empty

def test_limites_fisicos():
    _, erros = validar(pd.DataFrame([veiculo(f2_n_kmh2=-0.01, potencia_kw=0)]))
    assert len(erros) == 2 and all(e.startswith("linha 1: ") for e in erros)

def test_coerencia_propulsao():
    _, erros = validar(pd.DataFrame([veiculo(propulsao="Elétrico", combustivel="gasolina")]))
    assert any("Combustível" in e for e in erros)
    ok, _ = validar(pd.DataFrame([veiculo(propulsao="Combustão", bateria_kwh=60)]))
    assert ok.iloc[0]["bateria_kwh"] == 0

def test_combinacao_inedita_avisa(pacote_rapido):
    r = prever(veiculo(propulsao="Plug-in", combustivel="diesel", bateria_kwh=20), pacote_rapido).iloc[0]
    assert r["avisos"] and np.isfinite(r["mj_km"]) and np.isfinite(r["kml_diesel"])
```
Run → FAIL.

- [ ] **Step 2: Implementar `entrada.py` e `prever.py`.**
- [ ] **Step 3: Testes passam** — PASS.
- [ ] **Step 4: Commit** — "Validação de entrada e prever() com faixas e avisos".

---

### Task 10: Portfólio (motor)

**Files:**
- Create: `src/eficiencia/portfolio.py`
- Test: `tests/test_portfolio.py`

**Interfaces:**
- Consumes: `prever`, `entrada.validar`, `esquema`.
- Produces:
  - `COLUNAS_PORTFOLIO = ["nome", "volume", "mj_km_conhecido"] + [c for c in ENTRADAS if c != "bateria_estimada"]`
  - `modelo_planilha() -> bytes` (xlsx vazio com cabeçalhos = `ROTULOS` das colunas, ordem de `COLUNAS_PORTFOLIO`; rótulos de `nome`, `volume`, `mj_km_conhecido` = "Nome", "Volume", "MJ/km conhecido")
  - `ler_planilha(arquivo: bytes | BinaryIO) -> tuple[pd.DataFrame, list[str]]` (aceita cabeçalho por rótulo ou por nome de coluna; colunas obrigatórias faltando → erro; colunas extras ignoradas)
  - `@dataclass ResultadoPortfolio`: `tabela: pd.DataFrame` (nome, volume, mj_km, origem ∈ {"real", "estimado"}, mj_km_min, mj_km_max, avisos, propulsao, categoria), `media_mj_km: float` (NaN se volume total 0), `por_propulsao: pd.DataFrame` (propulsao, volume, media_mj_km), `por_categoria: pd.DataFrame` (categoria, volume, media_mj_km), `erros: list[str]`
  - `media_portfolio(df: pd.DataFrame, pacote: Pacote | None = None) -> ResultadoPortfolio` — linha com `mj_km_conhecido`: usa o real (min = max = real), especificações opcionais (propulsão/categoria usadas na quebra se presentes, senão "não informado"); sem `mj_km_conhecido`: valida e chama `prever`. Volume tem de ser número > 0. Média = Σ(volume·MJ)/Σvolume. O pacote só é carregado (`carregar()`) se houver linha a estimar — portfólio só com valores reais funciona sem modelo treinado.

- [ ] **Step 1: Testes que falham**

```python
def test_media_ponderada_valores_reais():
    df = pd.DataFrame({"nome": ["A", "B"], "volume": [100, 300], "mj_km_conhecido": [1.5, 2.0]})
    r = media_portfolio(df)
    assert r.media_mj_km == pytest.approx(1.875) and list(r.tabela["origem"]) == ["real", "real"]

def test_mistura_real_e_estimado(pacote_rapido):
    v = carregar_base()[ENTRADAS].iloc[0].drop("bateria_estimada").to_dict()
    df = pd.DataFrame([{"nome": "novo", "volume": 100, **v}, {"nome": "atual", "volume": 100, "mj_km_conhecido": 1.0}])
    r = media_portfolio(df, pacote_rapido)
    est = r.tabela.loc[r.tabela.origem == "estimado", "mj_km"].iloc[0]
    assert r.media_mj_km == pytest.approx((est + 1.0) / 2)

def test_volumes_invalidos():
    df = pd.DataFrame({"nome": ["A", "B", "C"], "volume": [0, -5, None], "mj_km_conhecido": [1.5, 2.0, 1.0]})
    r = media_portfolio(df)
    linhas = [e for e in r.erros if e.startswith("linha ")]
    assert np.isnan(r.media_mj_km) and len(linhas) == 3 and linhas[0].startswith("linha 1:")
    assert any(not e.startswith("linha ") for e in r.erros)   # erro geral: nenhum volume válido

def test_modelo_planilha_ida_e_volta():
    df, erros = ler_planilha(modelo_planilha())
    assert list(df.columns) == COLUNAS_PORTFOLIO and df.empty and erros == []

def test_ler_planilha_colunas_faltando():
    buf = io.BytesIO(); pd.DataFrame({"Nome": ["x"]}).to_excel(buf, index=False)
    _, erros = ler_planilha(buf.getvalue())
    assert erros and "Volume" in erros[0]
```
Run → FAIL.

- [ ] **Step 2: Implementar `portfolio.py`.**
- [ ] **Step 3: Testes passam** — PASS.
- [ ] **Step 4: Commit** — "Portfólio: média ponderada, MJ/km conhecido e planilha modelo".

---

### Task 11: App — página Veículo (formulário, "e se", cenários)

**Files:**
- Create: `app/Inicio.py` (mínimo: título e links das páginas), `app/formulario.py`, `app/pages/1_Veiculo.py`
- Test: `tests/test_app_veiculo.py`

**Interfaces:**
- Consumes: `carregar`, `prever`, `fisica.classe_inercia`, `esquema.ROTULOS/VALORES`, `Pacote.referencias`, `formato.numero`.
- Produces: `app/formulario.formulario_veiculo(chave: str, padrao: dict | None = None) -> dict` (widgets Streamlit; devolve dict com `ENTRADAS` menos `bateria_estimada`) — reutilizado pela Tarefa 12 para o "adicionar veículo".

Comportamento (spec §9.1–9.2):
- Categorias exibidas com rótulo legível (aba "Critérios Classificação" da planilha PED; o código continua sendo o valor).
- Formulário: F0 e F2 obrigatórios, com "típico da categoria: X" (de `referencias`) ao lado; classe de inércia preenchida por `classe_inercia(massa)` e editável; bateria só aparece para propulsão ≠ Combustão (obrigatória); em Elétrico, cilindrada fixa 0, transmissão sugerida "direta", combustível fixo "eletrico".
- Resultado: MJ/km combinado em destaque com `st.metric` (rótulo contendo "MJ/km") e faixa ("1,62 MJ/km (1,54 – 1,70)"), urbano/estrada; km/l dos combustíveis aplicáveis; avisos com `st.warning`.
- "E se": sliders ±20 % (passo 1 %) para massa, F0, F2 e potência; novo MJ/km e diferença para o original (absoluta e %); mexer na massa recalcula a classe de inércia.
- Cenários: botão "Salvar cenário" (`key="salvar_cenario"`) com nome (padrão "Cenário N"), lista em `st.session_state["cenarios"]`; primeiro cenário salvo vira referência "base"; gráfico de barras plotly com barras de erro (faixa) e rótulo de diferença para a base; botão de remover; `st.download_button` exportando xlsx (nome do cenário, entradas alteradas, saídas com faixas). Cenários em `st.session_state`.

- [ ] **Step 1: Testes que falham** (`streamlit.testing.v1.AppTest`, `pacote_rapido` define `EFICIENCIA_MODELO`)

```python
def test_pagina_veiculo_calcula(pacote_rapido):
    at = AppTest.from_file("app/pages/1_Veiculo.py", default_timeout=60).run()
    assert not at.exception
    assert any("MJ/km" in m.label for m in at.metric)

def test_salvar_cenario(pacote_rapido):
    at = AppTest.from_file("app/pages/1_Veiculo.py", default_timeout=60).run()
    at.button(key="salvar_cenario").click().run()
    assert len(at.session_state["cenarios"]) == 1
```
Run: `uv run pytest tests/test_app_veiculo.py -v` → FAIL.

- [ ] **Step 2: Implementar** `formulario.py`, `1_Veiculo.py`, `Inicio.py`. Padrão do formulário: um veículo típico da base (mediana numérica / moda categórica da propulsão Combustão).
- [ ] **Step 3: Testes passam** — PASS. Conferir manualmente: `uv run streamlit run app/Inicio.py`.
- [ ] **Step 4: Commit** — "App: página Veículo com e-se e cenários".

---

### Task 12: App — página Portfólio

**Files:**
- Create: `app/pages/2_Portfolio.py`
- Test: `tests/test_app_portfolio.py`

**Interfaces:**
- Consumes: `portfolio.COLUNAS_PORTFOLIO/modelo_planilha/ler_planilha/media_portfolio`, `formulario.formulario_veiculo`.

Comportamento (spec §9.3): `st.data_editor` com `COLUNAS_PORTFOLIO` (linhas dinâmicas, cabeçalhos = rótulos); botão "Baixar planilha modelo"; `st.file_uploader` xlsx que substitui a tabela; botão "Baixar portfólio" (xlsx da tabela atual); expander "Adicionar veículo pelo formulário". Resultado recalculado a cada edição: média ponderada em destaque, tabelas por propulsão e por categoria, tabela por veículo com origem (real/estimado) e faixa, `st.error` para cada erro de linha e `st.warning` para avisos. Sem km/l médio.

- [ ] **Step 1: Teste que falha**

```python
def test_pagina_portfolio_vazia_e_com_dados(pacote_rapido):
    at = AppTest.from_file("app/pages/2_Portfolio.py", default_timeout=60).run()
    assert not at.exception
```
Run → FAIL.
- [ ] **Step 2: Implementar.**
- [ ] **Step 3: Teste passa** — PASS. Conferir manualmente com a planilha modelo preenchida com 3 veículos (1 com MJ/km conhecido).
- [ ] **Step 4: Commit** — "App: página Portfólio".

---

### Task 13: App — página Comparação e documentação de uso

**Files:**
- Create: `app/pages/3_Comparacao.py`
- Modify: `README.md` (seção "Como usar": `uv sync`, `uv run eficiencia-limpar`, `uv run eficiencia-treinar --tentativas 50` — pode rodar à noite e repetir, os estudos continuam —, `uv run optuna-dashboard sqlite:///optuna/estudos.db`, `uv run streamlit run app/Inicio.py`; atualização anual em 2 comandos)
- Test: `tests/test_app_comparacao.py`

Comportamento (spec §9.4): lê `metricas.json` (variável de ambiente `EFICIENCIA_RELATORIOS`, senão `relatorios/`); tabela candidato × propulsão com erro (%), coluna de monotonia e tentativas; destaque do vencedor; figuras do relatório. Sem métricas: `st.info("Nenhum treino encontrado. Rode: uv run eficiencia-treinar")`.

- [ ] **Step 1: Testes que falham**

```python
def test_comparacao_com_metricas(treino_rapido):
    at = AppTest.from_file("app/pages/3_Comparacao.py", default_timeout=60).run()
    assert not at.exception and at.dataframe

def test_comparacao_sem_metricas(tmp_path, monkeypatch):
    monkeypatch.setenv("EFICIENCIA_RELATORIOS", str(tmp_path))
    at = AppTest.from_file("app/pages/3_Comparacao.py").run()
    assert not at.exception and "eficiencia-treinar" in at.info[0].value
```
- [ ] **Step 2: Implementar** a página e a seção do README.
- [ ] **Step 3: Testes passam** — PASS.
- [ ] **Step 4: Commit** — "App: página de comparação e guia de uso".

---

### Task 14: Treino completo e artefatos versionados

**Files:**
- Create (gerados): `modelos/vencedor/pacote.joblib`, `modelos/vencedor/metadados.json`, `relatorios/metricas.json`, `relatorios/relatorio.md`, `relatorios/figuras/*.png`

- [ ] **Step 1: Rodar o treino completo** (local, Windows; pode levar horas — rodar em segundo plano):
  `uv run eficiencia-treinar --tentativas 50`
  Expected: termina com "vencedor: <nome>" e gera os arquivos acima.
- [ ] **Step 2: Conferir** `relatorios/relatorio.md`: erro do vencedor por propulsão vs. meta de 5 % e vs. referência linear 7,7 %; candidatos reprovados na monotonia listados.
- [ ] **Step 3: Testes** — `uv run pytest -q` → PASS, agora **incluindo** `test_artefato_versionado_monotono` (não mais pulado).
- [ ] **Step 4: Commit + PR** — "Modelo vencedor treinado (50 tentativas) e relatório". No corpo do PR: tabela de erros do relatório. O usuário pode seguir acumulando tentativas à noite e reabrir PRs de atualização do artefato.
