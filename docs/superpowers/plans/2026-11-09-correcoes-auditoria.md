# Correções da auditoria — plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Corrigir os 8 pontos da auditoria de 2026-11-09: faixa de 90 % que cobre menos de 90 %, km/l incoerente com o MJ/km, simulador em degraus, erro alto dos plug-ins (autonomia elétrica), `bateria_estimada` como entrada, portfólio sem faixa, e registro das decisões.

**Architecture:** Mudanças pequenas nos módulos existentes de `src/eficiencia` e `app/`. Duas mudanças de entrada do modelo (entra `autonomia_eletrica_km`, sai `bateria_estimada`) exigem retreino, que o controlador faz depois das tarefas, a partir desta branch.

**Tech Stack:** Python 3.14, pandas, numpy, XGBoost, Streamlit 1.66, pytest, ruff (uv).

**Spec:** a auditoria feita no chat em 2026-11-09 (não há arquivo). Os números que a sustentam estão em Global Constraints e em cada tarefa.

## Global Constraints

- Textos de interface, mensagens, docstrings e comentários em pt-BR.
- Nenhuma dependência nova. `uv run ruff check .` e `uv run ruff format --check .` limpos (linha de 110).
- Rodar só os testes das áreas tocadas durante a tarefa; o conjunto inteiro (`uv run pytest -q`) roda no fim de cada tarefa.
- Commits terminam com a linha `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Nunca adicionar ao git: `.claude/`, `docs/superpowers/plans/2026-11-03-eficiencia-energetica.md`, `data/raw/ciclos/`.
- Não regravar `modelos/vencedor/` nem `relatorios/metricas.json`/`relatorio.md`: o retreino é feito pelo controlador depois.
- Os testes treinam um modelo rápido próprio (`tests/conftest.py`); nenhum teste depende do `pacote.joblib` versionado.

## Review Focus

- Veículo Plug-in ou Elétrico sem autonomia elétrica: o app e o `prever` devem pedir o campo, nunca prever com zero.
- Combustão/Híbrido com autonomia digitada: vira 0, como a bateria da Combustão hoje.
- Planilha de portfólio antiga (sem a coluna de autonomia): plug-ins e elétricos dessa planilha viram linhas com erro claro, não exceção.
- Plug-in: km/l de gasolina/etanol continua vindo do próprio modelo (para plug-in, km/l × MJ/km não é constante: CV de 27–32 %).
- Simulador com todas as barras em 0 %: resultado idêntico ao original (fator exatamente 1).

---

### Task 1: Autonomia elétrica entra, `bateria_estimada` sai (dados, esquema, validação)

**Files:**
- Modify: `src/eficiencia/dados.py`, `src/eficiencia/esquema.py`, `src/eficiencia/entrada.py`, `src/eficiencia/faixa.py`, `src/eficiencia/preprocess.py`, `src/eficiencia/portfolio.py`, `data/processed/pbev_2026_limpo.csv` (regerado), `data/README.md`
- Test: `tests/test_dados.py`, `tests/test_entrada.py`, `tests/test_esquema.py`, `tests/test_portfolio.py` e outros que quebrarem pela mudança de `ENTRADAS`

**Interfaces:**
- Produces: coluna `autonomia_eletrica_km` (float, km) no CSV e em `NUMERICAS`/`ENTRADAS`, rótulo `"Autonomia elétrica (km)"`; `bateria_estimada` fora de `NUMERICAS`/`ENTRADAS` (continua no CSV como marcação de qualidade).

- [ ] **Step 1: Testes que falham**
  - `test_dados.py`: no CSV limpo, `autonomia_eletrica_km == 0` em toda linha Combustão e Híbrido; `> 0` em todo Plug-in com valor; no máximo 6 linhas Plug-in/Elétrico com valor vazio.
  - `test_entrada.py`: Plug-in e Elétrico sem autonomia → erro `"Autonomia elétrica (km) obrigatório"`; Combustão/Híbrido com autonomia 50 → linha válida com autonomia 0; autonomia fora de 0 a 2000 → erro de limite físico.
  - `test_esquema.py`: `"bateria_estimada" not in ENTRADAS` e `"autonomia_eletrica_km" in NUMERICAS`.
- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar**
  - `dados.py`: `COLUNAS[78] = "autonomia_eletrica_km"` (cabeçalho "Autonomia no modo Elétrico Corrigida (km)"); converter com `num`; zerar em Combustão e Híbrido (mesmo padrão da bateria); incluir em `FEATURES`; invariante em `checar`: zero fora de Plug-in/Elétrico e entre 0 e 2000 quando presente. Regerar o CSV com `uv run eficiencia-limpar` e anotar no `data/README.md` a coluna nova e o número de homologações (pode mudar, porque a chave de deduplicação inclui `FEATURES`).
  - `esquema.py`: tirar `bateria_estimada` de `NUMERICAS` e do `ROTULOS` se ficar sem uso; adicionar `autonomia_eletrica_km` a `NUMERICAS` e `ROTULOS`.
  - `entrada.py`: `LIMITES_FISICOS["autonomia_eletrica_km"] = (0, 2000)`; obrigatória para Plug-in e Elétrico; vira 0 para Combustão e Híbrido; apagar os casos especiais de `bateria_estimada` (`OPCIONAIS`, o `continue`, a atribuição `False`).
  - Apagar o tratamento especial de `bateria_estimada` em `faixa.py`, `preprocess.py` (comentário) e `portfolio.py` (`COLUNAS_PORTFOLIO` passa a ser `["nome", "volume", "mj_km_conhecido"] + ENTRADAS`).
- [ ] **Step 4: `uv run pytest -q` passa; ruff limpo.**
- [ ] **Step 5: Commit** `Dados: autonomia elétrica como entrada; bateria_estimada deixa de ser entrada`.

### Task 2: App com o campo de autonomia elétrica

**Files:**
- Modify: `app/formulario.py`, `app/pages/1_Veiculo.py`, `app/visual.py` (só se `NOMES`/`pergunta` precisar do rótulo novo)
- Test: `tests/test_app_veiculo.py`, `tests/test_visual.py`

**Interfaces:**
- Consumes: `autonomia_eletrica_km` em `ENTRADAS` (Task 1); `bateria_estimada` fora de `ENTRADAS`.

- [ ] **Step 1: Testes que falham**
  - `test_app_veiculo.py`: com propulsão Plug-in e autonomia preenchida, a página mostra resultado sem exceção; a explicação SHAP soma `autonomia_eletrica_km` dentro de "Tipo de propulsão" (o campo é zerado para Combustão pelo `validar`, como a bateria).
  - `test_visual.py`: `pergunta` de um nó em `autonomia_eletrica_km` devolve texto com "Autonomia elétrica" e o limite em km com 0 casas.
- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar**
  - `formulario.py`: `number_input` "Autonomia elétrica (km)" ao lado da bateria, `step=1.0`; padrão de Plug-in/Elétrico = mediana da base daquela propulsão; `CAMPOS = ENTRADAS` (sem o filtro de `bateria_estimada`); apagar as outras menções a `bateria_estimada`.
  - `1_Veiculo.py`: `junta` passa a ser `["propulsao", "combustivel", "bateria_kwh", "autonomia_eletrica_km"]`; apagar o comentário sobre `bateria_estimada`; o texto da legenda passa a citar a autonomia ("junta propulsão, combustível, bateria e autonomia elétrica").
- [ ] **Step 4: `uv run pytest -q` passa; ruff limpo.**
- [ ] **Step 5: Commit** `App: campo de autonomia elétrica`.

### Task 3: km/l coerente com o MJ/km (Combustão e Híbrido)

**Files:**
- Modify: `src/eficiencia/fisica.py`, `src/eficiencia/artefatos.py`, `src/eficiencia/treino.py`, `src/eficiencia/prever.py`
- Test: `tests/test_fisica.py`, `tests/test_prever.py`

**Interfaces:**
- Produces: `fatores_kml(df: pd.DataFrame) -> dict[str, tuple[float, float]]` com chaves `"gasolina"`, `"etanol"`, `"diesel"` e valores `(urbano, estrada)` = mediana de `kml_<comb>_<ciclo> * mj_km_<ciclo>` nas linhas Combustão e Híbrido. Substitui `fatores_diesel`. Campo `Pacote.fatores_kml` substitui `Pacote.fatores_diesel`.

- [ ] **Step 1: Testes que falham**
  - `test_fisica.py`: `fatores_kml(carregar_base())["gasolina"]` ≈ `(22.4, 20.5)` (tolerância 0,2), `["etanol"]` ≈ `(15.5, 14.4)`; `["diesel"]` igual ao antigo `fatores_diesel`.
  - `test_prever.py`: para um veículo Combustão flex e um Híbrido do conjunto, `kml_gasolina_urbano * mj_km_urbano == fatores_kml["gasolina"][0]` (rel. 1e-9), idem etanol e estrada; `kml_<comb>_min/_max` = fator / (`mj_km_<ciclo>_max`/`_min`); para um Plug-in, `kml_gasolina_urbano` continua sendo a saída do modelo (`pacote.modelo.predict`).
- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar** — em `prever._registro`, gasolina e etanol de Combustão/Híbrido passam pelo mesmo laço de diesel e `eletrico_eq` (fator / MJ/km, faixa invertida); Plug-in mantém o laço atual que lê o modelo. O treino continua com os 6 alvos (os de km/l servem aos plug-ins). `treino.py` grava `fatores_kml=fatores_kml(df)`.
- [ ] **Step 4: `uv run pytest -q` passa; ruff limpo.**
- [ ] **Step 5: Commit** `Previsão: km/l de combustão e híbrido derivado do MJ/km`.

### Task 4: Faixa de 90 % com erro relativo ao previsto; erros fora da amostra no pacote

**Files:**
- Modify: `src/eficiencia/incerteza.py`, `src/eficiencia/artefatos.py`, `src/eficiencia/treino.py`
- Test: `tests/test_incerteza.py`, `tests/test_treino.py`

**Interfaces:**
- Produces: `calcular_quantis` mede `|real − previsto| / previsto`; `erros_relativos(df, oof) -> dict[str, list[float]]` = `(real − previsto) / previsto` do MJ/km combinado por propulsão (com sinal); campo `Pacote.erros_oof: dict[str, list[float]]`.

- [ ] **Step 1: Testes que falham**
  - `test_incerteza.py`: dados sintéticos com `previsto = 1`, reais `1 + e` para `e` = 0,00 a 0,19 (20 valores, um terço negativos) → a faixa `aplicar(1, q)` contém pelo menos 90 % dos reais; e um caso em que a fórmula antiga (dividir pelo real) não cobre: real = 1,25 com previsto 1,00 sai da faixa com o q antigo e fica dentro com o novo.
  - `test_incerteza.py`: `erros_relativos` devolve, por propulsão, a lista com sinal e o tamanho igual ao número de linhas da propulsão.
  - `test_treino.py`: `treino_rapido.pacote.erros_oof` tem as 4 propulsões e listas não vazias.
- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar** — uma linha em `calcular_quantis` (dividir por `pares["previsto"].abs()`), docstring do módulo atualizada; `erros_relativos` em `incerteza.py`; `treino.py` grava `erros_oof=erros_relativos(df, resultados[vencedor].oof)`.
- [ ] **Step 4: `uv run pytest -q` passa; ruff limpo.**
- [ ] **Step 5: Commit** `Incerteza: faixa de 90 % relativa ao previsto e erros fora da amostra no pacote`.

### Task 5: Simulador sem degraus

**Files:**
- Create: `src/eficiencia/cenario.py`
- Modify: `app/pages/1_Veiculo.py`
- Test: `tests/test_cenario.py`, `tests/test_app_veiculo.py`

**Interfaces:**
- Produces: `fator_suave(modelo, original: dict, alterado: dict) -> float` = média do MJ/km combinado previsto para `alterado` sobre as 11 variações conjuntas `δ ∈ numpy.linspace(-0.05, 0.05, 11)` de `massa_om_kg`, `f0_n`, `f2_n_kmh2` e `potencia_kw` (cada um × (1 + δ), com `classe_inercia_kg` recalculada da massa via `fisica.classe_inercia`), dividida pela mesma média para `original`. Os dicionários já vêm validados (saída de `entrada.validar`).

- [ ] **Step 1: Testes que falham** (`pacote_rapido` da conftest; 3 veículos da base)
  - `fator_suave(m, v, v) == 1.0` exatamente.
  - Varrendo F2 de −20 % a +20 % em passos de 1 %: o fator nunca diminui, e o maior salto entre passos vizinhos é menor ou igual ao maior salto do modelo sem suavização na mesma varredura.
- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar** `cenario.py`; em `1_Veiculo._e_se`, depois de `novo = _prever(...)`, multiplicar `mj_km`, `mj_km_min` e `mj_km_max` de `novo` por `original.mj_km * fator / novo.mj_km` (usando os veículos validados), para que medidor, métrica e cenários salvos usem o valor suavizado. Legenda curta abaixo do medidor: "Variação suavizada: média das previsões em ±5 % em torno de cada ajuste."
- [ ] **Step 4: `uv run pytest -q` passa; ruff limpo.**
- [ ] **Step 5: Commit** `Simulador: variação suavizada, sem degraus`.

### Task 6: Faixa de 90 % na média do portfólio

**Files:**
- Modify: `src/eficiencia/portfolio.py`, `app/pages/2_Portfolio.py`
- Test: `tests/test_portfolio.py`, `tests/test_app_portfolio.py`

**Interfaces:**
- Consumes: `Pacote.erros_oof` (Task 4).
- Produces: `ResultadoPortfolio.media_min` e `.media_max` (float; NaN quando o pacote não tem `erros_oof` ou nenhuma linha foi estimada).

- [ ] **Step 1: Testes que falham**
  - Portfólio só com MJ/km conhecido → `media_min == media_max == media_mj_km`.
  - Portfólio com linhas estimadas → `media_min < media_mj_km < media_max`, e a largura relativa da faixa da média é menor que a média ponderada das larguras relativas das faixas de cada linha (os erros se compensam).
  - Mesmo resultado em duas chamadas (semente fixa).
  - Pacote sem atributo `erros_oof` (`getattr(..., None)`) → `media_min`/`media_max` NaN, sem exceção.
- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar** — reamostragem: 2000 sorteios, `numpy.random.default_rng(0)`; em cada sorteio, cada linha estimada vira `mj_km × (1 + e)`, com `e` sorteado dos `erros_oof` da sua propulsão (sem propulsão informada: todos os erros juntos); linhas com MJ/km conhecido ficam fixas; faixa = percentis 5 e 95 da média ponderada. Comentário `shortcut:` dizendo que os sorteios supõem erros independentes entre carros (a faixa sai estreita se os erros de carros parecidos forem correlacionados) e que, com mais dados, o passo seguinte é sortear por grupo. Na página, a métrica da média ganha a linha "faixa de 90 %: X a Y MJ/km".
- [ ] **Step 4: `uv run pytest -q` passa; ruff limpo.**
- [ ] **Step 5: Commit** `Portfólio: faixa de 90 % na média ponderada`.

### Task 7: Registro do experimento H8

**Files:**
- Modify: `relatorios/experimentos/conclusoes.md`

- [ ] **Step 1:** Seção "H8 — entradas novas (2026-11-09)" com a tabela das sondas: `cv_fixa` do XGBoost com os hiperparâmetros publicados, sementes de dobras 0, 1 e 2, comparação pareada:

  | Variante | Erro MJ/km (média de 3) | Por semente | Plug-in por semente |
  |---|---|---|---|
  | Base | 5,83 % | 5,62 / 5,96 / 5,90 | 11,54 / 14,74 / 14,57 |
  | + autonomia elétrica | 5,67 % | 5,56 / 5,76 / 5,68 | 11,58 / 13,67 / 12,82 |
  | − bateria_estimada | 5,76 % | 5,56 / 5,90 / 5,82 | 11,66 / 14,41 / 14,25 |
  | + energia de tração no ciclo (FTP-75 e HWFET da EPA, só a parte positiva) | 5,80 % | 5,69 / 5,84 / 5,88 | 12,16 / 14,44 / 14,65 |

  Decisão: autonomia entra e `bateria_estimada` sai (melhora em todas as sementes); a energia no ciclo fica de fora (−0,03 ponto, dentro do ruído; no seed 0 piorou). Fonte dos ciclos: `https://www.epa.gov/sites/default/files/2015-10/ftpcol.txt` e `hwycol.txt`. Ressalva: a autonomia é medida no ensaio; no uso, é a autonomia de projeto.
- [ ] **Step 2: Commit** `Experimentos: conclusões do H8 (entradas novas)`.

### Depois das tarefas (controlador)

Retreino completo dos 5 candidatos a partir desta branch (`--tentativas 100`), commit de `modelos/vencedor/` e `relatorios/`, ficha do modelo (`relatorios/ficha_do_modelo.md`: o que faz, erro por propulsão, onde não usar, exemplo do simulador, validação com outro ano pendente até o PBEV 2027) e PR.
