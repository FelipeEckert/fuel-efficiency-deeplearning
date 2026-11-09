# App V6 · Laboratório de cenários — Plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repaginar as quatro páginas do app Streamlit na direção V6 escolhida pelo usuário: tema roxo/ciano, simulador de cenários com medidor, e a árvore real do XGBoost e uma rede neural esquemática animadas.

**Architecture:**
- O tema vem do `.streamlit/config.toml` e a navegação vai para o topo com `st.navigation`.
- Um módulo de domínio, `src/eficiencia/arvore.py`, extrai de um booster do XGBoost o caminho real de um veículo numa árvore.
- Um módulo de visual, `app/visual.py`, gera o HTML animado (CSS puro, como no protótipo), mostrado com `st.components.v1.html`.
- As páginas reaproveitam o que já existe: formulário, `prever`, `explicar`, o "E se" e os cenários. Só mudam o arranjo e o estilo.

**Tech Stack:** Python 3.14, Streamlit 1.66, plotly 7.1, xgboost 3.4, pytest, `streamlit.testing.v1.AppTest`.

**Spec:** a direção visual escolhida é o artboard `project/V6-Laboratorio.dc.html` do artefato https://claude.ai/artifact/4xSFuMLuKENWJoRyRRQCsR. O usuário pediu: "repaginar todas as páginas/interfaces do app… fácil uso… imagem do modelo rodando (árvore de grafos, rede de neurônios)" e escolheu a "V6 Laboratório".

## Global Constraints

- Texto da interface, docstrings e comentários em pt-BR. Números com vírgula decimal (`eficiencia.formato.numero` ou o `_fixo` da página Veículo).
- Nenhuma dependência nova. Plotly, Streamlit e xgboost já estão instalados.
- **Cores:**
  - primária `#7C3AED`;
  - ciano `#0891B2`;
  - tinta `#1E1B4B`;
  - fundo `#FAF5FF`;
  - cartão `#FFFFFF`;
  - borda `#DDD6FE`;
  - painel escuro `#1E1B4B` com destaque `#22D3EE` e lilás `#A78BFA`.
  - "Aumenta o consumo" usa `#7C3AED` e "reduz" usa `#0891B2`, nos gráficos e nas barras.
- **Fonte:** Inter (Google Fonts) pelo tema.
- **Animações:** só `transform`/`opacity`, com curvas de easing como no protótipo. Há botão de pausar e `@media (prefers-reduced-motion: reduce)` desliga tudo.
- **Modelos sem árvore:** só o XGBoost tem árvore. Com outro vencedor (o `pacote_rapido` dos testes pode ser `linear`), a aba da árvore mostra uma legenda e nada quebra.
- **Rede neural:** é um esquema ilustrativo e o texto diz isso. O modelo publicado é `pacote.algoritmo`.
- **Ruff:** `line-length = 110`. Não tocar em `.claude/` nem em `docs/superpowers/plans/2026-11-03-eficiencia-energetica.md`.

## Review Focus

1. **Veículo elétrico:** o caminho na árvore pode passar por perguntas de categoria (`propulsao=Elétrico`). Elas precisam virar texto legível ("Propulsão é Elétrico?"), nunca `propulsao=Elétrico < 1`. O teste fica na Tarefa 2.
2. **Vencedor não-XGBoost:** a página Veículo roda sem exceção e mostra a legenda no lugar da árvore. O teste fica na Tarefa 4.
3. **Escala do medidor:** um elétrico (~0,5 MJ/km) e um carro grande (~3 MJ/km) precisam caber no medidor. A faixa do eixo é relativa ao valor original (`[0,5 × original, 1,5 × original]`). O teste fica na Tarefa 4.
4. **Botões prontos:** um clique em "Aliviar 10 %" muda os controles do simulador, e "Voltar ao original" zera todos. O teste fica na Tarefa 4.
5. **Tela estreita:** o desenho da árvore e da rede (1000 px) fica numa caixa com rolagem horizontal própria, sem rolar a página. O teste fica na Tarefa 3 (`overflow-x:auto` no HTML).

---

### Task 1: Tema, navegação no topo e Início

**Files:**
- Create: `.streamlit/config.toml`
- Create: `app/pages/0_Inicio.py` (o conteúdo que hoje está em `app/Inicio.py`, repaginado)
- Modify: `app/Inicio.py` (vira o roteador)
- Test: `tests/test_app_inicio.py`

**Interfaces:**
- Produces: `app/Inicio.py` define as páginas com `st.navigation([...], position="top")`, em `st.Page` com títulos "Início", "Veículo", "Portfólio" e "Comparação", e `st.set_page_config(layout="wide", page_title="Eficiência energética")`. O comando `uv run streamlit run app/Inicio.py` continua igual.

- [ ] **Step 1: Teste que falha** em `tests/test_app_inicio.py`:

```python
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"


def test_roteador_abre_inicio_sem_erro():
    at = AppTest.from_file(str(APP / "Inicio.py"), default_timeout=60).run()
    assert not at.exception
    assert any("Eficiência energética" in t.value for t in at.title)


def test_tema_roxo():
    toml = (Path(__file__).resolve().parents[1] / ".streamlit/config.toml").read_text(encoding="utf-8")
    assert 'primaryColor = "#7C3AED"' in toml and "Inter" in toml
```

- [ ] **Step 2:** Rode `uv run pytest tests/test_app_inicio.py -v`. O esperado é FALHAR: falta `config.toml`, e o título atual pode passar ou não.
- [ ] **Step 3: Implementar.**
  - **Tema** em `.streamlit/config.toml` (seção `[theme]`):
    - `primaryColor = "#7C3AED"`, `backgroundColor = "#FAF5FF"`, `secondaryBackgroundColor = "#F5F3FF"`, `textColor = "#1E1B4B"`;
    - fonte Inter pela sintaxe de fonte do Streamlit 1.66, `font = "Inter:https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"`. Confira na documentação/changelog da 1.66 se a forma é essa. Se não for, use a forma que a versão aceita.
    - opcional: `baseRadius = "large"`.
  - **Roteador:** `app/Inicio.py` passa a ser só o roteador.
  - **Página inicial:** `app/pages/0_Inicio.py` tem:
    - o título "Eficiência energética de veículos";
    - uma linha de descrição;
    - três cartões, `st.container(border=True)` em 3 colunas, cada um com `st.page_link` para Veículo, Portfólio e Comparação e uma frase sobre o que a página faz.
- [ ] **Step 4:** Rode `uv run pytest tests/test_app_inicio.py tests/test_app_veiculo.py tests/test_app_portfolio.py tests/test_app_comparacao.py -q`. O esperado é que tudo PASSE: as páginas continuam rodando sozinhas no AppTest.
- [ ] **Step 5:** Commit: `git add .streamlit/config.toml app/Inicio.py app/pages/0_Inicio.py tests/test_app_inicio.py && git commit -m "App: tema roxo/ciano, navegação no topo e Início repaginado"`.

### Task 2: Caminho real de um veículo numa árvore do XGBoost

**Files:**
- Create: `src/eficiencia/arvore.py`
- Test: `tests/test_arvore.py`

**Interfaces:**
- Consumes:
  - `explicavel(modelo)` (`eficiencia.explicar`);
  - `modelo.prep` (`Preprocessador`, com `.nomes`, `.transform`, `._media`, `._desvio`) e `modelo.estimadores[alvo]` (`XGBRegressor`);
  - `validar` (`eficiencia.entrada`), que completa `bateria_estimada`.
- Produces: `caminho_arvore(modelo, veiculo: dict, alvo: str, indice: int = 0) -> list[dict]`.
  - **O que entra:** todos os nós dos níveis 0–2 da árvore `indice`, mais os nós do caminho do veículo nos níveis abaixo. A raiz vem primeiro e o resto em ordem de nível.
  - **Campos de cada nó:** `id` (int, o `Node` do XGBoost), `pai` (int ou None), `nivel` (int), `folha` (bool), `coluna` (str: a coluna de `ENTRADAS`; None na folha), `categoria` (str: o valor da categórica quando o nó testa um one-hot; senão None), `limite` (float: o corte em unidades originais nas numéricas; None nas categóricas e na folha), `valor` (float: o valor da folha em log; None nos internos), `carros` (int: o `Cover` arredondado), `no_caminho` (bool) e `sim` (bool ou None: num nó interno do caminho, True quando o veículo vai para o filho "Yes").
  - **Regra do XGBoost:** vai para "Yes" quando `x < Split`, com `x` já transformado pelo `prep`.
  - **Corte em unidades originais:** `Split × desvio + média` da coluna. Numa categórica, "Yes" (`x < Split`) significa que o veículo **não** é daquela categoria.

- [ ] **Step 1: Testes que falham** em `tests/test_arvore.py`:

```python
import numpy as np
import pandas as pd
import xgboost

from eficiencia.arvore import caminho_arvore
from eficiencia.entrada import validar
from eficiencia.esquema import ALVOS, ENTRADAS, carregar_base
from eficiencia.modelos import CANDIDATOS


def _modelo():
    df = carregar_base()
    m = CANDIDATOS["xgboost"](**CANDIDATOS["xgboost"].PARAMS_RAPIDOS).fit(df[ENTRADAS], df[ALVOS])
    return m, df


def _veiculo(df, propulsao):
    linha = df[df.propulsao == propulsao][ENTRADAS].iloc[0]
    return validar(pd.DataFrame([linha.to_dict()]))[0].iloc[0].to_dict()


def test_caminho_termina_na_folha_que_o_xgboost_usa():
    m, df = _modelo()
    for propulsao in ("Combustão", "Elétrico"):
        v = _veiculo(df, propulsao)
        nos = caminho_arvore(m, v, "mj_km_urbano")
        caminho = [n for n in nos if n["no_caminho"]]
        assert caminho[0]["nivel"] == 0 and caminho[-1]["folha"]
        Xt = m.prep.transform(pd.DataFrame([v]))
        folha = m.estimadores["mj_km_urbano"].get_booster().predict(xgboost.DMatrix(Xt), pred_leaf=True)[0][0]
        assert caminho[-1]["id"] == int(folha)


def test_niveis_de_cima_completos_e_perguntas_legiveis():
    m, df = _modelo()
    nos = caminho_arvore(m, _veiculo(df, "Elétrico"), "mj_km_urbano")
    assert sum(n["nivel"] == 0 for n in nos) == 1
    for n in nos:
        if not n["folha"]:
            assert n["coluna"] in ENTRADAS and "=" not in n["coluna"]
            assert (n["categoria"] is None) != (n["limite"] is None)
```

- [ ] **Step 2:** Rode `uv run pytest tests/test_arvore.py -v`. O esperado é FALHAR com `ModuleNotFoundError`.
- [ ] **Step 3: Implementar** `caminho_arvore` em `src/eficiencia/arvore.py`.
  - Leia a árvore com `get_booster().trees_to_dataframe()`, filtrada por `Tree == indice`. Os ids `"0-3"` viram `int` pelo número depois do hífen.
  - Para saber o nome de `fN`, use `modelo.prep.nomes[N]`.
  - Para cada nó do caminho, compare com o valor do veículo transformado (`Xt[0][N]`).
  - Use uma docstring de uma linha.
- [ ] **Step 4:** Rode `uv run pytest tests/test_arvore.py -v`. O esperado é que os 2 testes PASSEM, com o ruff limpo.
- [ ] **Step 5:** Commit: `git add src/eficiencia/arvore.py tests/test_arvore.py && git commit -m "Caminho real de um veículo numa árvore do XGBoost"`.

### Task 3: HTML animado da árvore e da rede

**Files:**
- Create: `app/visual.py`
- Test: `tests/test_visual.py`

**Interfaces:**
- Consumes: a lista de nós de `caminho_arvore` (Tarefa 2) e `ROTULOS` (`eficiencia.esquema`).
- Produces:
  - `CORES: dict[str, str]`, com as cores das Global Constraints e as chaves `primaria`, `ciano`, `tinta`, `fundo`, `borda`, `escuro`, `destaque` e `lilas`;
  - `pergunta(no: dict) -> str`, o texto de um nó interno;
  - `html_arvore(nos: list[dict]) -> str`;
  - `html_rede() -> str`.
  - Os dois `html_*` devolvem um documento HTML completo para `st.components.v1.html`.

**Texto de `pergunta`:**
- numérica: `f"{ROTULOS[coluna]} < {numero(limite, casas)}?"`, com `casas` = 0 para kg, kW e cm³, 1 para F0, 4 para F2 e 0 para as demais;
- categórica: `f"{ROTULOS[coluna]} é {categoria}?"`.

**Desenho** (porte do protótipo V6):
- **Árvore:** caixa de 1000 × 400 px dentro de `<div style="overflow-x:auto">`.
  - Os níveis 0–2 seguem o layout binário: `x = (posição + 0,5) / 2^nível × 1000` e `y = 30 + 70 × nível`.
  - Os nós do caminho abaixo do nível 2 descem numa coluna, no mesmo `x` do nó de nível 2 do caminho.
  - Arestas: `div` girados (`transform: rotate`), feitos por uma função `_linha(x1, y1, x2, y2)`.
  - **Arestas do caminho:** cada uma leva uma partícula `.part` que anda do começo ao fim (keyframes `translateX(var(--L))`, ciclo `4.8s`, atraso `(k−1) × 0.8s`). Os nós do caminho ganham um anel `.anel` que acende quando a partícula chega.
  - **Folha:** pílula lilás com `valor` em log e `carros`.
  - **Nós fora do caminho:** apagados (`#312E81`).
- **Rede:** camadas 6–7–7–2, caixa de 1000 × 340, rótulos "48 entradas", "oculta 1", "oculta 2" e "urbano · estrada".
  - Uma partícula por aresta, em onda por camada: atraso `k × 0.55s` mais um deslocamento pequeno, ciclo `2.6s`.
  - Neurônios com pulso `scale`.
- **Os dois:** fundo `#1E1B4B`, texto branco, e um botão "Pausar animação" que alterna a classe `pausado` (`animation-play-state: paused`) com um `onclick` de uma linha. Incluem `@media (prefers-reduced-motion: reduce)` desligando as animações.

- [ ] **Step 1: Testes que falham** em `tests/test_visual.py`. `sys.path` recebe a pasta `app/`, como em `tests/test_app_veiculo.py`.

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from visual import html_arvore, html_rede, pergunta  # noqa: E402

NOS = [
    {"id": 0, "pai": None, "nivel": 0, "folha": False, "coluna": "bateria_kwh", "categoria": None,
     "limite": 12.0, "valor": None, "carros": 320, "no_caminho": True, "sim": True},
    {"id": 1, "pai": 0, "nivel": 1, "folha": False, "coluna": "propulsao", "categoria": "Elétrico",
     "limite": None, "valor": None, "carros": 219, "no_caminho": True, "sim": False},
    {"id": 2, "pai": 0, "nivel": 1, "folha": True, "coluna": None, "categoria": None,
     "limite": None, "valor": 0.01, "carros": 101, "no_caminho": False, "sim": None},
    {"id": 4, "pai": 1, "nivel": 2, "folha": True, "coluna": None, "categoria": None,
     "limite": None, "valor": 0.0022, "carros": 32, "no_caminho": True, "sim": None},
]


def test_pergunta_legivel():
    assert pergunta(NOS[0]) == "Bateria (kWh) < 12?"
    assert pergunta(NOS[1]) == "Propulsão é Elétrico?"


def test_html_arvore_anima_so_o_caminho():
    h = html_arvore(NOS)
    assert h.count('class="part"') == 2  # duas arestas no caminho
    assert "Propulsão é Elétrico?" in h and "32 carros" in h
    assert "prefers-reduced-motion" in h and "overflow-x:auto" in h


def test_html_rede_uma_particula_por_aresta():
    h = html_rede()
    assert h.count('class="part"') == 6 * 7 + 7 * 7 + 7 * 2
    assert "esquema" in h.lower() and "prefers-reduced-motion" in h
```

  O rótulo exato de `bateria_kwh` vem de `ROTULOS`. Se ele não for "Bateria (kWh)", ajuste o texto esperado do teste ao rótulo real, e só ele.

- [ ] **Step 2:** Rode `uv run pytest tests/test_visual.py -v`. O esperado é FALHAR com `ModuleNotFoundError`.
- [ ] **Step 3:** Implemente `app/visual.py` como descrito. O CSS das animações pode ser copiado do protótipo V6, dos blocos `@keyframes viaja`, `anel` e `neur`.
- [ ] **Step 4:** Rode `uv run pytest tests/test_visual.py -v`. O esperado é que PASSE, com o ruff limpo.
- [ ] **Step 5:** Commit: `git add app/visual.py tests/test_visual.py && git commit -m "Visual animado da árvore real e da rede neural (HTML)"`.

### Task 4: Página Veículo no layout V6

**Files:**
- Modify: `app/pages/1_Veiculo.py`
- Test: `tests/test_app_veiculo.py`

**Interfaces:**
- Consumes: `caminho_arvore` (Tarefa 2), `CORES`, `html_arvore` e `html_rede` (Tarefa 3), e as funções que a página já tem (`_prever`, `_resultado`, `_explicacao`, `_e_se`, `_cenarios`, `_grafico`).
- Produces: as chaves de widget novas, para testes:
  - botões `preset_aliviar`, `preset_pneus`, `preset_aero`, `preset_pacote` e `zerar_simulador`;
  - os sliders mantêm as chaves que já existem (`es_massa`, `es_f0`, `es_f2`, `es_potencia`).

**Layout, de cima para baixo:**
1. **Ficha:** o formulário inteiro (`formulario_veiculo("veiculo")`) dentro de `st.expander("Ficha técnica do veículo", expanded=False)`. Acima dele, uma legenda curta com o resumo: propulsão · combustível · transmissão · massa · F0 · F2.
2. **Simulador:** duas colunas.
   - **Esquerda:** `st.container(border=True)` com o subtítulo "Simulador de cenários".
     - Quatro botões prontos numa linha. Cada um usa `on_click` para gravar os sliders no `session_state`:
       - "Aliviar 10 %": massa −10;
       - "Pneus verdes −15 % F0": F0 −15;
       - "Aero −10 % F2": F2 −10;
       - "Pacote −10 % em tudo": massa, F0 e F2 em −10.
     - Os sliders do `_e_se` atual, com `step` 5 e os mesmos limites.
     - "Salvar cenário" (já existe) e "Voltar ao original" (zera os quatro sliders).
   - **Direita:** um medidor `go.Indicator(mode="gauge+number+delta")` com o MJ/km combinado do veículo alterado.
     - O eixo vai de `0,5 × original` a `1,5 × original`. A faixa de 90 % (`mj_km_min`–`mj_km_max` do alterado) é um `step` lilás `#DDD6FE`, e a barra é `#7C3AED`.
     - O delta usa `reference = original.mj_km`, com `decreasing` em `#0891B2` e `increasing` em `#7C3AED`. Use `valueformat=".2f"` e `separators=",."`.
     - Abaixo do medidor fica o `st.metric("MJ/km com as mudanças", …)`, que já existe e é usado nos testes, e depois os avisos.
3. **Resultado do original:** `_resultado(original)`, como hoje.
4. **Explicação:** `_explicacao(pacote, veiculo)`, com as cores trocadas para `CORES["primaria"]` (aumenta) e `CORES["ciano"]` (reduz).
5. **"O modelo por dentro":** `st.tabs(["Árvore do XGBoost", "Rede neural"])`.
   - **Árvore:** se `explicavel(pacote.modelo)`, mostra `components.html(html_arvore(caminho_arvore(pacote.modelo, normalizado, f"mj_km_{ciclo}")), height=470)`. O `ciclo` é o do radio `shap_ciclo` e o `normalizado` é o veículo validado, os mesmos de `_explicacao`. Para os dois blocos usarem os mesmos valores, faça `_explicacao` devolver `(ciclo, normalizado)` ou None. Se não for XGBoost, mostra uma legenda.
   - **Rede:** `components.html(html_rede(), height=420)`.
6. **Cenários:** `_cenarios()`. As cores do gráfico passam a ser base `#94A3B8` e demais `#7C3AED`.

**Ordem do código:** o resultado do veículo alterado depende do original. Calcule `original` antes de desenhar o simulador e use `st.container()` para pôr cada coisa no lugar certo. O `_e_se` hoje desenha subtítulo, sliders e métrica juntos. Divida-o em duas funções, uma que desenha os sliders e devolve o veículo alterado e outra que mostra o resultado, sem duplicar a lógica.

- [ ] **Step 1: Testes que falham** em `tests/test_app_veiculo.py`. Reaproveite `_pagina()` e `_metrica()`:

```python
def test_preset_e_voltar_ao_original(pacote_rapido):
    at = _pagina()
    at.button(key="preset_pacote").click().run()
    assert (at.session_state["es_massa"], at.session_state["es_f0"], at.session_state["es_f2"]) == (-10, -10, -10)
    assert _metrica(at, "MJ/km com as mudanças").delta  # mudou em relação ao original
    at.button(key="zerar_simulador").click().run()
    assert at.session_state["es_massa"] == at.session_state["es_f0"] == at.session_state["es_f2"] == 0


def test_modelo_por_dentro_e_medidor(pacote_rapido):
    at = _pagina()
    assert not at.exception
    assert [t.label for t in at.tabs][-2:] == ["Árvore do XGBoost", "Rede neural"]


def test_eletrico_roda_simulador(pacote_rapido):
    at = _pagina()
    at.selectbox(key="veiculo_propulsao").set_value("Elétrico").run()
    at.number_input(key="veiculo_bateria_kwh").set_value(60.0).run()
    at.button(key="preset_aliviar").click().run()
    assert not at.exception
```

  Os testes atuais que leem gráficos por posição, como `plotly_chart[-1]` em `test_dois_cenarios_base_e_diferenca`, devem continuar mirando o gráfico de cenários. Ajuste o índice ou a busca se o medidor mudar a ordem, e diga no relatório o que mudou. Os outros testes da página devem continuar passando.

- [ ] **Step 2:** Rode `uv run pytest tests/test_app_veiculo.py -v`. O esperado é que os 3 testes novos FALHEM.
- [ ] **Step 3:** Implemente o layout acima em `app/pages/1_Veiculo.py`.
- [ ] **Step 4:** Rode `uv run pytest tests/test_app_veiculo.py tests/test_arvore.py tests/test_visual.py -v` e depois a suíte inteira uma vez: `uv run pytest -q`. O esperado é que tudo PASSE, com o ruff limpo.
- [ ] **Step 5:** Commit: `git add app/pages/1_Veiculo.py tests/test_app_veiculo.py && git commit -m "Página Veículo no layout V6: simulador, medidor e modelo por dentro animado"`.

### Task 5: Portfólio e Comparação no mesmo estilo

**Files:**
- Modify: `app/pages/2_Portfolio.py` e `app/pages/3_Comparacao.py`
- Test: `tests/test_app_comparacao.py` e `tests/test_app_portfolio.py`

**Interfaces:**
- Consumes: `CORES` (Tarefa 3). As páginas já põem `app/` no `sys.path` ou devem pôr, como faz `1_Veiculo.py`.

**O que muda:**
- **Comparação:**
  - Acima da tabela entra um gráfico de barras horizontais (plotly) do "Erro geral (%)" por candidato, do menor para o maior. O vencedor vai em `CORES["primaria"]` e os outros em `#C4B5FD`, com rótulo de valor `%{x:.1f} %` e `separators=",."`.
  - O aviso de vencedor e a tabela ficam num `st.container(border=True)`.
- **Portfólio:**
  - O topo (métrica do MJ/km médio, legenda de cobertura e as duas quebras) fica dentro de `st.container(border=True)`.
  - A quebra "Por propulsão" ganha, ao lado da tabela, um gráfico de barras (plotly) do MJ/km por propulsão em `CORES["primaria"]`.
  - Nada muda no cálculo.

- [ ] **Step 1: Testes que falham:**

```python
# tests/test_app_comparacao.py — novo teste, usando o mesmo arranjo dos testes que já existem nesse arquivo
def test_grafico_de_erro_por_candidato(...):  # mesma fixture/arranjo do teste de tabela já existente
    ...
    assert len(at.get("plotly_chart")) >= 1
```

  Leia o arquivo e escreva o teste no padrão dele: as mesmas fixtures e o mesmo jeito de montar as métricas. A asserção é a de cima. No Portfólio, acrescente uma asserção igual ao teste que já calcula um portfólio com veículos: pelo menos um gráfico plotly na tela.

- [ ] **Step 2:** Rode `uv run pytest tests/test_app_comparacao.py tests/test_app_portfolio.py -v`. O esperado é que os testes novos FALHEM.
- [ ] **Step 3:** Implemente o que está descrito acima.
- [ ] **Step 4:** Rode `uv run pytest -q`, a suíte inteira. O esperado é que tudo PASSE, com o ruff limpo.
- [ ] **Step 5:** Commit: `git add app/pages/2_Portfolio.py app/pages/3_Comparacao.py tests/test_app_comparacao.py tests/test_app_portfolio.py && git commit -m "Portfólio e Comparação no estilo V6, com gráficos"`.
