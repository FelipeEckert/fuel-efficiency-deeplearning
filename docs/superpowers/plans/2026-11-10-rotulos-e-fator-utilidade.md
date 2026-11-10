# Rótulos corrigidos e fator de utilidade dos plug-ins — plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Corrigir o rótulo de propulsão errado na limpeza dos dados (Toyota RAV4 S/SX Hybrid cadastrado como "Combustão") e testar uma entrada derivada nova para os plug-ins: a fração da distância rodada a combustão, `1 − UF(autonomia)`. A entrada nova só fica se passar na sonda pareada; depois, retreino e documentação.

**Architecture:**
- **Rótulo:** a correção é uma regra na limpeza (`dados.py`): "Combustão" com HYBRID, HÍBRIDO ou HEV no nome vira "Híbrido". Um invariante novo em `checar` impede que o erro volte.
- **Entrada derivada:** é calculada no `Preprocessador` a partir de `propulsao` e `autonomia_eletrica_km`. O formulário do app e as planilhas de portfólio não mudam.
- **Monotonia:** a entrada é monótona crescente no consumo, como massa, F0 e F2. Mais distância a combustão nunca baixa o MJ/km. Por isso ela entra em `MONOTONAS` e usa o mesmo mecanismo de sinais de todos os candidatos.
- **Por que a fração e não o UF:** é o mesmo dado com o sinal invertido. Para o XGBoost, o UF sozinho seria redundante com a autonomia (mesma ordem dentro dos plug-ins). O ganho esperado vem de unificar as propulsões numa escala física (combustão e híbrido = 1, plug-in = 1 − UF, elétrico = 0) e da restrição de monotonia.

**Tech Stack:** Python 3.14, pandas, numpy, XGBoost, Streamlit, pytest, ruff (uv).

**Spec:** a avaliação feita no chat em 2026-11-10 (não há arquivo). Os números que a sustentam estão abaixo e em cada tarefa.
- **RAV4:** "I/TOYOTA/RAV4 SX HYBRID" e "I/TOYOTA/RAV4 S HYBRID" são linhas 741 e 742 da aba "Dados de Entrada". A propulsão está como "Combustão", a bateria vazia e o MJ/km é 1,4239. Erram 45 % fora da amostra. O mesmo carro aparece como "Híbrido" nas linhas 708 e 709 ("RAV4H SX/S AWD CNT", 1,4337).
- **Plug-ins:** o viés fora da amostra é de +5,7 % (o modelo superestima). O erro absoluto tem correlação −0,59 com a autonomia e −0,62 com a bateria. Há 50 plug-ins, com autonomia de 9 a 128 km (mediana 61).
- **Outros suspeitos, sem prova, que ficam como estão:**
  - **AMG GLC63S E-Performance:** 2,985 MJ/km, bateria vazia na planilha, 18 km de autonomia. O C63S E-Performance, com o mesmo trem de força, tem 1,81. Não há esse carro na planilha "Especificação Técnica" para conferir.
  - **Lexus UX300h (1,29):** confirmado pela "Especificação Técnica".
  - **BYD King GL DM (0,532):** coerente com o GS DM (0,490, mais autonomia).
- **Coerência interna da base:** km/l × MJ/km está a ±6 % da mediana em todas as linhas de combustão e híbrido. Não há erro de digitação detectável por essa via.

## Global Constraints

- **Idioma:** textos que o usuário vê (rótulos, mensagens, README, relatórios) em inglês. Identificadores, comentários, docstrings e mensagens de commit em português, como o código atual.
- **Ferramentas:**
  - nenhuma dependência nova;
  - `uv run ruff check .` e `uv run ruff format --check .` limpos (linha de 110);
  - `uv run pytest -q` passando no fim de cada tarefa.
- **Commits:** terminam com a linha `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Nunca adicionar ao git:** `.claude/`, `docs/superpowers/plans/2026-11-03-eficiencia-energetica.md`, `data/raw/`, `data/processed/`. Os dados não são versionados, e o CI usa a tabela sintética de `tests/dados_sinteticos.py`.
- **Pacote versionado:** `modelos/vencedor/` e `relatorios/` só mudam na Tarefa 6, pelo controlador, depois do retreino. Até lá o `pacote.joblib` versionado (sem a entrada nova) tem de continuar carregando e prevendo igual.
- **Branch e PR:** trabalhar na branch `rotulos-fator-utilidade`, criada a partir de `origin/main`. O PR é aberto pelo controlador e mesclado pelo usuário: nunca mesclar nem commitar na `main`.
- **Processos longos** (sondas de horas, treino de ~21 h): rodam em segundo plano, lançados via WMI (`Invoke-CimMethod -ClassName Win32_Process -MethodName Create`) para sobreviver ao fim da sessão.
- **Fator de utilidade:** curva "fleet" do WLTP, Reg. (UE) 2017/1151, com dn = 800 km:
  - `UF(d) = 1 − exp(−Σᵢ aᵢ·(d/800)ⁱ)`, para i = 1..10;
  - `a = [26.25, −38.94, −631.05, 5964.83, −25095, 60380.2, −87517, 75513.8, −35749, 7154.94]`;
  - `d` é limitado a [0, 800] km.

## Review Focus

- **Plug-in sem autonomia na base** (NaN): a fração fica NaN e o `Preprocessador` a preenche com a média do treino, como as outras numéricas. Nunca pode virar erro ou `inf`. O teste fica na Tarefa 2.
- **Pacote antigo carregado com o código novo:** o `Preprocessador` salvo antes desta branch não conhece a fração. O `transform` dele tem de devolver o mesmo número de colunas de antes, senão o app quebra até o retreino. O teste fica na Tarefa 2.
- **Autonomia muito alta** (até 2.000 km, o limite de `entrada.py`): o polinômio do UF não pode estourar. Com o limite de 800 km, UF(2000) = UF(800). O teste fica na Tarefa 1.
- **A árvore do app passa pela fração:** `visual.py` busca `ROTULOS[coluna]`, então a fração precisa de rótulo, senão dá `KeyError`. O teste fica na Tarefa 2.
- **Explicação "Why this value?"** de um plug-in: as contribuições têm de fechar com a previsão, contando a fração dentro de "Powertrain type". O teste fica na Tarefa 2.

---

### Task 1: Fator de utilidade e fração a combustão em `fisica.py`

**Files:**
- Modify: `src/eficiencia/fisica.py`
- Test: `tests/test_fisica.py`

**Interfaces:**
- Produces:
  - `fator_utilidade(autonomia_km)`: aceita float ou `pd.Series` e devolve o mesmo tipo.
  - `fracao_combustao(propulsao: pd.Series, autonomia_km: pd.Series) -> pd.Series`, com valores em [0, 1] e NaN onde a autonomia do plug-in é NaN.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pandas as pd
from eficiencia.fisica import fator_utilidade, fracao_combustao

def test_fator_utilidade_wltp():
    assert fator_utilidade(0.0) == 0.0
    assert np.isclose(fator_utilidade(40.0), 0.689, atol=1e-3)
    assert np.isclose(fator_utilidade(100.0), 0.902, atol=1e-3)
    d = np.arange(0, 801, 1.0)
    assert (np.diff(fator_utilidade(pd.Series(d))) >= 0).all()  # nunca cai com mais autonomia
    assert fator_utilidade(2000.0) == fator_utilidade(800.0)     # acima de dn não estoura

def test_fracao_combustao_por_propulsao():
    prop = pd.Series(["Combustão", "Híbrido", "Plug-in", "Elétrico", "Plug-in"])
    aut = pd.Series([0.0, 0.0, 40.0, 300.0, np.nan])
    f = fracao_combustao(prop, aut)
    assert f.iloc[:2].tolist() == [1.0, 1.0]
    assert np.isclose(f.iloc[2], 1 - 0.689, atol=1e-3)
    assert f.iloc[3] == 0.0
    assert np.isnan(f.iloc[4])
```

- [ ] **Step 2:** `uv run pytest tests/test_fisica.py -q`. Esperado: FAIL com `ImportError`.
- [ ] **Step 3: Implement both functions in `fisica.py`.**
  - Use os coeficientes e o limite de `Global Constraints`, numa constante `_COEF_UF_WLTP`.
  - Calcule `fracao_combustao` com `np.select` ou `where`: Combustão e Híbrido = 1, Elétrico = 0, Plug-in = `1 − fator_utilidade(autonomia)`.
  - Atualize a docstring do módulo para mencionar o fator de utilidade.
- [ ] **Step 4:** `uv run pytest tests/test_fisica.py -q`. Esperado: PASS.
- [ ] **Step 5:** commit com a mensagem `Física: fator de utilidade WLTP e fração da distância a combustão`.

### Task 2: A fração como entrada derivada dos modelos

**Files:**
- Modify: `src/eficiencia/esquema.py`: `DERIVADAS = ["fracao_combustao"]`; `"fracao_combustao"` no fim de `MONOTONAS`; rótulo em `ROTULOS`.
- Modify: `src/eficiencia/preprocess.py`: `_numericas` acrescenta as derivadas; `transform` usa só as colunas vistas no `fit`.
- Modify: `src/eficiencia/monotonia.py`: o laço de `checar_monotonia` percorre só as colunas de `MONOTONAS` que estão em `ENTRADAS`, porque a derivada não pode ser multiplicada diretamente.
- Modify: `src/eficiencia/explicar.py`: `reindex(ENTRADAS + DERIVADAS, ...)`.
- Modify: `app/pages/1_Veiculo.py`: `"fracao_combustao"` entra na lista `junta`, que soma em "Powertrain type".
- Modify: `app/visual.py`: `_CASAS["fracao_combustao"] = 2`. Sem isso, a pergunta do nó arredonda o limite para 0 ou 1.
- Test: `tests/test_preprocess.py`, `tests/test_explicar.py`, `tests/test_visual.py`.

**Interfaces:**
- Consumes: `fracao_combustao` da Tarefa 1.
- Produces:
  - `eficiencia.esquema.DERIVADAS: list[str]`;
  - `Preprocessador.nomes`, que termina com `"fracao_combustao"` quando `DERIVADAS` a contém;
  - `ROTULOS["fracao_combustao"] = "Share of distance on fuel"`.
- **Para a sonda:** o `preprocess` lê `DERIVADAS` do próprio módulo na hora do `fit`. Assim, a Tarefa 3 desliga a entrada com `monkeypatch`/atribuição em `eficiencia.preprocess.DERIVADAS = []`. Use `from eficiencia.esquema import DERIVADAS` no topo de `preprocess.py` e referencie o nome do módulo.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_preprocess.py
import numpy as np

import eficiencia.preprocess as pp
from eficiencia.esquema import ENTRADAS, carregar_base
from eficiencia.preprocess import Preprocessador

def test_fracao_entra_como_numerica():
    df = carregar_base()
    p = Preprocessador().fit(df[ENTRADAS])
    assert p.nomes[-1] == "fracao_combustao"
    assert p.transform(df[ENTRADAS]).shape[1] == len(p.nomes)

def test_plugin_sem_autonomia_nao_quebra():
    df = carregar_base()
    X = df[ENTRADAS].copy()
    linha = X.index[X["propulsao"].eq("Plug-in")][0]
    X.loc[linha, "autonomia_eletrica_km"] = float("nan")
    Z = Preprocessador().fit(df[ENTRADAS]).transform(X)
    assert np.isfinite(Z).all()

def test_preprocessador_antigo_mantem_colunas(monkeypatch):
    df = carregar_base()
    monkeypatch.setattr(pp, "DERIVADAS", [])
    antigo = Preprocessador().fit(df[ENTRADAS])        # como o pacote salvo antes desta branch
    monkeypatch.setattr(pp, "DERIVADAS", ["fracao_combustao"])
    assert antigo.transform(df[ENTRADAS]).shape[1] == len(antigo.nomes)
```

```python
# tests/test_explicar.py: test_contribuicoes_fecham_com_a_previsao
# passa a esperar ENTRADAS + DERIVADAS e roda também para o primeiro Plug-in da base
assert list(contrib.index) == ENTRADAS + DERIVADAS
```

```python
# tests/test_visual.py, junto de test_pergunta_legivel
def test_pergunta_fracao_combustao():
    no = {"coluna": "fracao_combustao", "categoria": None, "limite": 0.31}
    assert pergunta(no) == "Share of distance on fuel < 0.31?"
```

- [ ] **Step 2:** `uv run pytest tests/test_preprocess.py tests/test_explicar.py tests/test_visual.py -q`. Esperado: FAIL.
- [ ] **Step 3: Implement the changes listed under Files.**
  - Em `_numericas`, acrescente `N["fracao_combustao"] = fracao_combustao(X["propulsao"], X["autonomia_eletrica_km"])` para cada nome de `DERIVADAS`.
  - Em `fit`, `self.nomes` usa `list(N.columns)`.
  - Em `transform`, selecione `N[self._media.index]` antes de padronizar.
- [ ] **Step 4:** rodar a suíte inteira, `uv run pytest -q`. Esperado: PASS, incluindo:
  - `test_artefato_versionado.py`, que usa o pacote antigo;
  - `test_monotonia*.py`;
  - `test_app_veiculo.py`.

  Rodar também `uv run ruff check . && uv run ruff format --check .`.
- [ ] **Step 5:** commit com a mensagem `Modelos: fração da distância a combustão como entrada derivada e monótona`.

### Task 3: Sonda pareada `eficiencia-experimento entradas`

**Files:**
- Modify: `src/eficiencia/experimentos.py`: nova função e nova escolha no `argparse`; atualizar a docstring de uso do módulo.
- Test: `tests/test_experimentos.py`

**Interfaces:**
- Consumes: `eficiencia.preprocess.DERIVADAS` (Tarefa 2) e `cv_fixa` (de `validacao.py`).
- Produces: `experimento_entradas(params: dict, sementes=(0, 1, 2), rapido: bool = False) -> dict`.
  - **Variantes:** `{"base": [], "fracao_combustao": ["fracao_combustao"]}`, aplicadas em `eficiencia.preprocess.DERIVADAS` e restauradas num `finally`.
  - **Execução:** para cada variante e cada semente de dobras, roda `cv_fixa("xgboost", df, params, n_externas=..., semente_dobras=s)`.
  - **Retorno:** `{"hash_csv": hash8(df), "params": params, "variantes": {nome: {"media": float, "por_semente": [_resumo(...)]}}}`.
  - **Arquivo:** grava em `relatorios/experimentos/entradas-{hash_csv}.json` (em `rapido/` quando `rapido`).
- **CLI:** a escolha `entradas` usa os parâmetros publicados, `carregar().metadados["params"]` de `eficiencia.artefatos`. Não use `melhores_params`: depois da correção de rótulo o hash muda e o estudo não existe. Com `--rapido`, use `{}`, 2 dobras e `sementes=(0, 1)`.

- [ ] **Step 1: Write the failing test** `test_entradas_rapido(tmp_path, monkeypatch)`, no estilo de `test_sementes_rapido`.
  - Grava em `tmp_path` (monkeypatch em `PASTA_EXPERIMENTOS`).
  - Verifica que as duas variantes aparecem com 2 resumos cada.
  - Verifica que `eficiencia.preprocess.DERIVADAS == ["fracao_combustao"]` depois da chamada.
- [ ] **Step 2:** `uv run pytest tests/test_experimentos.py -q`. Esperado: FAIL.
- [ ] **Step 3:** implementar como descrito em Interfaces.
- [ ] **Step 4:** `uv run pytest -q` e ruff. Esperado: PASS.
- [ ] **Step 5:** commit com a mensagem `Experimentos: sonda pareada de entradas (base × fração a combustão)`.

### Task 4: Correção do rótulo na limpeza

**Files:**
- Modify: `src/eficiencia/dados.py`
  - Em `limpar`, logo depois de `df["propulsao"] = ...`, toda linha "Combustão" cujo texto `modelo + versao + nome_homologado` (sem acento, em maiúsculas) contenha a palavra inteira HYBRID, HIBRIDO ou HEV passa a "Híbrido". Isso precisa vir antes do zeramento de bateria e de autonomia e antes do `transmissao`.
  - A regex fica numa constante `HIBRIDO_NO_NOME`.
  - O relatório da limpeza ganha a chave `"Combustão com híbrido no nome → Híbrido"` com a contagem.
  - Em `checar`, novo invariante: nenhuma linha "Combustão" com `HIBRIDO_NO_NOME` no `nome_homologado`.
- Modify: `data/README.md`: uma linha nova na tabela de correções (em inglês): "RAV4 S/SX Hybrid entered as combustion | powertrain set to hybrid when the name says HYBRID/HÍBRIDO/HEV".
- Test: `tests/test_dados.py`

**Interfaces:**
- Produces: a base limpa com o RAV4 "S/SX HYBRID" como "Híbrido".
  - A bateria vem de `completar_baterias`: o complemento, se houver, senão a mediana dos híbridos, com `bateria_estimada = True`.
  - As contagens viram Combustão 286 e Híbrido 65. Continuam 510 homologações.

- [ ] **Step 1: Write the failing tests**

```python
from eficiencia.dados import HIBRIDO_NO_NOME, sem_acento

def test_hibrido_no_nome():  # roda no CI, sem a planilha
    for nome in ("I/TOYOTA/RAV4 SX HYBRID", "KIA NIRO HEV 1.6", "Q7 HÍBRIDO"):
        assert HIBRIDO_NO_NOME.search(sem_acento(nome).upper())
    for nome in ("CHEV/ONIX PLUS 10TAT", "MHEVX", "THEVENIN"):
        assert not HIBRIDO_NO_NOME.search(sem_acento(nome).upper())

@precisa_planilha
def test_rav4_hybrid_vira_hibrido(limpo):
    df, rel = limpo
    rav4 = df[df["nome_homologado"].isin(["I/TOYOTA/RAV4 SX HYBRID", "I/TOYOTA/RAV4 S HYBRID"])]
    assert set(rav4["propulsao"]) == {"Híbrido"} and (rav4["bateria_kwh"] > 0).all()
    assert rel["Combustão com híbrido no nome → Híbrido"] == 2
```

Use o normalizador que `dados.py` já tem: `sem_acento`, que tira o acento e põe em minúsculas, seguido de `.upper()`. Não crie outro.

- [ ] **Step 2:** `uv run pytest tests/test_dados.py -q`. Esperado: FAIL.
- [ ] **Step 3: Implement.** Antes de regravar a base, guarde uma cópia da atual, para a Tarefa 5 medir o efeito da correção. `$SCRATCH` é a pasta de rascunho da sessão; qualquer pasta fora do repositório serve:

```bash
cp data/processed/pbev_2026_limpo.csv "$SCRATCH/pbev_antes_rotulos.csv"
```

- [ ] **Step 4:** `uv run eficiencia-limpar`. Ele tem de imprimir a nova linha do relatório com 2. Depois:

```bash
uv run python -c "from eficiencia.esquema import carregar_base as c; print(c().propulsao.value_counts())"
```

Esperado: Combustão 286, Híbrido 65, Plug-in 50, Elétrico 109. Rodar `uv run pytest -q`: o `test_regenerar_igual_ao_commitado` compara com o CSV regravado e tem de passar.
- [ ] **Step 5:** commit com a mensagem `Dados: RAV4 S/SX Hybrid cadastrado como combustão vira híbrido`. Só `dados.py`, `data/README.md` e o teste: o CSV não é versionado.

### Task 5: Medir e decidir (controlador, sem subagente)

**Files:**
- Create: `relatorios/experimentos/entradas-<hash>.json`, dois arquivos gerados pela sonda.
- Modify: `relatorios/experimentos/conclusoes.md`: nova seção "H9: label fix and share of distance on fuel (2026-11-…)", em inglês.

- [ ] **Step 1: Efeito da correção de rótulo.** Rodar a sonda na base antiga, em segundo plano via WMI:

```bash
EFICIENCIA_BASE="$SCRATCH/pbev_antes_rotulos.csv" uv run eficiencia-experimento entradas
```

  Leva cerca de 6 × 3 min. Só a variante `base` importa aqui.
- [ ] **Step 2: Fração a combustão.** `uv run eficiencia-experimento entradas`, na base corrigida.
- [ ] **Step 3: Aplicar as regras e registrar.**
  - **Correção de rótulo:** fica sempre, porque é correção de dado. Registre o erro `base` antes e depois, total e por propulsão.
  - **Fração a combustão:** fica se cumprir as três condições:
    - na média das 3 sementes, o erro de MJ/km cai pelo menos 0,10 ponto em relação à `base`;
    - melhora em pelo menos 2 das 3 sementes;
    - o erro médio dos plug-ins cai.
  - **Se a fração for reprovada:** `git revert` dos commits das Tarefas 1, 2 e 3, nesta ordem inversa, e registre o resultado na H9. A correção de rótulo segue sozinha para a Tarefa 6.
  - **A seção H9** tem:
    - a tabela das variantes, no formato da H8 (média, por semente e plug-in por semente);
    - o veredito;
    - a lista de suspeitos sem prova (GLC63S E-Performance), com o motivo, como em Spec.
- [ ] **Step 4:** `uv run pytest -q` e ruff, depois commit com a mensagem `Experimentos: H9, correção de rótulo e fração a combustão`.

### Task 6: Retreino, artefatos e documentação (controlador)

**Files:**
- Modify: `modelos/vencedor/{pacote.joblib,metadados.json}`, `relatorios/{metricas.json,relatorio.md,figuras/*}` (gerados pelo treino).
- Modify: `relatorios/ficha_do_modelo.md` e `README.md`: tabela de erros, contagens por propulsão (Combustão 286, Híbrido 65), tabela de cobertura e exemplo do simulador.
- Modify: `relatorios/experimentos/conclusoes.md`: uma linha "Full retrain (date)" no fim da H9, como no fim da H8.

- [ ] **Step 1: Treinar.** Numa pasta separada (reaproveitar `../fuel_treino` com `git switch rotulos-fator-utilidade`), rodar o treino via WMI num laço que retoma depois de falta de memória:

```powershell
do { uv run eficiencia-treinar --tentativas 100 } until ($LASTEXITCODE -eq 0)
```

  Antes, copie `data/processed/pbev_2026_limpo.csv` corrigido para essa pasta. O hash da base muda, então o Optuna cria estudos novos e não retoma os antigos. Leva cerca de 21 h. Acompanhe com `uv run eficiencia-progresso`.
- [ ] **Step 2: Copiar os artefatos para esta branch.** Copie `modelos/vencedor/` e `relatorios/{metricas.json,relatorio.md,figuras}` da pasta de treino. Confira:
  - `uv run pytest -q`, incluindo `test_artefato_versionado`, que exige monotonia no pacote novo;
  - `checar_monotonia` sem violações também na fração: rode `checar_monotonia` no pacote. A fração não está em `ENTRADAS`, então confira-a à mão, variando a autonomia de um plug-in de 20 para 80 km: o MJ/km não pode subir.
- [ ] **Step 3: Medir a cobertura da faixa de 90 % de novo.** Reproduza as previsões fora da amostra com `cv_aninhada("xgboost", carregar_base(), 100, <storage do treino>)` e passe cada carro por `prever._registro`. A cobertura é a fração de carros com o valor real dentro de `[_min, _max]`, por propulsão, para `mj_km*` e `kml_*`. É o mesmo procedimento da ficha atual; a meta é ≥ 90 % em MJ/km por propulsão.
- [ ] **Step 4: Atualizar a documentação.**
  - **Ficha e README:** erros, contagens, cobertura e exemplo do simulador (Citroën Aircross flex 1210 kg: massa, F0 e F2 a −10 %). Rode o exemplo com o pacote novo.
  - **README:** se a fração entrou, cite "share of distance on fuel (from electric range, WLTP utility factor)" entre as entradas derivadas.
- [ ] **Step 5: Fechar.** `uv run pytest -q`, ruff e commit com a mensagem `Modelo retreinado com rótulos corrigidos (e fração a combustão)`. Depois, push da branch e PR para a `main` com a descrição no formato do skill `mattpocock-skills:pr`. O usuário mescla.
