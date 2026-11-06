# Investigação da otimização dos modelos: plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Objetivo:** testar por que os erros estabilizaram (XGBoost em 5,77 %, redes acima disso), com experimentos que não mudam o modelo publicado. Depois, decidir com números o que adotar.

**Arquitetura:** três ganchos pequenos no código de produção, todos com o padrão igual ao comportamento atual:
- sampler escolhível na CV aninhada;
- CV com parâmetros fixos;
- nas redes: histórico de perdas, validação por grupo e média de várias redes.

Um módulo novo, `eficiencia.experimentos` (comando `eficiencia-experimento`), roda cada experimento e grava JSON e figuras em `relatorios/experimentos/`. Os estudos vão para um banco separado, `optuna/experimentos.db`.

**Stack:** Python 3.14, Optuna 5.0, XGBoost, PyTorch (CPU), pytest, uv.

**Spec:** a seção "Contexto e hipóteses" abaixo. Ela resume a conversa de 2026-11-05 e a análise dos estudos das rodadas 1 a 3.

## Contexto e hipóteses

Análise do `estudos.db` (cópia de 2026-11-05, 11h):
- Os modelos clássicos chegam a 1 % do melhor cedo: linear na tentativa 0, floresta até a 15, XGBoost entre a 19 e a 51.
- As redes ainda melhoram no fim: o melhor fica entre as tentativas 76 e 99 de 100.
- O erro da busca interna do XGBoost varia de 5,47 % a 7,04 % entre as dobras externas.
- Os melhores parâmetros do XGBoost encostam na borda da faixa em várias dobras:
  - `max_depth` = 6, o máximo, em ext0 e ext4;
  - `min_child_weight` nos limites em ext1 e ext4;
  - `colsample_bytree` = 0,5, o mínimo, em ext3.
- As redes escolhem `lr` entre 2e-4 e 2e-3, no meio da faixa, que vai de 1e-4 a 1e-2.
- Hoje a validação da parada antecipada é 15 % de **linhas** sorteadas ([rede.py:128](../../../src/eficiencia/modelos/rede.py)). A base tem 510 carros em 258 grupos, então a validação quase sempre tem um "irmão" no treino.

| # | Hipótese | Experimento |
|---|---|---|
| H1 | O TPE fica preso e outro buscador acharia algo melhor | `samplers`: mesma CV aninhada do XGBoost com tpe, random, cmaes, qmc e gp |
| H2 | A faixa (a "cerca") do XGBoost é estreita | `cerca`: CV aninhada com `xgboost_amplo` |
| H3 | Parte das diferenças é só ruído de semente | `sementes`: XGBoost com os parâmetros finais fixos e 5 sementes |
| H4 | As redes param na hora errada por causa da validação com irmãos | `redes`: validação por linha contra validação por grupo, 5 sementes cada |
| H5 | As redes variam muito com a semente e a média de várias ajuda | `redes`: as 5 sementes acima, mais a média de 5 redes |
| H6 | A taxa de aprendizado atrapalha | `redes`: grade fixa de `lr` |

**Critério de decisão:** uma diferença só conta se for maior que 2 desvios-padrão entre sementes:
- XGBoost: são dois desvios. O de semente do modelo vem do H3. O de semente da busca vem do TPE com 3 sementes de sampler, na Tarefa 5b. H1 e H2 só são confirmadas se passarem no critério da **busca**. O relatório mostra os dois desvios (ajuste aprovado pelo usuário);
- redes: vale o desvio das 5 sementes com validação por linha.

**Fora deste plano:**
- o Simulated Annealing próprio, que só faz sentido se o H1 mostrar que um buscador diferente do TPE ganha;
- `--tentativas 100` para as redes, que é da sessão de treino;
- dados de outros países;
- adotar qualquer mudança no modelo publicado. Isso vai num plano e num PR separados, depois dos resultados.

## Restrições globais

- Comentários, mensagens e relatórios em pt-BR, simples e diretos.
- Nunca commitar na `main`. Um PR para o código (Tarefas 1–5) e outro PR para os resultados (Tarefa 6). O usuário faz o merge.
- Nunca dar stage em `.claude/` nem em `docs/superpowers/plans/2026-11-03-eficiencia-energetica.md`.
- Não mexer em `C:\Users\Felip\Desktop\fuel_treino` enquanto a rodada 3 não terminar.
- Comportamento padrão inalterado: sem `sampler`, sem `grupo_cv` em `X` e com `n_redes=1`, os resultados são idênticos aos atuais. `VERSAO` de nenhum candidato muda, e os nomes de estudo do TPE não mudam.
- Os experimentos nunca gravam em `optuna/estudos.db`. Eles leem de lá só os parâmetros finais.
- Processos longos são lançados via WMI (`Invoke-CimMethod Win32_Process Create`), para sobreviver ao fim da sessão.

## Foco da revisão

1. **Rede com os padrões atuais:** com `grupos=None` e `n_redes=1`, ela tem de dar previsões idênticas às de hoje. Senão os estudos já feitos deixam de valer. O teste de captura fica na Tarefa 3, passo 1.
2. **Alvo com poucas linhas na validação por grupo:** por exemplo, os alvos de etanol só existem nos flex. Com poucos grupos, a separação tem de deixar ao menos um grupo no treino e um na validação. O teste fica na Tarefa 3.
3. **Média de redes:** a média de redes monótonas tem de continuar monótona. O teste usa `checar_monotonia` na Tarefa 3.
4. **CMA-ES e GP com parâmetros inteiros e tentativas podadas** não podem quebrar a busca. O teste roda todos os samplers no XGBoost rápido, na Tarefa 1.
5. **Rodar um experimento de novo retoma os estudos** em vez de recomeçar, e o JSON é regravado. Isso vale porque `cv_aninhada` já usa `load_if_exists`. O teste fica na Tarefa 4.

---

### Tarefa 1: sampler escolhível na CV aninhada

**Arquivos:**
- Modificar: `src/eficiencia/validacao.py` (`_estudo`, `buscar_params`, `cv_aninhada`)
- Modificar: `pyproject.toml`: adicionar `"cmaes"` em `dependencies`. O `CmaEsSampler` do Optuna exige esse pacote, que não está instalado.
- Teste: `tests/test_validacao.py`

**Interfaces:**
- Produz:
  - `SAMPLERS: dict[str, type[optuna.samplers.BaseSampler]]`, com as chaves `"tpe"`, `"random"`, `"cmaes"`, `"qmc"` e `"gp"`;
  - `cv_aninhada(..., sampler: str = "tpe")` e `buscar_params(..., sampler: str = "tpe")`;
  - nome de estudo: `f"{prefixo_estudo(nome, h)}-{sampler}-e{n_externas}-..."` quando `sampler != "tpe"`. Com `"tpe"`, o nome fica igual ao de hoje.

- [ ] **Passo 1: escrever os testes que falham**

```python
@pytest.mark.parametrize("sampler", ["tpe", "random", "cmaes", "qmc", "gp"])
def test_cv_com_cada_sampler(sampler):
    r = cv_aninhada("xgboost", carregar_base(), n_tentativas=3, storage=None,
                    n_externas=2, n_internas=2, rapido=True, sampler=sampler)
    assert np.isfinite(r.erro_mj)

def test_nome_do_estudo_leva_o_sampler(tmp_path):
    storage = f"sqlite:///{tmp_path / 'e.db'}"
    df = carregar_base()
    cv_aninhada("linear", df, n_tentativas=1, storage=storage, n_externas=2, n_internas=2, sampler="random")
    cv_aninhada("linear", df, n_tentativas=1, storage=storage, n_externas=2, n_internas=2)
    nomes = optuna.get_all_study_names(storage)
    assert sum("-random-" in n for n in nomes) == 2
    assert sum("-random-" not in n for n in nomes) == 2  # o TPE mantém o nome antigo

def test_sampler_desconhecido():
    with pytest.raises(ValueError, match="sampler desconhecido"):
        cv_aninhada("linear", carregar_base(), 1, None, n_externas=2, n_internas=2, sampler="sa")
```

- [ ] **Passo 2:** rode `uv run pytest tests/test_validacao.py -v`. Esperado: FAIL, porque `sampler` ainda não é um argumento aceito.
- [ ] **Passo 3: implementar.**
  - Em `_estudo`, troque a linha do `TPESampler` por `SAMPLERS[sampler](seed=semente + len(estudo.trials))`. Mantenha o comentário da semente deslocada.
  - Valide o nome antes de criar o estudo, com `ValueError(f"sampler desconhecido: {sampler}")`.
  - Adicione `cmaes` ao `pyproject.toml` e rode `uv sync`.
- [ ] **Passo 4:** rode `uv run pytest tests/test_validacao.py -v`. Esperado: PASS. O `test_nomes_de_estudo_separados_por_configuracao`, que já existe, continua passando.
- [ ] **Passo 5: commit**

```bash
git add src/eficiencia/validacao.py tests/test_validacao.py pyproject.toml uv.lock
git commit -m "Validação: sampler do Optuna escolhível na CV aninhada"
```

### Tarefa 2: CV externa com parâmetros fixos

**Arquivos:**
- Modificar: `src/eficiencia/validacao.py`. Extraia o cálculo de métricas do fim de `cv_aninhada` para `_resultado`, que as duas funções usam.
- Teste: `tests/test_validacao.py`

**Interfaces:**
- Produz:
  - `cv_fixa(nome: str, df: pd.DataFrame, params: dict, n_externas: int = 5, semente_dobras: int = 0, semente_modelo: int = 0, colunas: list[str] = ENTRADAS) -> ResultadoCV`, com `n_tentativas=0`;
  - as dobras são as mesmas de `cv_aninhada` com `semente=semente_dobras`;
  - `colunas` permite passar `ENTRADAS + ["grupo_cv"]`. O `Preprocessador` ignora as colunas extras.

- [ ] **Passo 1: escrever os testes que falham**

```python
def test_cv_fixa_linear():
    r = cv_fixa("linear", carregar_base(), {"alpha": 1.0}, n_externas=2)
    assert r.oof.notna()["mj_km_urbano"].all() and np.isfinite(r.erro_mj) and r.n_tentativas == 0

def test_cv_fixa_semente_do_modelo_muda_so_o_modelo():
    df, p = carregar_base(), {**CANDIDATOS["xgboost"].PARAMS_RAPIDOS, "subsample": 0.7}
    a = cv_fixa("xgboost", df, p, n_externas=2, semente_modelo=0)
    b = cv_fixa("xgboost", df, p, n_externas=2, semente_modelo=0)
    c = cv_fixa("xgboost", df, p, n_externas=2, semente_modelo=1)
    assert a.erro_mj == b.erro_mj and a.erro_mj != c.erro_mj
```

- [ ] **Passo 2:** rode `uv run pytest tests/test_validacao.py -k cv_fixa -v`. Esperado: FAIL com ImportError.
- [ ] **Passo 3:** implemente `cv_fixa` e `_resultado(nome, df, oof, params_por_dobra, n_tentativas) -> ResultadoCV`. `cv_aninhada` passa a chamar `_resultado`.
- [ ] **Passo 4:** rode `uv run pytest tests/test_validacao.py -v`. Esperado: PASS.
- [ ] **Passo 5: commit**: `git commit -m "Validação: CV externa com parâmetros fixos"`, só com os dois arquivos.

### Tarefa 3: redes com histórico de perdas, validação por grupo e média de várias redes

**Arquivos:**
- Modificar: `src/eficiencia/modelos/rede.py` (`_Mlp`, `_EstimadorMlp`, `RedeMulti.fit`, `RedePorAlvo`)
- Modificar: `src/eficiencia/modelos/base.py` (`CandidatoPorAlvo.fit`)
- Teste: `tests/test_modelos_rede.py`

**Interfaces:**
- Produz:
  - `_Mlp(..., n_redes: int = 1)`, que treina `n_redes` redes com as sementes `semente`, `semente+1`, … e prevê a média delas no espaço padronizado;
  - `_Mlp.fit(X, Y, grupos: np.ndarray | None = None)`;
  - `_Mlp.historicos: list[list[tuple[float, float]]]`: uma lista por rede, com uma tupla por época (perda média de treino, perda de validação);
  - `_separar_validacao(n: int, grupos: np.ndarray | None, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]`, que devolve `(validacao, treino)`;
  - `CandidatoPorAlvo.USA_GRUPOS: ClassVar[bool] = False`. Quando é `True` e `X` tem `grupo_cv`, o `fit` passa `grupos=grupos[presente]` ao estimador. `RedePorAlvo` define `True`, e `RedeMulti.fit` lê `X["grupo_cv"]` se existir;
  - `n_redes` entra pelos `**params`, como os outros hiperparâmetros.

- [ ] **Passo 1: capturar o comportamento atual antes de mexer** (foco da revisão 1). Rode no código atual:

```bash
uv run python -c "from eficiencia.esquema import *; from eficiencia.modelos import CANDIDATOS as C; df=carregar_base(); [print(n, repr(C[n](semente=1, max_epocas=5).fit(df[ENTRADAS], df[ALVOS]).predict(df[ENTRADAS]).iloc[:3].round(10).values.tolist())) for n in ('rede_por_alvo','rede_multi')]"
```

  Cole a saída como constantes no teste `test_padroes_mantem_previsoes`, que compara com `np.testing.assert_allclose(..., rtol=1e-6)`.

- [ ] **Passo 2: escrever os testes que falham**

```python
def test_padroes_mantem_previsoes(): ...  # constantes do passo 1, para as duas redes

def test_validacao_por_grupo_sem_irmaos():
    from eficiencia.modelos.rede import _separar_validacao
    g = carregar_base()["grupo_cv"].to_numpy()
    val, tr = _separar_validacao(len(g), g, np.random.default_rng(0))
    assert not set(g[val]) & set(g[tr])
    assert 0.15 <= len(val) / len(g) < 0.30

def test_validacao_por_grupo_com_dois_grupos():
    from eficiencia.modelos.rede import _separar_validacao
    val, tr = _separar_validacao(5, np.array(list("aaabb")), np.random.default_rng(0))
    assert len(val) and len(tr)

def test_historico_e_media_de_redes():
    df = carregar_base()
    m = CANDIDATOS["rede_por_alvo"](max_epocas=5, n_redes=3).fit(df[ENTRADAS + ["grupo_cv"]], df[ALVOS])
    mlp = m.estimadores["mj_km_urbano"].mlp
    assert len(mlp.historicos) == 3 and all(len(h) == 5 for h in mlp.historicos)
    assert np.isfinite(m.predict(df[ENTRADAS]).values).all()
    assert checar_monotonia(m, df) == []
```

- [ ] **Passo 3:** rode `uv run pytest tests/test_modelos_rede.py -v`. Esperado: só os testes novos falham.
- [ ] **Passo 4: implementar.**
  - Com `grupos=None`, `_separar_validacao` faz o mesmo sorteio de hoje (`rng.permutation(n)`, primeiros `n_val`).
  - Com grupos, ela embaralha os grupos únicos e acumula grupos inteiros até chegar a `FRACAO_VALIDACAO * n` linhas, deixando sempre ao menos um grupo no treino.
  - A rede `i` usa `torch.manual_seed(semente + i)` e `default_rng(semente + i)`, na mesma ordem de chamadas de hoje, para que a rede 0 saia idêntica.
  - `self.redes` guarda as redes. `predict` faz a média antes de desfazer a padronização.
- [ ] **Passo 5:** rode `uv run pytest tests/test_modelos_rede.py tests/test_monotonia_candidatos.py -v`. Esperado: PASS.
- [ ] **Passo 6: commit**: `git commit -m "Rede: histórico de perdas, validação por grupo e média de várias redes (padrão inalterado)"`.

### Tarefa 4: módulo de experimentos para XGBoost (samplers, cerca, sementes)

**Arquivos:**
- Criar: `src/eficiencia/experimentos.py`
- Modificar: `src/eficiencia/modelos/xgb.py`: classe `XGBoostAmplo`
- Modificar: `src/eficiencia/modelos/__init__.py`: `EXPERIMENTAIS`
- Modificar: `src/eficiencia/validacao.py`: busca em `CANDIDATOS | EXPERIMENTAIS`
- Modificar: `pyproject.toml`: script `eficiencia-experimento = "eficiencia.experimentos:main"`
- Teste: `tests/test_experimentos.py`

**Interfaces:**
- Consome: `cv_aninhada(..., sampler=)` (Tarefa 1) e `cv_fixa` (Tarefa 2).
- Produz:
  - `FAIXAS_AMPLO: dict[str, tuple[float, float]]` em `xgb.py`. É a fonte única da faixa ampliada:

    | Parâmetro | Faixa | Escala |
    |---|---|---|
    | `n_estimators` | 200–3000 | |
    | `max_depth` | 2–10 | |
    | `learning_rate` | 0,005–0,3 | log |
    | `subsample` | 0,4–1,0 | |
    | `colsample_bytree` | 0,2–1,0 | |
    | `min_child_weight` | 0–20 | inteiro |
    | `reg_lambda` | 1e-3–100 | log |

  - `XGBoostAmplo(XGBoost)`, com `nome = "xgboost_amplo"` e `sugerir` lendo `FAIXAS_AMPLO`. **Ela não entra em `CANDIDATOS`**, para que o treino oficial e o `eficiencia-progresso` não mudem. Em vez disso:
    - `modelos/__init__.py` ganha `EXPERIMENTAIS = {XGBoostAmplo.nome: XGBoostAmplo}`;
    - `validacao.py` busca a classe em `CANDIDATOS | EXPERIMENTAIS`, nas três consultas a `CANDIDATOS[nome]` (`prefixo_estudo`, `_estudo`, `_params_finais`/`cv_aninhada`/`cv_fixa`);
    - nada muta `CANDIDATOS`.
  - `STORAGE_EXPERIMENTOS = sqlite:///<RAIZ>/optuna/experimentos.db` e `PASTA_EXPERIMENTOS = RAIZ / "relatorios/experimentos"`.
  - `melhores_params(nome: str, storage: str = STORAGE) -> dict`: devolve os `best_params` do estudo `f"{prefixo_estudo(nome, h)}-i3-s0-final"`. Se o estudo não existir, dá erro claro.
  - `na_borda(params: dict, faixas: dict) -> list[str]`: parâmetros a menos de 5 % da largura da faixa de um dos limites. Nos parâmetros em escala log, a distância é medida em log10.
  - `experimento_samplers(n_tentativas=50, storage=STORAGE_EXPERIMENTOS, rapido=False) -> dict`;
  - `experimento_cerca(n_tentativas=100, ...) -> dict`, que inclui `bordas` por dobra externa;
  - `experimento_sementes(params: dict, n=5, rapido=False) -> dict`, que inclui `media` e `desvio` do `erro_mj`.
  - Cada função grava `PASTA_EXPERIMENTOS/<nome>.json` (ou `.../rapido/<nome>.json` com `rapido=True`). Por configuração, o JSON traz `erro_mj` e `erro_por_propulsao`.
  - `main(argv)`: `eficiencia-experimento {samplers,cerca,sementes,redes} [--tentativas N] [--rapido]`.
    - Nas funções, `rapido=True` muda só três coisas: dobras (2 externas e 2 internas), `PARAMS_RAPIDOS` e a pasta de saída (`rapido/`).
    - Tentativas e storage são sempre os que o chamador passa.
    - O `main` com `--rapido` passa 2 tentativas e `storage=None`.

- [ ] **Passo 1: escrever os testes que falham**

```python
def test_na_borda():
    assert na_borda({"max_depth": 10, "learning_rate": 0.05}, FAIXAS_AMPLO) == ["max_depth"]

def test_samplers_rapido(tmp_path, monkeypatch):
    monkeypatch.setattr(experimentos, "PASTA_EXPERIMENTOS", tmp_path)
    r = experimentos.experimento_samplers(n_tentativas=2, storage=None, rapido=True)
    assert set(r) == {"tpe", "random", "cmaes", "qmc", "gp"}
    assert json.loads((tmp_path / "rapido" / "samplers.json").read_text(encoding="utf-8")) == r

def test_cerca_retoma(tmp_path, monkeypatch):
    monkeypatch.setattr(experimentos, "PASTA_EXPERIMENTOS", tmp_path)
    storage = f"sqlite:///{(tmp_path / 'x.db').as_posix()}"
    experimentos.experimento_cerca(n_tentativas=2, storage=storage, rapido=True)
    experimentos.experimento_cerca(n_tentativas=2, storage=storage, rapido=True)
    estudos = optuna.get_all_study_summaries(storage)
    assert estudos and all(s.n_trials == 2 for s in estudos)

def test_sementes_rapido(tmp_path, monkeypatch):
    monkeypatch.setattr(experimentos, "PASTA_EXPERIMENTOS", tmp_path)
    r = experimentos.experimento_sementes(CANDIDATOS["xgboost"].PARAMS_RAPIDOS | {"subsample": 0.7}, n=2, rapido=True)
    assert r["desvio"] >= 0 and len(r["por_semente"]) == 2

def test_experimento_nao_esta_no_treino_oficial():
    assert "xgboost_amplo" not in CANDIDATOS  # mesmo com experimentos importado
    assert experimentos.STORAGE_EXPERIMENTOS != treino.STORAGE
```

- [ ] **Passo 2:** rode `uv run pytest tests/test_experimentos.py -v`. Esperado: FAIL com ImportError.
- [ ] **Passo 3:** implemente `XGBoostAmplo`, `FAIXAS_AMPLO` e as funções acima. Implemente também `main` com os subcomandos `samplers`, `cerca` e `sementes`. Em `sementes`, sem `--rapido`, os parâmetros vêm de `melhores_params("xgboost")`.
- [ ] **Passo 4:** rode `uv run pytest -q`. Esperado: tudo PASS.
- [ ] **Passo 5: commit**: `git commit -m "Experimentos: samplers, cerca ampla do XGBoost e ruído de semente"`.

### Tarefa 5: experimento das redes e curvas de perda

**Arquivos:**
- Modificar: `src/eficiencia/experimentos.py`
- Teste: `tests/test_experimentos.py`

**Interfaces:**
- Consome: `cv_fixa(..., colunas=)` (Tarefa 2), mais `n_redes` e `historicos` (Tarefa 3).
- Produz: `experimento_redes(params: dict, n: int = 5, lrs=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2), rapido=False) -> dict`, sempre com `rede_por_alvo`. As chaves do resultado:

  | Chave | Configuração |
  |---|---|
  | `linha_s{k}` | semente `k`, colunas `ENTRADAS` |
  | `grupo_s{k}` | semente `k`, colunas `ENTRADAS + ["grupo_cv"]` |
  | `media5` | `n_redes=5`, colunas `ENTRADAS` |
  | `lr_{valor}` | semente 0, colunas `ENTRADAS` |

  Também resume `media`/`desvio` de `linha_s*` e de `grupo_s*`.
- Produz também a figura `PASTA_EXPERIMENTOS/curvas_perda.png`. Ela vem de uma rede treinada na dobra externa 0, alvo `mj_km_urbano`, nos dois modos de validação. São dois painéis, cada um com a perda de treino e a de validação por época. Os títulos e eixos são em pt-BR.
- `main`: o subcomando `redes` usa `melhores_params("rede_por_alvo")`.

- [ ] **Passo 1: escrever o teste que falha**

```python
def test_redes_rapido(tmp_path, monkeypatch):
    monkeypatch.setattr(experimentos, "PASTA_EXPERIMENTOS", tmp_path)
    p = {"camadas": 1, "largura": 32, "dropout": 0.0, "lr": 1e-3, "weight_decay": 1e-4,
         "tamanho_lote": 32, "max_epocas": 3}
    r = experimentos.experimento_redes(p, n=2, lrs=(1e-3,), rapido=True)
    assert {"linha_s0", "linha_s1", "grupo_s0", "grupo_s1", "media5", "lr_0.001"} <= set(r)
    assert (tmp_path / "rapido" / "curvas_perda.png").exists()
```

- [ ] **Passo 2:** rode `uv run pytest tests/test_experimentos.py::test_redes_rapido -v`. Esperado: FAIL.
- [ ] **Passo 3:** implemente `experimento_redes`, a figura (com `Figure` + `FigureCanvasAgg`, como em `treino.py`) e o subcomando `redes`.
- [ ] **Passo 4:** rode `uv run pytest -q` e `uv run ruff check . && uv run ruff format --check .`. Esperado: tudo PASS.
- [ ] **Passo 5: commit e PR do código.**

```bash
git commit -m "Experimentos: redes (validação por grupo, sementes, média, lr) e curvas de perda"
git push -u origin <branch>
gh pr create --title "Experimentos de otimização: samplers, cerca, sementes e redes" --body "..."
```

### Tarefa 5b: ruído da busca (TPE com 3 sementes de sampler)

Ajuste aprovado pelo usuário depois da Tarefa 5. Hoje `cv_aninhada` usa a mesma `semente` para as dobras e para o sampler. Para medir só o ruído da busca, as dobras têm de ficar fixas e só a semente do sampler deve variar.

**Arquivos:**
- Modificar: `src/eficiencia/validacao.py` (`_estudo`, `cv_aninhada`)
- Modificar: `src/eficiencia/experimentos.py` (`experimento_samplers`)
- Teste: `tests/test_validacao.py`, `tests/test_experimentos.py`

**Interfaces:**
- Produz:
  - `_estudo(..., semente_sampler: int | None = None)`. O sampler usa `seed=(semente if semente_sampler is None else semente_sampler) + len(estudo.trials)`. As dobras internas continuam com `semente`.
  - `cv_aninhada(..., semente_sampler: int | None = None)`, que repassa o valor a `_estudo`. As dobras externas continuam com `semente`. Com `semente_sampler` diferente de `None`, o nome do estudo ganha `-ss{semente_sampler}` logo depois do sufixo do sampler: `f"{prefixo}{sufixo_sampler}-ss{k}-e{n_externas}-..."`. Com `None`, o nome não muda.
  - O resultado de `experimento_samplers` mantém as chaves de hoje (`tpe`, `random`, `cmaes`, `qmc`, `gp`) e acrescenta:
    - `tpe_ss1` e `tpe_ss2`: TPE com `semente_sampler` 1 e 2, nas mesmas dobras;
    - `ruido_busca_tpe`: `{"erros": [tpe, tpe_ss1, tpe_ss2], "media": float, "desvio": float}`, com o desvio em `ddof=1`, igual ao de `sementes`.
  - A chave `tpe` (semente 0, sem sufixo) já é a primeira das 3 sementes.

- [ ] **Passo 1: escrever os testes que falham**

```python
def test_semente_sampler_muda_so_a_busca(tmp_path):
    storage = f"sqlite:///{tmp_path / 'e.db'}"
    df = carregar_base()
    cv_aninhada("linear", df, n_tentativas=2, storage=storage, n_externas=2, n_internas=2)
    cv_aninhada("linear", df, n_tentativas=2, storage=storage, n_externas=2, n_internas=2, semente_sampler=1)
    nomes = optuna.get_all_study_names(storage)
    com = sorted(n for n in nomes if "-ss1-" in n)
    sem = sorted(n for n in nomes if "-ss" not in n)
    assert len(com) == len(sem) == 2
    a = optuna.load_study(study_name=sem[0], storage=storage).trials
    b = optuna.load_study(study_name=com[0], storage=storage).trials
    assert [t.params for t in a] != [t.params for t in b]
```

  Em `tests/test_experimentos.py`, o `test_samplers_rapido` passa a esperar `set(SAMPLERS) | {"tpe_ss1", "tpe_ss2", "ruido_busca_tpe"}` e confere que `len(r["ruido_busca_tpe"]["erros"]) == 3` e que `r["ruido_busca_tpe"]["desvio"] >= 0`.

- [ ] **Passo 2:** rode `uv run pytest tests/test_validacao.py tests/test_experimentos.py -v`. Esperado: só os testes novos ou alterados falham.
- [ ] **Passo 3:** implemente o que está em Interfaces.
- [ ] **Passo 4:** rode `uv run pytest -q` e depois `uv run ruff check . && uv run ruff format --check .`. Esperado: tudo PASS, menos a falha conhecida `test_regenerar_igual_ao_commitado`.
- [ ] **Passo 5: commit**: `git commit -m "Experimentos: ruído da busca com 3 sementes do TPE"`. Só commit, sem push.

### Tarefa 6: rodar os experimentos e escrever as conclusões

Depende de: o PR das Tarefas 1–5b estar mergeado, o PR #22 (rodada 3) estar mergeado e a rodada 3 ter terminado. Para conferir, rode `uv run eficiencia-progresso --tentativas 50` em `fuel_treino`: todos os modelos têm de estar completos.

**Arquivos:**
- Criar: `relatorios/experimentos/*.json`, `relatorios/experimentos/curvas_perda.png` e `relatorios/experimentos/conclusoes.md`

- [ ] **Passo 1:** em `fuel_treino`, rode `git switch main && git pull && uv sync`.
- [ ] **Passo 2:** lance via WMI, nesta ordem e um de cada vez, para não disputar CPU:
  1. `sementes`
  2. `redes`
  3. `cerca`
  4. `samplers`

  Exemplo:

```powershell
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = 'cmd /c "cd /d C:\Users\Felip\Desktop\fuel_treino && uv run eficiencia-experimento sementes > relatorios\experimentos\sementes.log 2>&1 && uv run eficiencia-experimento redes > relatorios\experimentos\redes.log 2>&1 && uv run eficiencia-experimento cerca > relatorios\experimentos\cerca.log 2>&1 && uv run eficiencia-experimento samplers > relatorios\experimentos\samplers.log 2>&1"' }
```

  Estimativa (baseada no tempo médio das tentativas no banco):

  | Experimento | Tempo |
  |---|---|
  | sementes | ~10 min |
  | redes | ~2–3 h |
  | cerca, 100 tentativas | ~3 h |
  | samplers, 7 × 50 tentativas | ~10 h |
  | **Total** | **~15 h** |

  Para acompanhar, abra o painel em `optuna/experimentos.db` com `uv run optuna-dashboard sqlite:///optuna/experimentos.db`.

- [ ] **Passo 3:** escreva `conclusoes.md` com:
  - uma tabela por hipótese (H1–H6) com o erro MJ/km e por propulsão;
  - o veredito de cada hipótese pelo critério de 2 desvios-padrão;
  - a ressalva de que o H1 foi testado só no XGBoost, cuja busca convergiu cedo. Ele não diz nada sobre as redes;
  - a ressalva de que o H4 reusa parâmetros ajustados com validação por linha. Um empate não descarta a validação por grupo;
  - as bordas da cerca ampla;
  - a figura das curvas;
  - uma lista do que adotar, cada item como um plano/PR futuro.
- [ ] **Passo 4:** numa branch nova a partir da `main`, dê `git add` só em `relatorios/experimentos/` (sem os `.log`), faça o commit e abra o PR "Resultados dos experimentos de otimização".

## Depois dos resultados (planos futuros, não executar aqui)

- **H2 confirmada:** a faixa ampla vira a do `XGBoost`, com `VERSAO = 2`, e o XGBoost é treinado de novo.
- **H4 ou H5 confirmadas:** `treino`/`validacao` passam a enviar `grupo_cv` e/ou `n_redes` vira padrão, com `VERSAO = 3` nas redes.
- **H1 confirmada:** trocar o sampler padrão e, aí sim, avaliar o Simulated Annealing como sampler próprio.
