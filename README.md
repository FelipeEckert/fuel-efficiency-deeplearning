# Eficiência energética de veículos

Estima o consumo energético de veículos a partir dos dados técnicos do PBEV 2026, com um modelo de aprendizado de máquina e um app Streamlit.

## Como usar

```bash
uv sync
uv run eficiencia-limpar
uv run streamlit run app/Inicio.py
```

Treino e comparação dos candidatos (gera `modelos/vencedor/` e `relatorios/`):

```bash
uv run eficiencia-treinar                    # 50 tentativas por candidato
uv run eficiencia-treinar --tentativas 200 --candidatos linear,xgboost
uv run eficiencia-treinar --rapido           # modo rápido, para desenvolvimento e CI
```

A atualização anual da base são dois comandos: `uv run eficiencia-limpar` e `uv run eficiencia-treinar`.

Os estudos do Optuna ficam em `optuna/estudos.db` e são retomados a cada execução.

### Treino noturno

- Treine em uma pasta separada, criada com `git worktree add ../fuel_treino main` (depois `uv sync` nela). Assim o desenvolvimento do app não mexe no modelo que está sendo treinado.
- Impeça o Windows de suspender a máquina enquanto o treino roda.
- Rode `uv run eficiencia-treinar --tentativas 50`. Rodar de novo continua os estudos do Optuna, sem recomeçar.
- Acompanhe o progresso com `uv run optuna-dashboard sqlite:///optuna/estudos.db`, em http://localhost:8080.

Dados e formato do CSV limpo: [`data/README.md`](data/README.md). Especificação do projeto: [`docs/superpowers/specs/2026-11-03-eficiencia-energetica-design.md`](docs/superpowers/specs/2026-11-03-eficiencia-energetica-design.md).
