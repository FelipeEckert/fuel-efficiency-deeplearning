# Eficiência energética de veículos

Estima o consumo energético de veículos a partir dos dados técnicos do PBEV 2026, com um modelo de aprendizado de máquina e um app Streamlit.

## Uso

```bash
uv sync
uv run eficiencia-limpar
```

Treino e comparação dos candidatos (gera `modelos/vencedor/` e `relatorios/`):

```bash
uv run eficiencia-treinar                    # 50 tentativas por candidato
uv run eficiencia-treinar --tentativas 200 --candidatos linear,xgboost
uv run eficiencia-treinar --rapido           # modo rápido, para desenvolvimento e CI
uv run optuna-dashboard sqlite:///optuna/estudos.db   # acompanhar os estudos salvos
```

Os estudos do Optuna ficam em `optuna/estudos.db` e são retomados a cada execução.

Dados e formato do CSV limpo: [`data/README.md`](data/README.md). Especificação do projeto: [`docs/superpowers/specs/2026-11-03-eficiencia-energetica-design.md`](docs/superpowers/specs/2026-11-03-eficiencia-energetica-design.md).
