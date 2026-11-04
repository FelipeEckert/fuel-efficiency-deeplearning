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
uv run eficiencia-treinar --rapido           # modo rápido, só para desenvolvimento e CI: grava em modelos/rapido/
```

A atualização anual da base são dois comandos: `uv run eficiencia-limpar` e `uv run eficiencia-treinar`.

Os estudos do Optuna ficam em `optuna/estudos.db` e são retomados a cada execução.

### Treino noturno

O treino roda em uma pasta separada, para que o desenvolvimento do app não mexa no modelo que está sendo treinado. Como a pasta principal já está com `main` aberto, o worktree usa uma branch própria:

```bash
git fetch
git worktree add ../fuel_treino -b treino-noturno origin/main
cd ../fuel_treino && uv sync
uv run eficiencia-treinar --tentativas 50
```

- Nas noites seguintes, atualize o código com `git -C ../fuel_treino pull` e rode o treino de novo. Rodar de novo continua os estudos do Optuna, sem recomeçar.
- Impeça o Windows de suspender a máquina: Configurações → Sistema → Energia → Suspender: Nunca (na tomada).
- Acompanhe o progresso com `uv run optuna-dashboard sqlite:///optuna/estudos.db` (dentro de `../fuel_treino`), em http://localhost:8080.

Para o app usar o modelo treinado, há duas opções:

- Rodar o app de dentro do worktree: `cd ../fuel_treino` e `uv run streamlit run app/Inicio.py`.
- Apontar o app da pasta principal para o treino, no PowerShell, antes de `uv run streamlit run app/Inicio.py`:

  ```powershell
  $env:EFICIENCIA_MODELO = "..\fuel_treino\modelos\vencedor"
  $env:EFICIENCIA_RELATORIOS = "..\fuel_treino\relatorios"
  ```

Para publicar um modelo versionado, faça commit de `modelos/vencedor/` e `relatorios/` em um PR.

O app precisa de um modelo treinado (`uv run eficiencia-treinar`) antes que a página Veículo mostre resultados.

Dados e formato do CSV limpo: [`data/README.md`](data/README.md). Especificação do projeto: [`docs/superpowers/specs/2026-11-03-eficiencia-energetica-design.md`](docs/superpowers/specs/2026-11-03-eficiencia-energetica-design.md).
