# Vehicle energy efficiency estimator

Estimates the energy consumption of a vehicle from its technical specifications, trained on Brazil's PBEV 2026 vehicle labeling data (Inmetro), with a Streamlit app on top.

The use case is early product planning: a team needs the label consumption (MJ/km and km/l, PBEV cycle) of vehicles that do not exist yet. The app estimates one vehicle or a whole portfolio weighted by sales volume, shows an uncertainty range, warns when a vehicle is outside what the data covers, and lets you try "what if" scenarios such as removing 80 kg.

It is also an honest comparison between classic regression and deep learning on the same data and the same validation.

## Results

Five candidates, each tuned with Optuna and evaluated with grouped cross-validation (versions of the same model never appear in both training and validation). Error is the mean absolute percentage error on combined MJ/km. A candidate only wins if it also passes a monotonicity check: more mass, drag or power must never predict lower consumption.

| Model | Error (MJ/km) | Combustion | Electric | Hybrid | Plug-in | Monotonicity |
|---|---|---|---|---|---|---|
| Linear regression | 7.52 % | 6.04 % | 7.39 % | 7.31 % | 16.57 % | pass |
| Random forest | 7.02 % | 5.36 % | 4.80 % | 11.47 % | 15.83 % | pass |
| **XGBoost (winner)** | **5.77 %** | **4.63 %** | **4.92 %** | **7.61 %** | **11.84 %** | pass |
| Neural net, one per target | 7.16 % | 5.81 % | 5.55 % | 8.68 % | 16.59 % | pass |
| Neural net, multi-output | 7.72 % | 6.12 % | 6.71 % | 9.00 % | 17.53 % | pass |

Plug-in hybrids are the hardest group, with only about 50 homologations in the data. The full training report, with figures, is in [`relatorios/relatorio.md`](relatorios/relatorio.md).

## Stack

Python, pandas, scikit-learn, XGBoost, PyTorch, Optuna, SHAP, Streamlit, pytest, Ruff, uv and GitHub Actions.

## How to use

```bash
uv sync
uv run eficiencia-limpar
uv run streamlit run app/Inicio.py
```

Train and compare the candidates (writes `modelos/vencedor/` and `relatorios/`):

```bash
uv run eficiencia-treinar                    # 50 trials in each Optuna study
uv run eficiencia-treinar --tentativas 200 --candidatos linear,xgboost   # winner and report only among the listed ones
uv run eficiencia-treinar --rapido           # fast mode, for development and CI only: writes to modelos/rapido/
```

The yearly data update is two commands: `uv run eficiencia-limpar` and `uv run eficiencia-treinar`.

The Optuna studies live in `optuna/estudos.db` and resume on every run. `--tentativas N` is the total for each study: an interrupted run only completes what is missing, and a study that already has N trials does not run again. To search further, raise N (for example, from 50 to 100).

### Overnight training

Training runs in a separate folder, so app development does not touch the model being trained. Since the main folder already has `main` checked out, the worktree uses its own branch:

```bash
git fetch
git worktree add ../fuel_treino -b treino-noturno origin/main
cd ../fuel_treino && uv sync
uv run eficiencia-treinar --tentativas 50
```

- On the following nights, update the code with `git -C ../fuel_treino pull` and train again with a larger `--tentativas` (100, then 150...). Each study continues where it stopped until it reaches the new total.
- Keep Windows from sleeping: Settings → System → Power → Sleep: Never (when plugged in).
- Follow progress with `uv run eficiencia-progresso --tentativas 100` (inside `../fuel_treino`, with the same number passed to the training run). It shows, for each model, how many trials are done, the best error and whether the model is still training. The "best in search" value comes from the internal validation and is a bit more optimistic than the error in the report.
- To see the charts of each study, run `uv run optuna-dashboard sqlite:///optuna/estudos.db` and open http://localhost:8080.

The app can use the trained model in two ways:

- Run the app from inside the worktree: `cd ../fuel_treino` and `uv run streamlit run app/Inicio.py`.
- Point the app in the main folder to the training folder, in PowerShell, before `uv run streamlit run app/Inicio.py`:

  ```powershell
  $env:EFICIENCIA_MODELO = "..\fuel_treino\modelos\vencedor"
  $env:EFICIENCIA_RELATORIOS = "..\fuel_treino\relatorios"
  ```

To publish a versioned model, commit `modelos/vencedor/` and `relatorios/` in a pull request.

The app needs a trained model (`uv run eficiencia-treinar`) before the Vehicle page shows results.

Data sources and the clean CSV format: [`data/README.md`](data/README.md). Project specification (in Portuguese): [`docs/superpowers/specs/2026-11-03-eficiencia-energetica-design.md`](docs/superpowers/specs/2026-11-03-eficiencia-energetica-design.md).

## Language

The code, the app interface and the internal docs (specification, plans, training report) are in Portuguese. This README and the data README are in English.
