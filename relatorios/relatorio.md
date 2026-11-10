# Training report: vehicle energy efficiency

Date: 2026-11-10 · data hash: `ad3d96d0`

**Selected model: XGBoost**, the lowest error among the candidates that pass the monotonicity check.

Hyperparameters: `{'n_estimators': 1178, 'max_depth': 5, 'learning_rate': 0.03492377029556782, 'subsample': 0.6420182922473265, 'colsample_bytree': 0.6093514545082879, 'min_child_weight': 2, 'reg_lambda': 0.015516427527411565}`

Errors are the mean absolute percentage error on combined MJ/km, out of sample (nested cross-validation grouped by model family).

## Candidates

| Candidate | MJ/km error | Monotonicity | Trials |
|---|---|---|---|
| Linear regression | 7.46 % | pass | 600 |
| Random forest | 6.55 % | pass | 600 |
| XGBoost | 5.71 % | pass | 600 |
| Neural network, one per target | 6.77 % | pass | 600 |
| Neural network, multi-output | 6.95 % | pass | 600 |

## Error by powertrain (MJ/km)

| Candidate | Combustion | Electric | Hybrid | Plug-in hybrid |
|---|---|---|---|---|
| Linear regression | 5.93 % | 7.38 % | 7.42 % | 16.52 % |
| Random forest | 5.31 % | 4.64 % | 11.61 % | 11.46 % |
| XGBoost | 4.71 % | 4.57 % | 7.2 % | 12.07 % |
| Neural network, one per target | 5.51 % | 5.54 % | 7.32 % | 16.02 % |
| Neural network, multi-output | 6.12 % | 5.71 % | 7.16 % | 14.21 % |

## Figures

![Error by powertrain](figuras/erro_por_propulsao.png)

![Predicted vs. label MJ/km for the selected model](figuras/previsto_real.png)
