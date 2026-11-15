# Training report: vehicle energy efficiency

Date: 2026-11-13 · data hash: `7084f5b9`

**Selected model: XGBoost**, the lowest error among the candidates that pass the monotonicity check.

Hyperparameters: `{'n_estimators': 1262, 'max_depth': 6, 'learning_rate': 0.03160110464004988, 'subsample': 0.6161206013916076, 'colsample_bytree': 0.5205588203150939, 'min_child_weight': 2, 'reg_lambda': 0.03753736138591272}`

Errors are the mean absolute percentage error on combined MJ/km, out of sample (nested cross-validation grouped by model family).

## Candidates

| Candidate | MJ/km error | Monotonicity | Trials |
|---|---|---|---|
| Linear regression | 7.36 % | pass | 600 |
| Random forest | 6.53 % | pass | 600 |
| XGBoost | 5.64 % | pass | 600 |
| Neural network, one per target | 6.52 % | pass | 600 |
| Neural network, multi-output | 6.62 % | pass | 600 |

## Error by powertrain (MJ/km)

| Candidate | Combustion | Electric | Hybrid | Plug-in hybrid |
|---|---|---|---|---|
| Linear regression | 5.63 % | 7.43 % | 7.88 % | 16.48 % |
| Random forest | 5.01 % | 4.56 % | 12.14 % | 12.25 % |
| XGBoost | 4.17 % | 4.76 % | 7.93 % | 13.05 % |
| Neural network, one per target | 5.17 % | 5.67 % | 7.14 % | 15.28 % |
| Neural network, multi-output | 5.41 % | 5.29 % | 6.73 % | 16.33 % |

## Figures

![Error by powertrain](figuras/erro_por_propulsao.png)

![Predicted vs. label MJ/km for the selected model](figuras/previsto_real.png)
