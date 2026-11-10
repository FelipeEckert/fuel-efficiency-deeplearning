# Relatório de treino: eficiência energética

Data: 2026-11-10 · hash da base: `ad3d96d0`

**Vencedor: xgboost**, o de menor erro entre os que passam no teste de monotonia.
Hiperparâmetros: `{'n_estimators': 1178, 'max_depth': 5, 'learning_rate': 0.03492377029556782, 'subsample': 0.6420182922473265, 'colsample_bytree': 0.6093514545082879, 'min_child_weight': 2, 'reg_lambda': 0.015516427527411565}`

## Candidatos

| Candidato | Erro MJ/km | Monotonia | Tentativas |
|---|---|---|---|
| linear | 7,46 % | aprovado | 600 |
| floresta | 6,55 % | aprovado | 600 |
| xgboost | 5,71 % | aprovado | 600 |
| rede_por_alvo | 6,77 % | aprovado | 600 |
| rede_multi | 6,95 % | aprovado | 600 |

## Erro por propulsão (MJ/km)

| Candidato | Combustão | Elétrico | Híbrido | Plug-in |
|---|---|---|---|---|
| linear | 5,93 % | 7,38 % | 7,42 % | 16,52 % |
| floresta | 5,31 % | 4,64 % | 11,61 % | 11,46 % |
| xgboost | 4,71 % | 4,57 % | 7,2 % | 12,07 % |
| rede_por_alvo | 5,51 % | 5,54 % | 7,32 % | 16,02 % |
| rede_multi | 6,12 % | 5,71 % | 7,16 % | 14,21 % |

## Figuras

![Erro por propulsão](figuras/erro_por_propulsao.png)

![Previsto × real do vencedor](figuras/previsto_real.png)
