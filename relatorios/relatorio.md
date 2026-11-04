# Relatório de treino: eficiência energética

Data: 2026-11-04 · hash da base: `dd888117`

**Vencedor: xgboost**, o de menor erro entre os que passam no teste de monotonia.
Hiperparâmetros: `{'n_estimators': 666, 'max_depth': 5, 'learning_rate': 0.030076202335325257, 'subsample': 0.6321468646720626, 'colsample_bytree': 0.5967692763670944, 'min_child_weight': 5, 'reg_lambda': 0.0041800801362600745}`

## Candidatos

| Candidato | Erro MJ/km | Monotonia | Tentativas |
|---|---|---|---|
| linear | 7,52 % | aprovado | 600 |
| floresta | 7,02 % | aprovado | 374 |
| xgboost | 5,77 % | aprovado | 300 |
| rede_por_alvo | 8,15 % | aprovado | 300 |
| rede_multi | 8,38 % | aprovado | 300 |

## Erro por propulsão (MJ/km)

| Candidato | Combustão | Elétrico | Híbrido | Plug-in |
|---|---|---|---|---|
| linear | 6,04 % | 7,39 % | 7,31 % | 16,57 % |
| floresta | 5,36 % | 4,8 % | 11,47 % | 15,83 % |
| xgboost | 4,66 % | 4,91 % | 7,6 % | 11,77 % |
| rede_por_alvo | 6,72 % | 6,87 % | 9,6 % | 17,29 % |
| rede_multi | 6,15 % | 8,04 % | 9,39 % | 20,7 % |

## Figuras

![Erro por propulsão](figuras/erro_por_propulsao.png)

![Previsto × real do vencedor](figuras/previsto_real.png)
