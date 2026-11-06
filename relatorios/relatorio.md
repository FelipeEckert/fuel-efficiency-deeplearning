# Relatório de treino: eficiência energética

Data: 2026-11-05 · hash da base: `dd888117`

**Vencedor: xgboost**, o de menor erro entre os que passam no teste de monotonia.
Hiperparâmetros: `{'n_estimators': 969, 'max_depth': 5, 'learning_rate': 0.032637691707401356, 'subsample': 0.6437132638022895, 'colsample_bytree': 0.5795595752000016, 'min_child_weight': 5, 'reg_lambda': 0.002883631811041725}`

## Candidatos

| Candidato | Erro MJ/km | Monotonia | Tentativas |
|---|---|---|---|
| linear | 7,52 % | aprovado | 900 |
| floresta | 7,02 % | aprovado | 674 |
| xgboost | 5,77 % | aprovado | 600 |
| rede_por_alvo | 7,16 % | aprovado | 301 |
| rede_multi | 7,72 % | aprovado | 300 |

## Erro por propulsão (MJ/km)

| Candidato | Combustão | Elétrico | Híbrido | Plug-in |
|---|---|---|---|---|
| linear | 6,04 % | 7,39 % | 7,31 % | 16,57 % |
| floresta | 5,36 % | 4,8 % | 11,47 % | 15,83 % |
| xgboost | 4,63 % | 4,92 % | 7,61 % | 11,84 % |
| rede_por_alvo | 5,81 % | 5,55 % | 8,68 % | 16,59 % |
| rede_multi | 6,12 % | 6,71 % | 9 % | 17,53 % |

## Figuras

![Erro por propulsão](figuras/erro_por_propulsao.png)

![Previsto × real do vencedor](figuras/previsto_real.png)
