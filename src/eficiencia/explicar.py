"""Explicação de uma previsão do XGBoost: contribuição de cada entrada em log(alvo), via pred_contribs."""

import pandas as pd
import xgboost

from eficiencia.esquema import DERIVADAS, ENTRADAS


def explicavel(modelo) -> bool:
    estimadores = getattr(modelo, "estimadores", {})
    return bool(estimadores) and all(hasattr(e, "get_booster") for e in estimadores.values())


def explicar(modelo, veiculo: dict, alvo: str) -> tuple[float, pd.Series]:
    Xt = modelo.prep.transform(pd.DataFrame([veiculo]))
    phi = modelo.estimadores[alvo].get_booster().predict(xgboost.DMatrix(Xt), pred_contribs=True)[0]
    # one-hot "coluna=valor" volta para a coluna original; a numérica não tem "=" e fica como está
    contrib = pd.Series(phi[:-1], index=[n.split("=")[0] for n in modelo.prep.nomes])
    return float(phi[-1]), contrib.groupby(level=0).sum().reindex(ENTRADAS + DERIVADAS, fill_value=0.0)
