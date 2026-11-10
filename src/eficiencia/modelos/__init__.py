"""Registry of the candidate models, indexed by name."""

from eficiencia.modelos.base import Candidato
from eficiencia.modelos.boosting import CatBoost, LightGBM
from eficiencia.modelos.floresta import Floresta
from eficiencia.modelos.linear import Linear
from eficiencia.modelos.rede import RedeMulti, RedePorAlvo
from eficiencia.modelos.xgb import XGBoost, XGBoostAmplo

CANDIDATOS: dict[str, type[Candidato]] = {
    c.nome: c for c in (Linear, Floresta, XGBoost, RedePorAlvo, RedeMulti)
}

# só para experimentos: fora de CANDIDATOS, o treino oficial e o progresso não os enxergam
EXPERIMENTAIS: dict[str, type[Candidato]] = {c.nome: c for c in (XGBoostAmplo, CatBoost, LightGBM)}

# display name of each candidate, for the report and the app
NOMES_MODELOS: dict[str, str] = {
    "linear": "Linear regression",
    "floresta": "Random forest",
    "xgboost": "XGBoost",
    "rede_por_alvo": "Neural network, one per target",
    "rede_multi": "Neural network, multi-output",
    "xgboost_amplo": "XGBoost, wide search space",
    "catboost": "CatBoost",
    "lightgbm": "LightGBM",
}
