"""Registro dos candidatos de modelo, indexados pelo nome."""

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
