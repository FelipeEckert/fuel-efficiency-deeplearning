"""Registro dos candidatos de modelo, indexados pelo nome."""

from eficiencia.modelos.base import Candidato
from eficiencia.modelos.floresta import Floresta
from eficiencia.modelos.linear import Linear
from eficiencia.modelos.xgb import XGBoost

CANDIDATOS: dict[str, type[Candidato]] = {c.nome: c for c in (Linear, Floresta, XGBoost)}
