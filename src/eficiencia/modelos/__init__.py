"""Registro dos candidatos de modelo, indexados pelo nome."""

from eficiencia.modelos.base import Candidato
from eficiencia.modelos.linear import Linear

CANDIDATOS: dict[str, type[Candidato]] = {Linear.nome: Linear}
