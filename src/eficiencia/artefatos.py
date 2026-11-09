"""Pacote do modelo vencedor: modelo, quantis, faixas e metadados, salvos com joblib e metadados em JSON."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import joblib

from eficiencia.dados import RAIZ
from eficiencia.modelos.base import Candidato

PASTA_MODELO = RAIZ / "modelos/vencedor"


@dataclass
class Pacote:
    algoritmo: str
    modelo: Candidato
    quantis: dict[str, dict[str, float]]
    faixas: dict
    fatores_kml: dict[str, tuple[float, float]]
    referencias: dict[str, dict[str, float]]
    metadados: dict


def salvar(pacote: Pacote, pasta: Path = PASTA_MODELO) -> None:
    pasta.mkdir(parents=True, exist_ok=True)
    joblib.dump(pacote, pasta / "pacote.joblib")
    (pasta / "metadados.json").write_text(
        json.dumps(pacote.metadados, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def carregar(pasta: Path | None = None) -> Pacote:
    if pasta is None:
        pasta = Path(os.environ.get("EFICIENCIA_MODELO", PASTA_MODELO))
    return joblib.load(pasta / "pacote.joblib")
