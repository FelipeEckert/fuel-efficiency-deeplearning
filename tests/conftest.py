import os
from types import SimpleNamespace

import pytest

from eficiencia.artefatos import carregar
from eficiencia.treino import treinar


@pytest.fixture(scope="session")
def treino_rapido(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("treino")
    pasta_modelo, pasta_relatorios = tmp / "modelo", tmp / "relatorios"
    os.environ["EFICIENCIA_MODELO"] = str(pasta_modelo)
    os.environ["EFICIENCIA_RELATORIOS"] = str(pasta_relatorios)
    metricas = treinar(
        rapido=True,
        candidatos=["linear", "xgboost"],
        pasta_modelo=pasta_modelo,
        pasta_relatorios=pasta_relatorios,
    )
    pacote = carregar(pasta_modelo)
    return SimpleNamespace(
        pacote=pacote, metricas=metricas, pasta_modelo=pasta_modelo, pasta_relatorios=pasta_relatorios
    )


@pytest.fixture(scope="session")
def pacote_rapido(treino_rapido):
    return treino_rapido.pacote
