import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from eficiencia.artefatos import carregar
from eficiencia.esquema import caminho_base
from eficiencia.treino import treinar

BASE_REAL = caminho_base().exists()


def pytest_configure(config):
    """Without the local PBEV table (it is not versioned), the suite runs on a synthetic one."""
    if not BASE_REAL:
        from dados_sinteticos import salvar

        pasta = Path(tempfile.mkdtemp(prefix="eficiencia-sintetico-"))
        os.environ["EFICIENCIA_BASE"] = str(salvar(pasta / "base_sintetica.csv"))


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
