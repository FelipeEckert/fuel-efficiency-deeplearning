import pytest

from eficiencia.artefatos import PASTA_MODELO, carregar
from eficiencia.esquema import carregar_base
from eficiencia.monotonia import checar_monotonia


@pytest.mark.skipif(not (PASTA_MODELO / "pacote.joblib").exists(), reason="modelo ainda não treinado")
def test_artefato_versionado_monotono():
    p = carregar(PASTA_MODELO)
    assert checar_monotonia(p.modelo, carregar_base()) == []
