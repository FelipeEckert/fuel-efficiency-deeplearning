import pytest

from eficiencia.esquema import ALVOS, ENTRADAS, carregar_base
from eficiencia.modelos import CANDIDATOS, EXPERIMENTAIS
from eficiencia.modelos.restricoes import sinais_monotonia
from eficiencia.monotonia import checar_monotonia
from eficiencia.treino import treinar


@pytest.mark.parametrize(
    "nome", ["linear", "floresta", "xgboost", "rede_por_alvo", "rede_multi", "catboost", "lightgbm"]
)
def test_candidato_monotono_por_construcao(nome):
    df = carregar_base()
    cls = (CANDIDATOS | EXPERIMENTAIS)[nome]
    m = cls(**cls.PARAMS_RAPIDOS).fit(df[ENTRADAS], df[ALVOS])
    assert checar_monotonia(m, df) == []


def test_rede_monotona_mesmo_sem_treino():
    df = carregar_base()
    m = CANDIDATOS["rede_multi"](max_epocas=0).fit(df[ENTRADAS], df[ALVOS])
    assert checar_monotonia(m, df) == []


def test_sinais_monotonia():
    nomes = ["massa_om_kg", "potencia_kw", "f2_n_kmh2", "categoria=G"]
    assert list(sinais_monotonia(nomes, "mj_km_urbano")) == [1, 0, 1, 0]
    assert list(sinais_monotonia(nomes, "kml_gasolina_estrada")) == [-1, 0, -1, 0]


def test_treino_rapido_todos_aprovados(tmp_path):
    m = treinar(rapido=True, pasta_modelo=tmp_path / "m", pasta_relatorios=tmp_path / "r")
    assert all(c["monotonia"] == "aprovado" for c in m["candidatos"].values())
