import numpy as np

from eficiencia.esquema import ALVOS, DERIVADAS, ENTRADAS, carregar_base
from eficiencia.explicar import explicar, explicavel
from eficiencia.modelos import CANDIDATOS


def _modelo(nome):
    df = carregar_base()
    cls = CANDIDATOS[nome]
    return cls(**cls.PARAMS_RAPIDOS).fit(df[ENTRADAS], df[ALVOS]), df


def test_contribuicoes_fecham_com_a_previsao():
    m, df = _modelo("xgboost")
    primeiro_plugin = df.index[df["propulsao"].eq("Plug-in")][0]
    for linha in (df.index[0], primeiro_plugin):
        veiculo = df[ENTRADAS].loc[linha].to_dict()
        for alvo in ("mj_km_urbano", "mj_km_estrada"):
            base, contrib = explicar(m, veiculo, alvo)
            assert list(contrib.index) == ENTRADAS + DERIVADAS  # uma linha por entrada; one-hot já agregado
            previsto = m.predict(df[ENTRADAS].loc[[linha]])[alvo].iloc[0]
            assert np.isclose(base + contrib.sum(), np.log(previsto), atol=1e-4)


def test_explicavel_so_xgboost():
    assert explicavel(_modelo("xgboost")[0])
    assert not explicavel(_modelo("linear")[0])
