import numpy as np

from eficiencia.esquema import ALVOS, ENTRADAS, carregar_base
from eficiencia.modelos import CANDIDATOS


def test_linear_fit_predict_formato():
    df = carregar_base()
    m = CANDIDATOS["linear"](alpha=1.0).fit(df[ENTRADAS], df[ALVOS])
    prev = m.predict(df[ENTRADAS].iloc[:5])
    assert list(prev.columns) == ALVOS and (prev.index == df.index[:5]).all()
    assert (prev > 0).all().all()


def test_por_alvo_ignora_alvo_ausente():
    df = carregar_base()
    m = CANDIDATOS["linear"](alpha=1.0).fit(df[ENTRADAS], df[ALVOS])  # etanol só em 168 linhas
    assert np.isfinite(m.predict(df[ENTRADAS])["kml_etanol_urbano"]).all()
