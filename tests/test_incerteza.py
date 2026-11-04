import numpy as np
import pandas as pd
import pytest

from eficiencia.esquema import ALVOS
from eficiencia.incerteza import SAIDAS_COM_FAIXA, aplicar, calcular_quantis


def test_quantil_conformal_exato():
    df = pd.DataFrame(
        {"propulsao": ["Combustão"] * 10, "mj_km_urbano": 1.0, "mj_km_estrada": 1.0, "mj_km": 1.0}
    )
    for a in ALVOS[2:]:
        df[a] = np.nan
    oof = pd.DataFrame(
        {"mj_km_urbano": 1 + np.arange(1, 11) / 100, "mj_km_estrada": 1 + np.arange(1, 11) / 100}
    )
    assert calcular_quantis(df, oof)["Combustão"]["mj_km_urbano"] == pytest.approx(0.10)


def test_saidas_com_faixa():
    assert SAIDAS_COM_FAIXA == ALVOS + ["mj_km", "kml_gasolina", "kml_etanol"]


def test_aplicar_faixa():
    assert aplicar(10.0, 0.1) == pytest.approx((9.0, 11.0))
