import numpy as np
import pandas as pd
import pytest

from eficiencia.esquema import ALVOS
from eficiencia.incerteza import SAIDAS_COM_FAIXA, aplicar, calcular_quantis, erros_relativos


def test_quantil_conformal_exato():
    df = pd.DataFrame(
        {"propulsao": ["Combustão"] * 10, "mj_km_urbano": 1.0, "mj_km_estrada": 1.0, "mj_km": 1.0}
    )
    for a in ALVOS[2:]:
        df[a] = np.nan
    oof = pd.DataFrame(
        {"mj_km_urbano": 1 + np.arange(1, 11) / 100, "mj_km_estrada": 1 + np.arange(1, 11) / 100}
    )
    assert calcular_quantis(df, oof)["Combustão"]["mj_km_urbano"] == pytest.approx(
        0.10 / 1.10
    )  # erro relativo ao previsto


def test_saidas_com_faixa():
    assert SAIDAS_COM_FAIXA == ALVOS + ["mj_km", "kml_gasolina", "kml_etanol"]


def test_aplicar_faixa():
    assert aplicar(10.0, 0.1) == pytest.approx((9.0, 11.0))


def _sintetico(reais, previsto=1.0, propulsao="Combustão"):
    n = len(reais)
    df = pd.DataFrame(
        {"propulsao": [propulsao] * n, "mj_km_urbano": 1.0, "mj_km_estrada": 1.0, "mj_km": reais}
    )
    for a in ALVOS[2:]:
        df[a] = np.nan
    oof = pd.DataFrame({"mj_km_urbano": [previsto] * n, "mj_km_estrada": [previsto] * n})
    return df, oof


def test_faixa_cobre_90_por_cento_dos_reais():
    erros = np.arange(20) / 100
    reais = 1 + np.where(np.arange(20) % 3 == 0, -erros, erros)
    df, oof = _sintetico(reais)
    q = calcular_quantis(df, oof)["Combustão"]["mj_km"]
    baixo, alto = aplicar(1.0, q)
    assert ((reais >= baixo - 1e-12) & (reais <= alto + 1e-12)).mean() >= 0.9


def test_real_acima_do_previsto_fica_na_faixa():
    # com q medido sobre o real (fórmula antiga), 1,25 vs 1,00 dá q = 0,2 e sai da faixa; relativo ao previsto, q = 0,25
    df, oof = _sintetico([1.25])
    q = calcular_quantis(df, oof)["Combustão"]["mj_km"]
    assert q == pytest.approx(0.25)
    assert aplicar(1.0, q)[1] >= 1.25 - 1e-12


def test_erros_relativos_com_sinal():
    df, oof = _sintetico([1.1, 0.9, 1.2])
    df.loc[2, "propulsao"] = "Híbrido"
    erros = erros_relativos(df, oof)
    assert erros["Combustão"] == pytest.approx([0.1, -0.1])
    assert erros["Híbrido"] == pytest.approx([0.2])
