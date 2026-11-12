import numpy as np
import pandas as pd
import pytest

from eficiencia.esquema import CAMINHO_BASE, carregar_base
from eficiencia.fisica import (
    classe_inercia,
    combinar_kml,
    combinar_mj,
    combustiveis_aplicaveis,
    fator_utilidade,
    fatores_kml,
    fracao_combustao,
    kml_eletrico_eq,
)

# golden values measured on the real PBEV table, which is not versioned
base_real = pytest.mark.skipif(not CAMINHO_BASE.exists(), reason="real PBEV table not available")


@pytest.mark.parametrize("massa,classe", [(1490, 1644), (1650, 1758), (300, 454), (3000, 2495)])
def test_classe_inercia(massa, classe):
    assert classe_inercia(massa) == classe


def test_combinar():
    assert combinar_mj(2.23326400123257, 1.598742623945293) == pytest.approx(1.947729381453296, abs=1e-9)
    assert round(combinar_kml(10.3, 13.5), 1) == 11.5
    assert kml_eletrico_eq(0.5) == pytest.approx(40.58)


@base_real
def test_fatores_kml():
    f = fatores_kml(carregar_base())
    assert f["gasolina"] == pytest.approx((22.4, 20.5), abs=0.2)
    assert f["etanol"] == pytest.approx((15.5, 14.4), abs=0.2)
    assert f["diesel"] == pytest.approx((27.96, 25.57), abs=0.05)


@pytest.mark.parametrize(
    "prop,comb,esperado",
    [
        ("Combustão", "flex", {"gasolina", "etanol"}),
        ("Combustão", "gasolina", {"gasolina"}),
        ("Plug-in", "gasolina", {"gasolina", "eletrico_eq"}),
        ("Plug-in", "flex", {"gasolina", "etanol", "eletrico_eq"}),
        ("Elétrico", "eletrico", {"eletrico_eq"}),
        ("Híbrido", "diesel", {"diesel"}),
    ],
)
def test_combustiveis_aplicaveis(prop, comb, esperado):
    assert combustiveis_aplicaveis(prop, comb) == esperado


def test_fator_utilidade_wltp():
    assert fator_utilidade(0.0) == 0.0
    assert np.isclose(fator_utilidade(40.0), 0.689, atol=1e-3)
    assert np.isclose(fator_utilidade(100.0), 0.902, atol=1e-3)
    d = np.arange(0, 801, 1.0)
    assert (np.diff(fator_utilidade(pd.Series(d))) >= 0).all()  # nunca cai com mais autonomia
    assert fator_utilidade(2000.0) == fator_utilidade(800.0)  # acima de dn não estoura


def test_fracao_combustao_por_propulsao():
    prop = pd.Series(["Combustão", "Híbrido", "Plug-in", "Elétrico", "Plug-in"])
    aut = pd.Series([0.0, 0.0, 40.0, 300.0, np.nan])
    f = fracao_combustao(prop, aut)
    assert f.iloc[:2].tolist() == [1.0, 1.0]
    assert np.isclose(f.iloc[2], 1 - 0.689, atol=1e-3)
    assert f.iloc[3] == 0.0
    assert np.isnan(f.iloc[4])
