import pytest

from eficiencia.esquema import carregar_base
from eficiencia.fisica import (
    classe_inercia,
    combinar_kml,
    combinar_mj,
    combustiveis_aplicaveis,
    fatores_kml,
    kml_eletrico_eq,
)


@pytest.mark.parametrize("massa,classe", [(1490, 1644), (1650, 1758), (300, 454), (3000, 2495)])
def test_classe_inercia(massa, classe):
    assert classe_inercia(massa) == classe


def test_combinar():
    assert combinar_mj(2.23326400123257, 1.598742623945293) == pytest.approx(1.947729381453296, abs=1e-9)
    assert round(combinar_kml(10.3, 13.5), 1) == 11.5
    assert kml_eletrico_eq(0.5) == pytest.approx(40.58)


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
