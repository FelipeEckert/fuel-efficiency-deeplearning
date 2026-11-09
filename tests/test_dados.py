import pandas as pd
import pytest

from eficiencia.dados import (
    CV_PARA_KW,
    ENTRADA,
    SAIDA,
    checar,
    grupo_cv,
    limpar,
    potencia_kw,
    salvar,
    transmissao,
)

# the source spreadsheet is not versioned: these tests run only where it exists locally
precisa_planilha = pytest.mark.skipif(
    not ENTRADA.exists(), reason="PBEV spreadsheet not available (data/raw)"
)


@pytest.fixture(scope="module")
def limpo():
    return limpar()


@precisa_planilha
def test_invariantes(limpo):
    df, _ = limpo
    checar(df)


@precisa_planilha
def test_regenerar_igual_ao_commitado(tmp_path):
    df, _ = limpar()
    salvar(df, tmp_path / "x.csv")
    novo = pd.read_csv(tmp_path / "x.csv", dtype={"lcvm": str})
    atual = pd.read_csv(SAIDA, dtype={"lcvm": str})
    pd.testing.assert_frame_equal(novo, atual)


@precisa_planilha
def test_contagens(limpo):
    df, rel = limpo
    assert len(df) == 510
    assert df["grupo_cv"].nunique() == 258
    assert rel["F2 convertido de N/(m/s)² para N/(km/h)²"] == 10
    assert int(df["bateria_estimada"].sum()) == 17


@pytest.mark.parametrize(
    ("txt", "marca", "esperado"),
    [
        ("176 cv (G) e 176 cv (E)", "JEEP", round(176 * CV_PARA_KW, 1)),
        ("170cv (125kW)", "FORD", 125),
        ("120 + 10", "MERCEDES-BENZ", 130),
        ("150/110", "JAC", 110),
        ("116", "VOLKSWAGEN", round(116 * CV_PARA_KW, 1)),
        ("58,8 kW", "HYUNDAI", 58.8),
    ],
)
def test_potencia_kw(txt, marca, esperado):
    assert potencia_kw(txt, marca) == esperado


def test_transmissao():
    assert transmissao("A - 6", "Combustão") == ("automatica", 6.0)
    assert transmissao("N.A.", "Elétrico") == ("direta", 1.0)


def test_grupo_cv():
    assert (
        grupo_cv("TOYOTA", "HILUX DIESEL 4X4 AT")
        == grupo_cv("TOYOTA", "Hilux Diesel 4x4 MT")
        == "TOYOTA HILUX"
    )
    assert grupo_cv("HONDA", "HR-V (mod. 26)") == "HONDA HR-V"
    assert grupo_cv("CHEVROLET", "ONIX PLUS") != grupo_cv("CHEVROLET", "ONIX")
