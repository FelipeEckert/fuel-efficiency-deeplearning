import pandas as pd

from eficiencia.entrada import validar
from eficiencia.esquema import ENTRADAS, carregar_base


def veiculo(**kw):  # A3 Sedan da base como ponto de partida
    v = carregar_base()[ENTRADAS].iloc[0].to_dict()
    v.update(kw)
    return v


def test_validar_texto_e_virgula():
    df = pd.DataFrame([veiculo(f2_n_kmh2="0,035", massa_om_kg="abc", categoria=" g ")])
    ok, erros = validar(df)
    assert erros == ["row 1: Curb mass (kg) is not a number"] and ok.empty


def test_validar_normaliza_categoria_sem_acento_e_caixa():
    ok, erros = validar(pd.DataFrame([veiculo(propulsao="combustao", categoria=" g ")]))
    assert erros == []
    assert ok.iloc[0]["propulsao"] == "Combustão"
    assert ok.iloc[0]["categoria"] == "G"


def test_limites_fisicos():
    _, erros = validar(pd.DataFrame([veiculo(f2_n_kmh2=-0.01, potencia_kw=0)]))
    assert len(erros) == 2 and all(e.startswith("row 1: ") for e in erros)


def test_coerencia_propulsao():
    _, erros = validar(pd.DataFrame([veiculo(propulsao="Elétrico", combustivel="gasolina")]))
    assert any(e.startswith("row 1: Fuel must be Electric") for e in erros)
    ok, _ = validar(pd.DataFrame([veiculo(propulsao="Combustão", bateria_kwh=60)]))
    assert ok.iloc[0]["bateria_kwh"] == 0


def test_eletrico_dispensa_cilindrada():
    ok, erros = validar(
        pd.DataFrame(
            [veiculo(propulsao="Elétrico", combustivel="eletrico", cilindrada_cm3=None, bateria_kwh=100)]
        )
    )
    assert erros == [] and ok.iloc[0]["cilindrada_cm3"] == 0


def test_booleano_nao_e_numero():
    _, erros = validar(pd.DataFrame([veiculo(potencia_kw=True)]))
    assert erros == ["row 1: Power (kW) is not a number"]


def test_minimos_por_propulsao():
    _, erros = validar(pd.DataFrame([veiculo(propulsao="Combustão", cilindrada_cm3=0)]))
    assert erros == ["row 1: Displacement (cm³) must be greater than zero for a Combustion powertrain"]
    _, erros = validar(pd.DataFrame([veiculo(propulsao="Plug-in", combustivel="flex", bateria_kwh=0)]))
    assert erros == ["row 1: Battery (kWh) must be greater than zero for a Plug-in hybrid powertrain"]
