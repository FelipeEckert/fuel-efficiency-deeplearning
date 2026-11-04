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
    assert erros == ["linha 1: Massa em ordem de marcha (kg) não é número"] and ok.empty


def test_validar_normaliza_categoria_sem_acento_e_caixa():
    ok, erros = validar(pd.DataFrame([veiculo(propulsao="combustao", categoria=" g ")]))
    assert erros == []
    assert ok.iloc[0]["propulsao"] == "Combustão"
    assert ok.iloc[0]["categoria"] == "G"


def test_limites_fisicos():
    _, erros = validar(pd.DataFrame([veiculo(f2_n_kmh2=-0.01, potencia_kw=0)]))
    assert len(erros) == 2 and all(e.startswith("linha 1: ") for e in erros)


def test_coerencia_propulsao():
    _, erros = validar(pd.DataFrame([veiculo(propulsao="Elétrico", combustivel="gasolina")]))
    assert any("Combustível" in e for e in erros)
    ok, _ = validar(pd.DataFrame([veiculo(propulsao="Combustão", bateria_kwh=60)]))
    assert ok.iloc[0]["bateria_kwh"] == 0
