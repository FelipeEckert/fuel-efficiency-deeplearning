import pandas as pd

from eficiencia.esquema import ALVOS, carregar_base
from eficiencia.monotonia import checar_monotonia


class Falso:  # MJ/km cai com a massa → deve reprovar
    def predict(self, X):
        v = 3 - X["massa_om_kg"].to_numpy() / 1000
        return pd.DataFrame({a: v for a in ALVOS}, index=X.index)


class Crescente:  # MJ/km sobe com a massa e kml cai: monotonia correta
    def predict(self, X):
        v = 1 + X["massa_om_kg"].to_numpy() / 1000
        return pd.DataFrame({a: (v if a.startswith("mj_km") else 1 / v) for a in ALVOS}, index=X.index)


def test_monotonia_reprova_modelo_invertido():
    assert checar_monotonia(Falso(), carregar_base())


def test_monotonia_aprova_modelo_coerente():
    assert checar_monotonia(Crescente(), carregar_base()) == []


class Ausente:  # devolve NaN: o teste de direção não acusaria nada
    def __init__(self, alvos):
        self.alvos = alvos

    def predict(self, X):
        v = 1 + X["massa_om_kg"].to_numpy() / 1000
        prev = pd.DataFrame({a: (v if a.startswith("mj_km") else 1 / v) for a in ALVOS}, index=X.index)
        prev[self.alvos] = float("nan")
        return prev


def test_monotonia_reprova_mj_km_ausente():
    assert checar_monotonia(Ausente(["mj_km_urbano"]), carregar_base())
    assert checar_monotonia(Ausente(ALVOS), carregar_base())


def test_monotonia_aceita_km_l_ausente():  # combustível que não se aplica ao carro
    assert checar_monotonia(Ausente(["kml_etanol_urbano", "kml_etanol_estrada"]), carregar_base()) == []
