import numpy as np
import pandas as pd
import pytest

from eficiencia.cenario import fator_suave
from eficiencia.entrada import validar
from eficiencia.esquema import ENTRADAS, carregar_base
from eficiencia.fisica import combinar_mj


def _validado(linha: pd.Series) -> dict:
    return validar(pd.DataFrame([linha[ENTRADAS].to_dict()]))[0].iloc[0].to_dict()


def _bruto(modelo, v: dict) -> float:
    p = modelo.predict(pd.DataFrame([v])[ENTRADAS])
    return float(combinar_mj(p["mj_km_urbano"], p["mj_km_estrada"]).iloc[0])


@pytest.fixture(scope="module")
def veiculos():
    base = carregar_base()
    return [_validado(base.iloc[i]) for i in (0, len(base) // 2, len(base) - 1)]


def test_sem_mudanca_fator_e_um(pacote_rapido, veiculos):
    for v in veiculos:
        assert fator_suave(pacote_rapido.modelo, v, v) == 1.0


def test_varredura_f2_monotona_e_sem_saltos_maiores(pacote_rapido, veiculos):
    m = pacote_rapido.modelo
    for v in veiculos:
        passos = np.linspace(-0.20, 0.20, 41)
        suave, bruto = [], []
        for d in passos:
            alt = {**v, "f2_n_kmh2": v["f2_n_kmh2"] * (1 + d)}
            suave.append(fator_suave(m, v, alt))
            bruto.append(_bruto(m, alt) / _bruto(m, v))
        assert np.all(np.diff(suave) >= -1e-12)
        # sem saltos maiores que os do modelo; 1 % de folga porque a média de ±5 % desloca de leve uma curva já suave
        assert np.diff(suave).max() <= np.diff(bruto).max() * 1.01 + 1e-12


def test_escalar_saidas_mantem_kml_vezes_mj(pacote_rapido, veiculos):
    from eficiencia.cenario import escalar_saidas
    from eficiencia.prever import prever

    linha = prever(veiculos[0], pacote_rapido).iloc[0]
    nova = escalar_saidas(linha, 1.07)
    assert nova.mj_km == pytest.approx(linha.mj_km * 1.07)
    for s in ("gasolina_urbano", "etanol_estrada", "eletrico_eq"):
        for suf in ("", "_min", "_max"):
            a, b = linha[f"kml_{s}{suf}"], nova[f"kml_{s}{suf}"]
            assert (pd.isna(a) and pd.isna(b)) or b == pytest.approx(a / 1.07)
    ref = "mj_km_urbano"
    if pd.notna(linha.kml_gasolina_urbano):
        assert nova.kml_gasolina_urbano * nova[ref] == pytest.approx(linha.kml_gasolina_urbano * linha[ref])
