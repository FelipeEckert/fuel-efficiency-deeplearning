from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from eficiencia.esquema import ENTRADAS, carregar_base
from eficiencia.prever import prever


def veiculo(**kw):  # A3 Sedan da base como ponto de partida
    v = carregar_base()[ENTRADAS].iloc[0].to_dict()
    v.update(kw)
    return v


def test_prever_saidas_e_faixa(pacote_rapido):
    r = prever(veiculo(), pacote_rapido).iloc[0]
    assert r["mj_km_min"] < r["mj_km"] < r["mj_km_max"]
    assert r["mj_km"] == pytest.approx(0.55 * r["mj_km_urbano"] + 0.45 * r["mj_km_estrada"])
    assert np.isnan(r["kml_etanol"]) and np.isnan(r["kml_diesel"])  # A3 é gasolina


def test_eletrico_km_l_equivalente(pacote_rapido):
    df = carregar_base()
    v = df[df.propulsao == "Elétrico"][ENTRADAS].iloc[0].to_dict()
    r = prever(v, pacote_rapido).iloc[0]
    assert r["kml_eletrico_eq_urbano"] == pytest.approx(20.29 / r["mj_km_urbano"])
    assert np.isnan(r["kml_gasolina"])


def test_combinacao_inedita_avisa(pacote_rapido):
    r = prever(
        veiculo(propulsao="Plug-in", combustivel="diesel", bateria_kwh=20, autonomia_eletrica_km=40),
        pacote_rapido,
    ).iloc[0]
    assert r["avisos"] and np.isfinite(r["mj_km"]) and np.isfinite(r["kml_diesel"])


def test_entrada_invalida_levanta_todas_as_mensagens(pacote_rapido):
    with pytest.raises(ValueError) as erro:
        prever(veiculo(f2_n_kmh2=-0.01, potencia_kw=0), pacote_rapido)
    linhas = str(erro.value).split("\n")
    assert len(linhas) == 2 and all(linha.startswith("linha 1: ") for linha in linhas)


def test_varias_linhas_mantem_indice_original(pacote_rapido):
    df = pd.DataFrame([veiculo(), veiculo(propulsao="Elétrico", combustivel="gasolina")], index=[10, 20])
    with pytest.raises(ValueError, match="linha 2: Combustível"):
        prever(df, pacote_rapido)
    r = prever(df.iloc[[0]], pacote_rapido)
    assert list(r.index) == [10]


def test_quantil_grande_nao_gera_faixa_invalida(pacote_rapido):
    quantis = {p: {s: 1.5 for s in d} for p, d in pacote_rapido.quantis.items()}
    pacote = replace(pacote_rapido, quantis=quantis)
    df = pd.DataFrame(
        [
            veiculo(),
            veiculo(propulsao="Elétrico", combustivel="eletrico", bateria_kwh=100, autonomia_eletrica_km=400),
        ]
    )
    r = prever(df, pacote)
    for coluna in r.columns:
        if coluna.endswith(("_min", "_max")):
            valores = r[coluna].to_numpy(dtype=float)
            assert np.all(np.isnan(valores) | ((valores > 0) & np.isfinite(valores))), coluna
    assert np.isnan(r["mj_km_min"].iloc[0]) and r["mj_km_max"].iloc[0] > r["mj_km"].iloc[0]
    assert np.isnan(r["kml_eletrico_eq_urbano_max"].iloc[1])


@pytest.mark.parametrize("propulsao", ["Combustão", "Híbrido"])
def test_kml_de_combustao_e_hibrido_deriva_do_mj_km(pacote_rapido, propulsao):
    df = carregar_base()
    v = df[(df.propulsao == propulsao) & (df.combustivel == "flex")][ENTRADAS].iloc[0].to_dict()
    r = prever(v, pacote_rapido).iloc[0]
    for comb in ("gasolina", "etanol"):
        for c, i in (("urbano", 0), ("estrada", 1)):
            fator = pacote_rapido.fatores_kml[comb][i]
            assert r[f"kml_{comb}_{c}"] * r[f"mj_km_{c}"] == pytest.approx(fator, rel=1e-9)
            assert r[f"kml_{comb}_{c}_min"] == pytest.approx(fator / r[f"mj_km_{c}_max"], rel=1e-9)
            assert r[f"kml_{comb}_{c}_max"] == pytest.approx(fator / r[f"mj_km_{c}_min"], rel=1e-9)


def test_kml_de_plugin_continua_vindo_do_modelo(pacote_rapido):
    df = carregar_base()
    v = df[df.propulsao == "Plug-in"][ENTRADAS].iloc[0].to_dict()
    r = prever(v, pacote_rapido).iloc[0]
    previsto = pacote_rapido.modelo.predict(pd.DataFrame([v])[ENTRADAS]).iloc[0]
    assert r["kml_gasolina_urbano"] == pytest.approx(previsto["kml_gasolina_urbano"])
