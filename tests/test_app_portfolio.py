import sys
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from eficiencia.portfolio import COLUNAS_PORTFOLIO

PAGINA = (
    Path(__file__).resolve().parents[1] / "app/pages/2_Portfolio.py"
)  # AppTest resolve caminho relativo a tests/
sys.path.insert(0, str(PAGINA.parents[1]))  # pasta app/, para importar formulario nos testes


def _pagina(portfolio=None):
    at = AppTest.from_file(str(PAGINA), default_timeout=60)
    if portfolio is not None:
        at.session_state["portfolio"] = portfolio
    return at.run()


def _tabela(linhas):
    df = pd.DataFrame(columns=COLUNAS_PORTFOLIO)
    for nome, volume, mj in linhas:
        linha = {c: np.nan for c in COLUNAS_PORTFOLIO}
        linha.update(nome=nome, volume=volume, mj_km_conhecido=mj)
        df = pd.concat([df, pd.DataFrame([linha])], ignore_index=True)
    return df


def _metrica(at, rotulo):
    return next(m for m in at.metric if m.label == rotulo)


def test_pagina_portfolio_vazia_e_com_dados(pacote_rapido):
    at = _pagina()
    assert not at.exception
    at = _pagina(_tabela([("Carro A", 100, 1.5), ("Carro B", 300, 2.0)]))
    assert not at.exception


def test_media_ponderada_aparece_formatada(pacote_rapido):
    at = _pagina(_tabela([("Carro A", 100, 1.5), ("Carro B", 300, 2.0)]))
    assert "1,88" in _metrica(at, "MJ/km médio do portfólio").value


def test_volume_zero_mostra_erro_da_linha(pacote_rapido):
    at = _pagina(_tabela([("Carro A", 0, 1.5)]))
    assert any("linha 1" in erro.value for erro in at.error)


def test_adicionar_veiculo_pelo_formulario(pacote_rapido):
    at = _pagina()
    at.text_input(key="portfolio_nome").input("Carro novo").run()
    at.number_input(key="portfolio_volume").set_value(200.0).run()
    at.button(key="adicionar_veiculo").click().run()
    assert len(at.session_state["portfolio"]) == 1
    assert at.session_state["portfolio"].iloc[0]["nome"] == "Carro novo"
