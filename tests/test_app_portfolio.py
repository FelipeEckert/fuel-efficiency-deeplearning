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
    at.run()  # a linha adicionada não pode ser aplicada de novo pelo editor
    assert len(at.session_state["portfolio_atual"]) == 1


def test_edicao_sobrevive_a_trocar_de_pagina(pacote_rapido):
    at = _pagina(_tabela([("Carro A", 100, 1.5)]))
    at.session_state["portfolio_atual"] = _tabela([("Carro A", 100, 1.5), ("Carro B", 300, 2.0)])
    del at.session_state["editor_0"]  # o widget some quando o usuário sai da página
    at.run()
    assert "1,88" in _metrica(at, "MJ/km médio do portfólio").value
    assert len(at.session_state["portfolio_atual"]) == 2


def test_valor_fora_das_opcoes_e_reportado(pacote_rapido):
    df = _tabela([("Carro A", 100, 1.5)])
    df.loc[0, "categoria"] = "XYZ"
    at = _pagina(df)
    assert not at.exception
    assert any("linha 1" in erro.value for erro in at.error)


def test_sem_modelo_calcula_so_linhas_conhecidas(pacote_rapido, monkeypatch):
    import formulario

    def sem_modelo():
        raise FileNotFoundError("sem modelo")

    monkeypatch.setattr(formulario, "modelo", sem_modelo)
    df = _tabela([("A", 100, 1.5), ("B", 300, np.nan), ("C", 200, 2.0), ("D", 100, np.nan), ("E", 0, 1.0)])
    at = _pagina(df)
    assert not at.exception
    assert any("Sem modelo" in i.value for i in at.info)
    assert _metrica(at, "MJ/km médio do portfólio").value == "1,83 MJ/km"
    erros = [e.value for e in at.error]
    assert any(e.startswith("linha 2:") for e in erros)
    assert any(e.startswith("linha 4:") for e in erros)
    assert any(e.startswith("linha 5:") for e in erros)  # volume zero, número da linha na tabela inteira


def test_sem_modelo_sem_linhas_conhecidas_nao_quebra(pacote_rapido, monkeypatch):
    import formulario

    def sem_modelo():
        raise FileNotFoundError("sem modelo")

    monkeypatch.setattr(formulario, "modelo", sem_modelo)
    at = _pagina(_tabela([("B", 300, np.nan)]))
    assert not at.exception
    assert _metrica(at, "MJ/km médio do portfólio").value == "—"
    assert any(e.value.startswith("linha 1:") for e in at.error)
    at = _pagina()
    assert not at.exception
