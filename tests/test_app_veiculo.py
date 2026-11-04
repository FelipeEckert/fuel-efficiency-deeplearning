import io
import sys
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from eficiencia.esquema import ENTRADAS, carregar_base
from eficiencia.fisica import classe_inercia
from eficiencia.prever import prever

PAGINA = (
    Path(__file__).resolve().parents[1] / "app/pages/1_Veiculo.py"
)  # AppTest resolve caminho relativo a tests/
sys.path.insert(0, str(PAGINA.parents[1]))  # pasta app/, para importar formulario nos testes


def _pagina():
    return AppTest.from_file(str(PAGINA), default_timeout=60).run()


def _metrica(at, rotulo):
    return next(m for m in at.metric if m.label == rotulo)


def test_pagina_veiculo_calcula(pacote_rapido):
    at = _pagina()
    assert not at.exception
    assert any("MJ/km" in m.label for m in at.metric)


def test_salvar_cenario(pacote_rapido):
    at = _pagina()
    at.button(key="salvar_cenario").click().run()
    assert len(at.session_state["cenarios"]) == 1


def test_eletrico_preset(pacote_rapido):
    at = _pagina()
    at.selectbox(key="veiculo_propulsao").set_value("Elétrico").run()
    assert at.session_state["veiculo_combustivel"] == "eletrico"
    assert at.session_state["veiculo_cilindrada_cm3"] == 0.0
    assert at.session_state["veiculo_transmissao"] == "direta"


def test_classe_inercia_preenchida_e_edicao_mantida(pacote_rapido):
    at = _pagina()
    at.number_input(key="veiculo_massa_om_kg").set_value(900.0).run()
    assert at.session_state["veiculo_classe_inercia_kg"] == classe_inercia(900.0)
    at.number_input(key="veiculo_classe_inercia_kg").set_value(1200.0).run()
    assert at.session_state["veiculo_classe_inercia_kg"] == 1200.0
    assert at.session_state["veiculo_massa_om_kg"] == 900.0


def test_e_se_massa_muda_resultado_e_classe(pacote_rapido):
    at = _pagina()
    base = at.session_state["veiculo_massa_om_kg"]
    at.slider(key="es_massa").set_value(-20).run()
    delta = _metrica(at, "MJ/km com as mudanças").delta
    assert delta.startswith("-")  # consumo menor: seta para baixo
    at.button(key="salvar_cenario").click().run()
    salvo = at.session_state["cenarios"][0]["veiculo"]
    assert salvo["massa_om_kg"] == base * 0.8
    assert salvo["classe_inercia_kg"] == classe_inercia(base * 0.8)


def test_dois_cenarios_base_e_diferenca(pacote_rapido):
    at = _pagina()
    at.button(key="salvar_cenario").click().run()
    at.slider(key="es_massa").set_value(10).run()
    at.button(key="salvar_cenario").click().run()
    nomes = [c["nome"] for c in at.session_state["cenarios"]]
    assert nomes == ["Cenário 1", "Cenário 2"]
    mj = [c["resultado"]["mj_km"] for c in at.session_state["cenarios"]]
    assert mj[1] > mj[0]
    grafico = at.get("plotly_chart")[0].proto.spec
    assert "base" in grafico and "%)" in grafico  # rótulos de diferença (spec JSON escapa "/")


def test_remover_cenario(pacote_rapido):
    at = _pagina()
    at.button(key="salvar_cenario").click().run()
    at.slider(key="es_massa").set_value(10).run()
    at.button(key="salvar_cenario").click().run()
    at.button(key="remover_1").click().run()
    assert len(at.session_state["cenarios"]) == 1


def test_nomes_de_cenario_unicos(pacote_rapido):
    at = _pagina()
    at.text_input(key="nome_cenario").input("Teste").run()
    at.button(key="salvar_cenario").click().run()
    at.text_input(key="nome_cenario").input("Teste").run()
    at.button(key="salvar_cenario").click().run()
    nomes = [c["nome"] for c in at.session_state["cenarios"]]
    assert nomes == ["Teste", "Teste (2)"]


def test_exportar_excel(pacote_rapido):
    from formulario import excel_cenarios

    at = _pagina()
    at.button(key="salvar_cenario").click().run()
    assert at.get("download_button")
    v = carregar_base()[ENTRADAS].iloc[0].to_dict()
    r = prever(v, pacote_rapido).iloc[0].to_dict()
    dados = excel_cenarios([{"nome": "base", "veiculo": v, "resultado": r}])
    assert len(dados) > 0
    assert len(pd.read_excel(io.BytesIO(dados))) == 1


def test_entrada_invalida_mostra_erro(pacote_rapido):
    at = _pagina()
    at.selectbox(key="veiculo_propulsao").set_value("Híbrido").run()  # bateria padrão é 0
    assert any("Bateria" in e.value for e in at.error)
