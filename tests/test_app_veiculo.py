from pathlib import Path

from streamlit.testing.v1 import AppTest

PAGINA = (
    Path(__file__).resolve().parents[1] / "app/pages/1_Veiculo.py"
)  # AppTest resolve caminho relativo a tests/


def test_pagina_veiculo_calcula(pacote_rapido):
    at = AppTest.from_file(str(PAGINA), default_timeout=60).run()
    assert not at.exception
    assert any("MJ/km" in m.label for m in at.metric)


def test_salvar_cenario(pacote_rapido):
    at = AppTest.from_file(str(PAGINA), default_timeout=60).run()
    at.button(key="salvar_cenario").click().run()
    assert len(at.session_state["cenarios"]) == 1
