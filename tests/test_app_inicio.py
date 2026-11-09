from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"


def test_roteador_abre_inicio_sem_erro():
    at = AppTest.from_file(str(APP / "Inicio.py"), default_timeout=60).run()
    assert not at.exception
    assert any("Vehicle energy efficiency" in t.value for t in at.title)


def test_tema_roxo():
    toml = (Path(__file__).resolve().parents[1] / ".streamlit/config.toml").read_text(encoding="utf-8")
    assert 'primaryColor = "#7C3AED"' in toml and "Inter" in toml
