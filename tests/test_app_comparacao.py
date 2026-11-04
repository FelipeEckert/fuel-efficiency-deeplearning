from pathlib import Path

from streamlit.testing.v1 import AppTest

PAGINA = Path(__file__).resolve().parents[1] / "app/pages/3_Comparacao.py"


def test_comparacao_com_metricas(treino_rapido):
    at = AppTest.from_file(str(PAGINA), default_timeout=60).run()
    assert not at.exception and at.dataframe


def test_comparacao_sem_metricas(tmp_path, monkeypatch):
    monkeypatch.setenv("EFICIENCIA_RELATORIOS", str(tmp_path))
    at = AppTest.from_file(str(PAGINA)).run()
    assert not at.exception and "eficiencia-treinar" in at.info[0].value
