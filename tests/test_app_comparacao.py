import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

PAGINA = Path(__file__).resolve().parents[1] / "app/pages/3_Comparacao.py"
MSG_CORROMPIDO = (
    "The training report is incomplete or corrupted. Run the training again: uv run eficiencia-treinar"
)


def test_comparacao_com_metricas(treino_rapido):
    at = AppTest.from_file(str(PAGINA), default_timeout=30).run()
    assert not at.exception and at.dataframe


def test_comparacao_sem_metricas(tmp_path, monkeypatch):
    monkeypatch.setenv("EFICIENCIA_RELATORIOS", str(tmp_path))
    at = AppTest.from_file(str(PAGINA), default_timeout=30).run()
    assert not at.exception and "eficiencia-treinar" in at.info[0].value


def _relatorio(tmp_path, monkeypatch, conteudo: str) -> AppTest:
    (tmp_path / "metricas.json").write_text(conteudo, encoding="utf-8")
    monkeypatch.setenv("EFICIENCIA_RELATORIOS", str(tmp_path))
    return AppTest.from_file(str(PAGINA), default_timeout=30).run()


def _metricas_completas():
    violacoes = [f"violação {i}" for i in range(12)]
    return {
        "vencedor": "xgboost",
        "candidatos": {
            "floresta": {
                "erro_mj": 0.0912,
                "erro_por_propulsao": {"Combustão": 0.09, "Elétrico": 0.10},
                "n_tentativas": 20,
                "monotonia": "aprovado",
                "violacoes": [],
            },
            "linear": {
                "erro_mj": 0.2051,
                "erro_por_propulsao": {"Combustão": 0.2, "Híbrido": 0.21, "Elétrico": 0.22},
                "n_tentativas": 1,
                "monotonia": "reprovado",
                "violacoes": violacoes,
            },
            "xgboost": {
                "erro_mj": 0.0679,
                "erro_por_propulsao": {"Combustão": 0.065, "Híbrido": 0.07, "Elétrico": 0.068},
                "n_tentativas": 50,
                "monotonia": "aprovado",
                "violacoes": [],
            },
        },
    }


def test_comparacao_tabela_vencedor_e_violacoes(tmp_path, monkeypatch):
    at = _relatorio(tmp_path, monkeypatch, json.dumps(_metricas_completas()))
    assert not at.exception
    assert at.success[0].value == "Selected: XGBoost, 6.8 % mean error on MJ/km"

    tabela = at.dataframe[0].value
    assert list(tabela["Model"]) == ["🏆 XGBoost", "Random forest", "Linear regression"]
    assert list(tabela["Overall error (%)"]) == ["6.8", "9.1", "20.5"]
    assert list(tabela.columns) == [
        "Model",
        "Overall error (%)",
        "Combustion (%)",
        "Hybrid (%)",
        "Electric (%)",
        "Monotonicity",
        "Trials",
    ]
    # Random Forest não tem Híbrido: célula com travessão
    assert tabela.loc[1, "Hybrid (%)"] == "—"
    assert tabela.loc[2, "Monotonicity"] == "❌ fail"

    expander = next(e for e in at.expander if "Linear regression" in e.label)
    textos = [t.value for t in expander.get("text")]
    assert textos == ["\n".join(f"violação {i}" for i in range(10))]
    assert any("and 2 more" in m.value for m in expander.get("markdown"))


def test_comparacao_relatorio_corrompido(tmp_path, monkeypatch):
    at = _relatorio(tmp_path, monkeypatch, "{")
    assert not at.exception
    assert at.error[0].value == MSG_CORROMPIDO


def test_comparacao_sem_candidatos(tmp_path, monkeypatch):
    at = _relatorio(tmp_path, monkeypatch, json.dumps({"vencedor": "xgboost", "candidatos": {}}))
    assert not at.exception
    assert at.error[0].value == MSG_CORROMPIDO


def test_comparacao_sem_vencedor(tmp_path, monkeypatch):
    metricas = _metricas_completas()
    metricas["vencedor"] = None
    at = _relatorio(tmp_path, monkeypatch, json.dumps(metricas))
    assert not at.exception and not at.error
    assert at.warning[0].value == "No model passed the monotonicity check in this training run."
    assert len(at.dataframe[0].value) == 3


def test_grafico_de_erro_por_candidato(tmp_path, monkeypatch):
    at = _relatorio(tmp_path, monkeypatch, json.dumps(_metricas_completas()))
    assert not at.exception
    assert len(at.get("plotly_chart")) >= 1
