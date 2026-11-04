import pytest

from eficiencia import treino
from eficiencia.artefatos import carregar
from eficiencia.treino import treinar


def test_treino_rapido(treino_rapido):
    t = treino_rapido
    assert t.pacote.algoritmo == t.metricas["vencedor"] in {"linear", "xgboost"}
    assert t.metricas["candidatos"][t.metricas["vencedor"]]["monotonia"] == "aprovado"
    assert set(t.pacote.quantis) == {"Combustão", "Híbrido", "Plug-in", "Elétrico"}
    assert (t.pasta_relatorios / "metricas.json").exists() and (t.pasta_relatorios / "relatorio.md").exists()
    assert carregar(t.pasta_modelo).algoritmo == t.pacote.algoritmo


def test_nenhum_candidato_aprovado_levanta_erro_e_nao_salva(tmp_path, monkeypatch):
    monkeypatch.setattr(treino, "checar_monotonia", lambda modelo, df: ["mj_km_urbano viola monotonia"])
    with pytest.raises(RuntimeError, match="linear"):
        treinar(
            rapido=True,
            candidatos=["linear"],
            pasta_modelo=tmp_path / "modelo",
            pasta_relatorios=tmp_path / "relatorios",
        )
    assert not (tmp_path / "modelo").exists()
    assert not (tmp_path / "relatorios").exists()


def test_candidato_reprovado_perde_mesmo_com_erro_menor(tmp_path, monkeypatch):
    from eficiencia.modelos import XGBoost

    def monotonia_falsa(modelo, df):
        return ["violação forçada"] if isinstance(modelo, XGBoost) else []

    monkeypatch.setattr(treino, "checar_monotonia", monotonia_falsa)
    metricas = treinar(
        rapido=True,
        candidatos=["linear", "xgboost"],
        storage=None,
        pasta_modelo=tmp_path / "modelo",
        pasta_relatorios=tmp_path / "relatorios",
    )
    # só o linear passa: vence independentemente de qual tem o menor erro de validação cruzada
    assert metricas["vencedor"] == "linear"
    assert metricas["candidatos"]["xgboost"]["monotonia"] == "reprovado"
    assert "violação forçada" in metricas["candidatos"]["xgboost"]["violacoes"]
