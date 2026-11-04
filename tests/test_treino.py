import json

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
    assert t.pacote.metadados["rapido"] is True


def test_rapido_sem_pastas_nao_toca_o_modelo_real(tmp_path, monkeypatch):
    monkeypatch.setattr(treino, "RAIZ", tmp_path)
    treinar(rapido=True, candidatos=["linear"])
    assert (tmp_path / "modelos/rapido/pacote.joblib").exists()
    assert (tmp_path / "relatorios/rapido/metricas.json").exists()
    assert not (tmp_path / "modelos/vencedor").exists()
    assert not (tmp_path / "relatorios/metricas.json").exists()


def test_nenhum_candidato_aprovado_salva_relatorio_sem_pacote(tmp_path, monkeypatch):
    monkeypatch.setattr(treino, "checar_monotonia", lambda modelo, df: ["mj_km_urbano viola monotonia"])
    with pytest.raises(RuntimeError, match="linear"):
        treinar(
            rapido=True,
            candidatos=["linear"],
            pasta_modelo=tmp_path / "modelo",
            pasta_relatorios=tmp_path / "relatorios",
        )
    metricas = json.loads((tmp_path / "relatorios/metricas.json").read_text(encoding="utf-8"))
    assert metricas["vencedor"] is None
    assert metricas["candidatos"]["linear"]["monotonia"] == "reprovado"
    relatorio = (tmp_path / "relatorios/relatorio.md").read_text(encoding="utf-8")
    assert "mj_km_urbano viola monotonia" in relatorio
    assert not (tmp_path / "modelo" / "pacote.joblib").exists()


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
