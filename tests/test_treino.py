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
