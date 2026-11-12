import json

import optuna
import pytest

from eficiencia import experimentos, preprocess, treino
from eficiencia.modelos import CANDIDATOS
from eficiencia.modelos.xgb import FAIXAS_AMPLO


def test_na_borda():
    assert experimentos.na_borda({"max_depth": 10, "learning_rate": 0.05}, FAIXAS_AMPLO) == ["max_depth"]


def test_na_borda_log():
    # 0.3 está no teto; 0.01 fica longe do piso (0.005) em log10
    assert experimentos.na_borda({"learning_rate": 0.3, "reg_lambda": 0.5}, FAIXAS_AMPLO) == ["learning_rate"]


def test_samplers_rapido(tmp_path, monkeypatch):
    monkeypatch.setattr(experimentos, "PASTA_EXPERIMENTOS", tmp_path)
    r = experimentos.experimento_samplers(n_tentativas=2, storage=None, rapido=True)
    assert set(r) == set(experimentos.SAMPLERS) | {"tpe_ss1", "tpe_ss2", "ruido_busca_tpe"}
    assert len(r["ruido_busca_tpe"]["erros"]) == 3
    assert r["ruido_busca_tpe"]["desvio"] >= 0
    assert json.loads((tmp_path / "rapido" / "samplers.json").read_text(encoding="utf-8")) == r


def test_cerca_retoma(tmp_path, monkeypatch):
    monkeypatch.setattr(experimentos, "PASTA_EXPERIMENTOS", tmp_path)
    storage = f"sqlite:///{(tmp_path / 'x.db').as_posix()}"
    experimentos.experimento_cerca(n_tentativas=2, storage=storage, rapido=True)
    experimentos.experimento_cerca(n_tentativas=2, storage=storage, rapido=True)
    estudos = optuna.get_all_study_summaries(storage)
    assert estudos and all(s.n_trials == 2 for s in estudos)


def test_sementes_rapido(tmp_path, monkeypatch):
    monkeypatch.setattr(experimentos, "PASTA_EXPERIMENTOS", tmp_path)
    p = CANDIDATOS["xgboost"].PARAMS_RAPIDOS | {"subsample": 0.7}
    r = experimentos.experimento_sementes(p, n=2, rapido=True)
    assert r["desvio"] >= 0 and len(r["por_semente"]) == 2


def test_entradas_rapido(tmp_path, monkeypatch):
    monkeypatch.setattr(experimentos, "PASTA_EXPERIMENTOS", tmp_path)
    r = experimentos.experimento_entradas({}, sementes=(0, 1), rapido=True)
    assert set(r["variantes"]) == {"base", "fracao_combustao"}
    assert all(len(v["por_semente"]) == 2 for v in r["variantes"].values())
    assert "hash_csv" in r
    assert (tmp_path / "rapido" / f"entradas-{r['hash_csv']}.json").exists()
    assert preprocess.DERIVADAS == ["fracao_combustao"]


def test_boosting_rapido(tmp_path, monkeypatch):
    monkeypatch.setattr(experimentos, "PASTA_EXPERIMENTOS", tmp_path)
    r = experimentos.experimento_boosting(n_tentativas=2, storage=None, rapido=True)
    assert set(r) == {"catboost", "lightgbm"}
    assert all(x["erro_mj"] < 0.15 and len(x["params_por_dobra"]) == 2 for x in r.values())
    assert (tmp_path / "rapido" / "boosting.json").exists()


def test_experimento_nao_esta_no_treino_oficial():
    assert not {"xgboost_amplo", "catboost", "lightgbm"} & set(CANDIDATOS)
    assert experimentos.STORAGE_EXPERIMENTOS != treino.STORAGE


def test_redes_rapido(tmp_path, monkeypatch):
    monkeypatch.setattr(experimentos, "PASTA_EXPERIMENTOS", tmp_path)
    p = {
        "camadas": 1,
        "largura": 32,
        "dropout": 0.0,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "tamanho_lote": 32,
        "max_epocas": 3,
    }
    r = experimentos.experimento_redes(p, n=2, lrs=(1e-3,), rapido=True)
    assert {"linha_s0", "linha_s1", "grupo_s0", "grupo_s1", "media5", "lr_0.001"} <= set(r)
    assert (tmp_path / "rapido" / "curvas_perda.png").exists()


def test_melhores_params_sem_banco_nao_cria_nada(tmp_path):
    caminho = tmp_path / "optuna" / "estudos.db"
    with pytest.raises(ValueError, match="database not found"):
        experimentos.melhores_params("xgboost", storage=f"sqlite:///{caminho.as_posix()}")
    assert not caminho.parent.exists()


def test_melhores_params_pasta_sem_banco(tmp_path):
    caminho = tmp_path / "optuna" / "estudos.db"
    caminho.parent.mkdir()
    with pytest.raises(ValueError, match="database not found"):
        experimentos.melhores_params("xgboost", storage=f"sqlite:///{caminho.as_posix()}")
    assert not caminho.exists()
