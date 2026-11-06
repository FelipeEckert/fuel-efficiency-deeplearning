import json

import optuna

from eficiencia import experimentos, treino
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


def test_experimento_nao_esta_no_treino_oficial():
    assert "xgboost_amplo" not in CANDIDATOS
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
