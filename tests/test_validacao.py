from typing import ClassVar

import numpy as np
import optuna
import pandas as pd
import pytest

from eficiencia.esquema import ALVOS, carregar_base
from eficiencia.modelos import CANDIDATOS
from eficiencia.modelos.base import Candidato
from eficiencia.validacao import buscar_params, cv_aninhada, dobras_por_grupo


def test_dobras_sem_vazamento():
    df = carregar_base()
    dobras = dobras_por_grupo(df["grupo_cv"], 5)
    testes = np.concatenate([te for _, te in dobras])
    assert sorted(testes) == list(range(len(df)))
    for tr, te in dobras:
        assert not set(df["grupo_cv"].iloc[tr]) & set(df["grupo_cv"].iloc[te])


def test_cv_aninhada_linear():
    r = cv_aninhada("linear", carregar_base(), n_tentativas=3, storage=None, n_externas=2, n_internas=2)
    assert r.oof.notna()["mj_km_urbano"].all()
    assert r.erro_mj < 0.10  # referência medida: 7,7 %
    assert set(r.erro_por_propulsao) == {"Combustão", "Híbrido", "Plug-in", "Elétrico"}


def test_estudo_retomavel(tmp_path):
    storage = f"sqlite:///{tmp_path / 'e.db'}"
    df = carregar_base()
    buscar_params("linear", df, 2, storage, "t", n_dobras=2)
    buscar_params("linear", df, 2, storage, "t", n_dobras=2)  # estudo já completo: não roda nada
    assert len(optuna.load_study(study_name="t", storage=storage).trials) == 2
    buscar_params("linear", df, 4, storage, "t", n_dobras=2)  # completa só as que faltam
    trials = optuna.load_study(study_name="t", storage=storage).trials
    assert len(trials) == 4
    # a retomada não pode repetir as amostras da primeira leva
    assert [t.params for t in trials[:2]] != [t.params for t in trials[2:]]


def test_nomes_de_estudo_separados_por_configuracao(tmp_path):
    storage = f"sqlite:///{tmp_path / 'e.db'}"
    df = carregar_base()
    cv_aninhada("linear", df, n_tentativas=1, storage=storage, n_externas=2, n_internas=2)
    primeiros = set(optuna.get_all_study_names(storage))
    cv_aninhada("linear", df, n_tentativas=1, storage=storage, n_externas=3, n_internas=2)
    todos = set(optuna.get_all_study_names(storage))
    assert len(primeiros) == 2
    assert len(todos) == len(primeiros) + 3


def test_busca_rapida_treina_com_params_rapidos(monkeypatch):
    registrados: list[dict] = []

    class Falso(Candidato):
        nome = "falso"
        PARAMS_RAPIDOS: ClassVar[dict] = {"max_epocas": 5}

        @staticmethod
        def sugerir(trial):
            return {"x": trial.suggest_float("x", 0.0, 1.0)}

        def fit(self, X, Y):
            registrados.append(dict(self.params))
            return self

        def predict(self, X):
            return pd.DataFrame(1.0, index=X.index, columns=ALVOS)

    monkeypatch.setitem(CANDIDATOS, "falso", Falso)
    buscar_params("falso", carregar_base(), 2, None, "falso-rapido", n_dobras=2, rapido=True)
    assert registrados and all(p["max_epocas"] == 5 for p in registrados)


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.parametrize("sampler", ["tpe", "random", "cmaes", "qmc", "gp"])
def test_cv_com_cada_sampler(sampler):
    r = cv_aninhada(
        "xgboost",
        carregar_base(),
        n_tentativas=3,
        storage=None,
        n_externas=2,
        n_internas=2,
        rapido=True,
        sampler=sampler,
    )
    assert np.isfinite(r.erro_mj)


def test_nome_do_estudo_leva_o_sampler(tmp_path):
    storage = f"sqlite:///{tmp_path / 'e.db'}"
    df = carregar_base()
    cv_aninhada("linear", df, n_tentativas=1, storage=storage, n_externas=2, n_internas=2, sampler="random")
    cv_aninhada("linear", df, n_tentativas=1, storage=storage, n_externas=2, n_internas=2)
    nomes = optuna.get_all_study_names(storage)
    assert sum("-random-" in n for n in nomes) == 2
    assert sum("-random-" not in n for n in nomes) == 2  # o TPE mantém o nome antigo


def test_sampler_desconhecido():
    with pytest.raises(ValueError, match="sampler desconhecido"):
        cv_aninhada("linear", carregar_base(), 1, None, n_externas=2, n_internas=2, sampler="sa")
