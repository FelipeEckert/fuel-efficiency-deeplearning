import numpy as np
import optuna

from eficiencia.esquema import carregar_base
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
    buscar_params("linear", df, 2, storage, "t", n_dobras=2)
    assert len(optuna.load_study(study_name="t", storage=storage).trials) == 4
