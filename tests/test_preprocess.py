import numpy as np

from eficiencia.esquema import ENTRADAS, carregar_base
from eficiencia.preprocess import Preprocessador


def test_preprocessador_categoria_nova_e_marchas_vazias():
    df = carregar_base()
    p = Preprocessador().fit(df[ENTRADAS])
    novo = df[ENTRADAS].iloc[[0]].copy()
    novo["categoria"] = "INEXISTENTE"
    novo["n_marchas"] = np.nan
    x = p.transform(novo)
    assert np.isfinite(x).all()
    assert all(x[0, i] == 0 for i, n in enumerate(p.nomes) if n.startswith("categoria="))
