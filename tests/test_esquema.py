from eficiencia.esquema import (
    ALVOS,
    CATEGORICAS,
    ENTRADAS,
    ROTULOS,
    VALORES,
    carregar_base,
)


def test_entradas_sem_colunas_proibidas():
    proibidas = {"marca", "modelo", "area_m2", "lcvm", "nome_homologado", "grupo_cv", "versoes", "n_versoes"}
    assert not proibidas & set(ENTRADAS)
    assert not [c for c in ENTRADAS if c.startswith(("kml_", "mj_km"))]


def test_base_tem_todas_as_colunas():
    df = carregar_base()
    assert set(ENTRADAS + ALVOS + ["grupo_cv", "mj_km"]) <= set(df.columns)
    for c in CATEGORICAS:
        assert set(df[c].dropna()) <= set(VALORES[c])
    assert set(ROTULOS) >= set(ENTRADAS)
