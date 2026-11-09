import pandas as pd
import xgboost

from eficiencia.arvore import caminho_arvore
from eficiencia.entrada import validar
from eficiencia.esquema import ALVOS, ENTRADAS, carregar_base
from eficiencia.modelos import CANDIDATOS


def _modelo():
    df = carregar_base()
    m = CANDIDATOS["xgboost"](**CANDIDATOS["xgboost"].PARAMS_RAPIDOS).fit(df[ENTRADAS], df[ALVOS])
    return m, df


def _veiculo(df, propulsao):
    linha = df[df.propulsao == propulsao][ENTRADAS].iloc[0]
    return validar(pd.DataFrame([linha.to_dict()]))[0].iloc[0].to_dict()


def test_caminho_termina_na_folha_que_o_xgboost_usa():
    m, df = _modelo()
    for propulsao in ("Combustão", "Elétrico"):
        v = _veiculo(df, propulsao)
        nos = caminho_arvore(m, v, "mj_km_urbano")
        caminho = [n for n in nos if n["no_caminho"]]
        assert caminho[0]["nivel"] == 0 and caminho[-1]["folha"]
        Xt = m.prep.transform(pd.DataFrame([v]))
        folha = m.estimadores["mj_km_urbano"].get_booster().predict(xgboost.DMatrix(Xt), pred_leaf=True)[0][0]
        assert caminho[-1]["id"] == int(folha)


def test_niveis_de_cima_completos_e_perguntas_legiveis():
    m, df = _modelo()
    nos = caminho_arvore(m, _veiculo(df, "Elétrico"), "mj_km_urbano")
    assert sum(n["nivel"] == 0 for n in nos) == 1
    for n in nos:
        if not n["folha"]:
            assert n["coluna"] in ENTRADAS and "=" not in n["coluna"]
            assert (n["categoria"] is None) != (n["limite"] is None)


def test_caminho_bate_com_pred_leaf_em_amostra_da_base():
    m, df = _modelo()
    amostra = pd.concat([g.sample(min(len(g), 40), random_state=0) for _, g in df.groupby("propulsao")])
    validos, _ = validar(amostra[ENTRADAS])  # linhas da base que o app aceitaria
    amostra = amostra.loc[validos.index]
    assert len(amostra) >= 100 and amostra.propulsao.nunique() == 4
    for alvo in ("mj_km_urbano", "mj_km_estrada"):
        booster = m.estimadores[alvo].get_booster()
        for _, linha in amostra.iterrows():
            v = validos.loc[linha.name].to_dict()
            nos = caminho_arvore(m, v, alvo)
            folha = booster.predict(xgboost.DMatrix(m.prep.transform(pd.DataFrame([v]))), pred_leaf=True)[0][
                0
            ]
            assert [n for n in nos if n["no_caminho"]][-1]["id"] == int(folha), alvo
