import io

import numpy as np
import pandas as pd
import pytest

from eficiencia.esquema import ENTRADAS, carregar_base
from eficiencia.portfolio import COLUNAS_PORTFOLIO, ler_planilha, media_portfolio, modelo_planilha


def test_media_ponderada_valores_reais():
    df = pd.DataFrame({"nome": ["A", "B"], "volume": [100, 300], "mj_km_conhecido": [1.5, 2.0]})
    r = media_portfolio(df)
    assert r.media_mj_km == pytest.approx(1.875) and list(r.tabela["origem"]) == ["real", "real"]


def test_mistura_real_e_estimado(pacote_rapido):
    v = carregar_base()[ENTRADAS].iloc[0].drop("bateria_estimada").to_dict()
    df = pd.DataFrame(
        [{"nome": "novo", "volume": 100, **v}, {"nome": "atual", "volume": 100, "mj_km_conhecido": 1.0}]
    )
    r = media_portfolio(df, pacote_rapido)
    est = r.tabela.loc[r.tabela.origem == "estimado", "mj_km"].iloc[0]
    assert r.media_mj_km == pytest.approx((est + 1.0) / 2)


def test_volumes_invalidos():
    df = pd.DataFrame({"nome": ["A", "B", "C"], "volume": [0, -5, None], "mj_km_conhecido": [1.5, 2.0, 1.0]})
    r = media_portfolio(df)
    linhas = [e for e in r.erros if e.startswith("linha ")]
    assert np.isnan(r.media_mj_km) and len(linhas) == 3 and linhas[0].startswith("linha 1:")
    assert any(not e.startswith("linha ") for e in r.erros)  # erro geral: nenhum volume válido


def test_modelo_planilha_ida_e_volta():
    df, erros = ler_planilha(modelo_planilha())
    assert list(df.columns) == COLUNAS_PORTFOLIO and df.empty and erros == []


def test_ler_planilha_colunas_faltando():
    buf = io.BytesIO()
    pd.DataFrame({"Nome": ["x"]}).to_excel(buf, index=False)
    _, erros = ler_planilha(buf.getvalue())
    assert erros and "Volume" in erros[0]


def test_numeracao_de_linha_e_mj_textual():
    df = pd.DataFrame(
        {
            "nome": ["ok", "sem especificação", "volume ruim", "mj zero"],
            "volume": [100, 50, 0, 10],
            "mj_km_conhecido": ["1,5", None, 2.0, 0],
        }
    )
    r = media_portfolio(df)  # nenhuma linha a estimar: o modelo não é carregado
    assert r.media_mj_km == pytest.approx(1.5)
    assert any(e.startswith("linha 2: ") for e in r.erros)  # especificação faltando, na posição da tabela
    assert "linha 3: volume deve ser maior que zero" in r.erros
    assert "linha 4: MJ/km conhecido deve ser número maior que zero" in r.erros


def test_quebra_usa_nao_informado():
    df = pd.DataFrame(
        {
            "nome": ["A", "B"],
            "volume": [100, 300],
            "mj_km_conhecido": [1.0, 2.0],
            "propulsao": ["Combustão", None],
        }
    )
    r = media_portfolio(df)
    assert set(r.por_propulsao["propulsao"]) == {"Combustão", "não informado"}
    assert r.por_propulsao.set_index("propulsao").loc["não informado", "media_mj_km"] == pytest.approx(2.0)


def test_ler_planilha_aceita_nomes_de_coluna():
    buf = io.BytesIO()
    pd.DataFrame({"nome": ["x"], "volume": [10], "mj_km_conhecido": [1.2], "extra": [0]}).to_excel(
        buf, index=False
    )
    df, erros = ler_planilha(buf.getvalue())
    assert erros == [] and list(df.columns) == COLUNAS_PORTFOLIO and df.loc[0, "volume"] == 10
