"""Página Portfólio: MJ/km médio dos veículos ponderado pelo volume, com valores reais ou estimados pelo modelo."""

import re
import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # pasta app/, para importar formulario

import numpy as np
import pandas as pd
import streamlit as st
from formulario import CATEGORIAS, formulario_veiculo, modelo

from eficiencia.esquema import CATEGORICAS, VALORES
from eficiencia.formato import numero
from eficiencia.portfolio import (
    COLUNAS_PORTFOLIO,
    ROTULOS_PORTFOLIO,
    ResultadoPortfolio,
    ler_planilha,
    media_portfolio,
    modelo_planilha,
)

XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
ORIGEM = {"real": "Real", "estimado": "Estimado"}


def _para_editor(df: pd.DataFrame) -> pd.DataFrame:
    """Tipos que o editor aceita: volume e categóricas como texto (o "10.000" digitado não vira número)."""
    df = df.copy()
    texto = [v if isinstance(v, str) else "" if pd.isna(v) else numero(v) for v in df["volume"]]
    df["volume"] = pd.Series(texto, index=df.index, dtype="string")  # tipo texto mesmo sem linhas
    for coluna in CATEGORICAS:
        df[coluna] = df[coluna].astype("string")
    return df


def _atualizar(df: pd.DataFrame) -> None:
    """Troca a tabela e recria o editor, descartando as edições pendentes dele."""
    st.session_state["portfolio"] = df
    st.session_state["portfolio_atual"] = df
    st.session_state["versao_editor"] += 1


def _ao_enviar() -> None:
    arquivo = st.session_state["upload_portfolio"]
    if arquivo is None:  # o "x" do uploader também dispara a mudança
        return
    df, erros = ler_planilha(arquivo.getvalue())
    if erros:
        st.session_state["erros_upload"] = erros
    else:
        _atualizar(_para_editor(df))


def _adicionar(nome: str, volume: float, veiculo: dict) -> None:
    linha = {"nome": nome, "volume": numero(volume), "mj_km_conhecido": np.nan, **veiculo}
    atual = st.session_state["portfolio_atual"]
    _atualizar(pd.DataFrame([*atual.to_dict("records"), linha], columns=COLUNAS_PORTFOLIO))


def _colunas() -> dict:
    colunas = {c: st.column_config.Column(ROTULOS_PORTFOLIO[c]) for c in COLUNAS_PORTFOLIO}
    codigos = "Códigos: " + "; ".join(f"{k} = {v}" for k, v in CATEGORIAS.items())
    for coluna in CATEGORICAS:
        ajuda = codigos if coluna == "categoria" else None
        colunas[coluna] = st.column_config.SelectboxColumn(
            ROTULOS_PORTFOLIO[coluna], options=VALORES[coluna], help=ajuda
        )
    colunas["volume"] = st.column_config.TextColumn(ROTULOS_PORTFOLIO["volume"])
    colunas["mj_km_conhecido"] = st.column_config.NumberColumn(
        ROTULOS_PORTFOLIO["mj_km_conhecido"], min_value=0.0, format="%.2f"
    )
    return colunas


def _calcular(df: pd.DataFrame, pacote) -> ResultadoPortfolio:
    """Sem modelo, só as linhas com MJ/km conhecido são calculadas; as outras viram erro, com o número da linha."""
    if pacote is not None:
        return media_portfolio(df, pacote)
    conhecida = df["mj_km_conhecido"].notna().to_numpy()
    numeros = np.flatnonzero(conhecida) + 1
    resultado = media_portfolio(df[conhecida].reset_index(drop=True))
    resultado.erros = [
        re.sub(r"^linha (\d+):", lambda m: f"linha {numeros[int(m[1]) - 1]}:", erro)
        for erro in resultado.erros
    ]
    resultado.erros += [
        f"linha {n}: sem MJ/km conhecido e sem modelo treinado para estimar"
        for n in np.flatnonzero(~conhecida) + 1
    ]
    return resultado


def _quebra(df: pd.DataFrame, rotulo: str, nome=lambda v: v) -> pd.DataFrame:
    return pd.DataFrame(
        {
            rotulo: [nome(v) for v in df.iloc[:, 0]],
            "Volume": [numero(v) for v in df["volume"]],
            "MJ/km médio": [numero(v, 2) for v in df["media_mj_km"]],
        }
    )


def _veiculos(tabela: pd.DataFrame) -> pd.DataFrame:
    faixa = [
        "" if origem == "real" else f"{numero(a, 2)} – {numero(b, 2)}"
        for origem, a, b in zip(tabela.origem, tabela.mj_km_min, tabela.mj_km_max)
    ]
    return pd.DataFrame(
        {
            "Nome": tabela.nome,
            "Volume": [numero(v) for v in tabela.volume],
            "Origem": tabela.origem.map(ORIGEM),
            "MJ/km": [numero(v, 2) for v in tabela.mj_km],
            "Faixa (MJ/km)": faixa,
            "Propulsão": tabela.propulsao,
            "Categoria": tabela.categoria.map(lambda c: CATEGORIAS.get(c, c)),
        }
    )


def _excel(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    df[COLUNAS_PORTFOLIO].rename(columns=ROTULOS_PORTFOLIO).to_excel(buffer, index=False)
    return buffer.getvalue()


st.set_page_config(page_title="Portfólio", layout="wide")
st.title("Portfólio")
st.caption("MJ/km médio dos veículos, ponderado pelo volume de vendas de cada um.")
st.session_state.setdefault("portfolio", pd.DataFrame(columns=COLUNAS_PORTFOLIO))
st.session_state.setdefault("portfolio_atual", st.session_state["portfolio"])  # última tabela editada
st.session_state.setdefault("versao_editor", 0)
chave_editor = f"editor_{st.session_state['versao_editor']}"
if (
    chave_editor not in st.session_state
):  # o editor não estava na tela (outra página ou primeira vez): retoma a edição
    st.session_state["portfolio"] = st.session_state["portfolio_atual"]

try:
    pacote = modelo()
except FileNotFoundError:
    pacote = None
    st.info("Sem modelo treinado: só linhas com MJ/km conhecido serão calculadas.")

topo = st.container()  # resultado aparece acima da tabela, mas é calculado com a tabela editada

st.subheader("Veículos")
editado = st.data_editor(
    _para_editor(st.session_state["portfolio"]),
    key=chave_editor,
    num_rows="dynamic",
    hide_index=True,
    column_config=_colunas(),
)
st.caption(
    "Volume aceita 10.000 (milhar) e 1.000,5. Preencha MJ/km conhecido para veículos já homologados; "
    "deixe vazio para o modelo estimar."
)
st.session_state["portfolio_atual"] = editado

c1, c2, c3 = st.columns(3)
c1.download_button("Baixar planilha modelo", modelo_planilha(), "planilha_modelo_portfolio.xlsx", mime=XLSX)
c2.file_uploader("Substituir por planilha (xlsx)", type="xlsx", key="upload_portfolio", on_change=_ao_enviar)
for erro in st.session_state.pop("erros_upload", []):
    c2.error(erro)
c3.download_button("Baixar portfólio", _excel(editado), "portfolio.xlsx", mime=XLSX)

if pacote is not None:
    with st.expander("Adicionar veículo pelo formulário"):
        c1, c2 = st.columns(2)
        nome = c1.text_input("Nome do veículo", key="portfolio_nome")
        volume = c2.number_input("Volume", min_value=0.0, step=100.0, key="portfolio_volume")
        veiculo = formulario_veiculo("portfolio_form")
        st.button("Adicionar", key="adicionar_veiculo", on_click=_adicionar, args=(nome, volume, veiculo))

with topo:
    if editado.empty:
        st.info("Adicione veículos na tabela, pelo formulário ou por planilha.")
    else:
        resultado = _calcular(editado, pacote)
        for erro in resultado.erros:
            st.error(erro)
        for linha in resultado.tabela.itertuples():
            for aviso in linha.avisos:
                st.warning(f"{linha.nome}: {aviso}")
        media = resultado.media_mj_km
        st.metric("MJ/km médio do portfólio", "—" if pd.isna(media) else f"{numero(media, 2)} MJ/km")
        if not resultado.tabela.empty:
            c1, c2 = st.columns(2)
            c1.subheader("Por propulsão")
            c1.dataframe(_quebra(resultado.por_propulsao, "Propulsão"), hide_index=True)
            c2.subheader("Por categoria")
            c2.dataframe(
                _quebra(resultado.por_categoria, "Categoria", lambda c: CATEGORIAS.get(c, c)), hide_index=True
            )
            st.subheader("Por veículo")
            st.dataframe(_veiculos(resultado.tabela), hide_index=True)
