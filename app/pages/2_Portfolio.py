"""Portfolio page: sales-weighted average MJ/km, using label values where known and model estimates otherwise."""

import re
import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # pasta app/, para importar formulario

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from formulario import CATEGORIAS, formulario_veiculo, modelo
from visual import CORES

from eficiencia.esquema import CATEGORICAS, NOMES_VALORES, VALORES, nome_valor
from eficiencia.formato import numero
from eficiencia.portfolio import (
    COLUNAS_PORTFOLIO,
    ROTULOS_PORTFOLIO,
    ResultadoPortfolio,
    ler_planilha,
    media_portfolio,
    modelo_planilha,
    totais_portfolio,
)

XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
ORIGEM = {"real": "Label", "estimado": "Estimated"}


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
    codigos = {"categoria": CATEGORIAS} | NOMES_VALORES
    for coluna in CATEGORICAS:
        nomes = codigos.get(coluna)
        ajuda = "Codes: " + "; ".join(f"{k} = {v}" for k, v in nomes.items()) if nomes else None
        colunas[coluna] = st.column_config.SelectboxColumn(
            ROTULOS_PORTFOLIO[coluna], options=VALORES[coluna], help=ajuda
        )
    colunas["volume"] = st.column_config.TextColumn(ROTULOS_PORTFOLIO["volume"])
    colunas["mj_km_conhecido"] = st.column_config.NumberColumn(
        ROTULOS_PORTFOLIO["mj_km_conhecido"], min_value=0.0, format="%.2f"
    )
    return colunas


def _com_nomes(df: pd.DataFrame) -> pd.DataFrame:
    """Veículo sem nome aparece como "linha N" (posição na tabela), e não como "nan"."""
    posicoes = pd.Series([f"row {i}" for i in range(1, len(df) + 1)], index=df.index)
    sem_nome = df["nome"].isna() | df["nome"].astype(str).str.strip().eq("")
    return df.assign(nome=df["nome"].where(~sem_nome, posicoes))


def _calcular(df: pd.DataFrame, pacote) -> ResultadoPortfolio:
    """Sem modelo, só as linhas com MJ/km conhecido são calculadas; as outras viram erro, com o número da linha."""
    df = _com_nomes(df)
    if pacote is not None:
        return media_portfolio(df, pacote)
    conhecida = df["mj_km_conhecido"].notna().to_numpy()
    numeros = np.flatnonzero(conhecida) + 1
    resultado = media_portfolio(df[conhecida].reset_index(drop=True))
    resultado.erros = [
        re.sub(r"^row (\d+):", lambda m: f"row {numeros[int(m[1]) - 1]}:", erro) for erro in resultado.erros
    ]
    resultado.erros += [
        f"row {n}: no known MJ/km and no trained model to estimate it" for n in np.flatnonzero(~conhecida) + 1
    ]
    resultado.n_total, resultado.volume_total = totais_portfolio(df)  # as linhas sem MJ/km também contam
    return resultado


def _quebra(df: pd.DataFrame, rotulo: str, nome=lambda v: v) -> pd.DataFrame:
    return pd.DataFrame(
        {
            rotulo: [nome(v) for v in df.iloc[:, 0]],
            "Volume": [numero(v) for v in df["volume"]],
            "Average MJ/km": [numero(v, 2) for v in df["media_mj_km"]],
        }
    )


def _veiculos(tabela: pd.DataFrame) -> pd.DataFrame:
    faixa = [
        "" if origem == "real" else f"{numero(a, 2)} – {numero(b, 2)}"
        for origem, a, b in zip(tabela.origem, tabela.mj_km_min, tabela.mj_km_max)
    ]
    return pd.DataFrame(
        {
            "Name": tabela.nome,
            "Volume": [numero(v) for v in tabela.volume],
            "Source": tabela.origem.map(ORIGEM),
            "MJ/km": [numero(v, 2) for v in tabela.mj_km],
            "Range (MJ/km)": faixa,
            "Powertrain": tabela.propulsao.map(lambda p: nome_valor("propulsao", p)),
            "Category": tabela.categoria.map(lambda c: CATEGORIAS.get(c, c)),
        }
    )


def _excel(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    df[COLUNAS_PORTFOLIO].rename(columns=ROTULOS_PORTFOLIO).to_excel(buffer, index=False)
    return buffer.getvalue()


st.set_page_config(page_title="Portfolio", layout="wide")
st.title("Portfolio")
st.caption("Average MJ/km of a set of vehicles, weighted by the sales volume of each one.")
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
    st.info("No trained model: only rows with a known MJ/km will be included.")

topo = st.container(border=True)  # resultado aparece acima da tabela, mas é calculado com a tabela editada

st.subheader("Vehicles")
editado = st.data_editor(
    _para_editor(st.session_state["portfolio"]),
    key=chave_editor,
    num_rows="dynamic",
    hide_index=True,
    column_config=_colunas(),
)
st.caption(
    "Volume accepts thousands separators (10,000). Fill in Known MJ/km for vehicles that already have a "
    "label; leave it empty for the model to estimate."
)
st.session_state["portfolio_atual"] = editado

c1, c2, c3 = st.columns(3)
c1.download_button("Download template", modelo_planilha(), "portfolio_template.xlsx", mime=XLSX)
c2.file_uploader(
    "Replace with a spreadsheet (xlsx)", type="xlsx", key="upload_portfolio", on_change=_ao_enviar
)
for erro in st.session_state.pop("erros_upload", []):
    c2.error(erro)
c3.download_button("Download portfolio", _excel(editado), "portfolio.xlsx", mime=XLSX)

if pacote is not None:
    with st.expander("Add a vehicle with the form"):
        c1, c2 = st.columns(2)
        nome = c1.text_input("Vehicle name", key="portfolio_nome")
        volume = c2.number_input("Volume", min_value=0.0, step=100.0, key="portfolio_volume")
        veiculo = formulario_veiculo("portfolio_form")
        st.button("Add", key="adicionar_veiculo", on_click=_adicionar, args=(nome, volume, veiculo))

with topo:
    if editado.empty:
        st.info("Add vehicles in the table, with the form or from a spreadsheet.")
    else:
        resultado = _calcular(editado, pacote)
        for erro in resultado.erros:
            st.error(erro)
        for linha in resultado.tabela.itertuples():
            for aviso in linha.avisos:
                st.warning(f"{linha.nome}: {aviso}")
        media = resultado.media_mj_km
        st.metric("Portfolio average MJ/km", "—" if pd.isna(media) else f"{numero(media, 2)} MJ/km")
        pct = 100 * resultado.volume_incluido / resultado.volume_total if resultado.volume_total else 0
        st.caption(
            f"{resultado.n_incluidos} of {resultado.n_total} vehicles included ({numero(pct, 1)} % of the volume)"
        )
        if not resultado.tabela.empty:
            c1, c2 = st.columns(2)
            c1.subheader("By powertrain")
            por_propulsao = _quebra(
                resultado.por_propulsao, "Powertrain", lambda p: nome_valor("propulsao", p)
            )
            c1.dataframe(por_propulsao, hide_index=True)
            fig = go.Figure(
                go.Bar(
                    x=[nome_valor("propulsao", p) for p in resultado.por_propulsao.iloc[:, 0]],
                    y=list(resultado.por_propulsao["media_mj_km"]),
                    marker_color=CORES["primaria"],
                    texttemplate="%{y:.2f}",
                    hovertemplate="%{x}: %{y:.2f} MJ/km<extra></extra>",
                )
            )
            fig.update_layout(yaxis_title="Average MJ/km", height=260, margin={"t": 10})
            c1.plotly_chart(fig)
            c2.subheader("By category")
            c2.dataframe(
                _quebra(resultado.por_categoria, "Category", lambda c: CATEGORIAS.get(c, c)), hide_index=True
            )
            st.subheader("By vehicle")
            st.dataframe(_veiculos(resultado.tabela), hide_index=True)
