"""Página Veículo: estimativa de um veículo, "e se" sobre ele e cenários salvos para comparação."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # pasta app/, para importar formulario

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from formulario import excel_cenarios, formulario_veiculo, modelo, nome_livre, nome_padrao

from eficiencia.fisica import classe_inercia
from eficiencia.formato import numero
from eficiencia.prever import prever

NOMES_KML = {
    "gasolina": "gasolina",
    "etanol": "etanol",
    "eletrico_eq": "elétrico equivalente",
    "diesel": "diesel",
}


def _sinal(valor: float, casas: int) -> str:
    """Sinal ASCII: o Streamlit decide a seta do delta pelo "-" inicial."""
    sinal = "+" if valor > 0 else "-" if valor < 0 else ""
    return sinal + numero(abs(valor), casas)


def _prever(pacote, veiculo: dict) -> pd.Series | None:
    """Previsão de uma linha; com entrada inválida, mostra cada mensagem e devolve None."""
    try:
        return prever(veiculo, pacote).iloc[0]
    except ValueError as erro:
        for mensagem in str(erro).split("\n"):
            st.error(mensagem)
        return None


def _valor(r: pd.Series, saida: str, casas: int, unidade: str) -> str:
    return "—" if pd.isna(r[saida]) else f"{numero(r[saida], casas)} {unidade}"


def _faixa(r: pd.Series, saida: str, casas: int, unidade: str) -> str:
    """Texto no formato "1,62 MJ/km (1,54 – 1,70)"; "—" quando a saída não se aplica."""
    texto = _valor(r, saida, casas, unidade)
    if pd.isna(r[f"{saida}_min"]) or pd.isna(r[f"{saida}_max"]):
        return texto
    return f"{texto} ({numero(r[f'{saida}_min'], casas)} – {numero(r[f'{saida}_max'], casas)})"


def _metrica(coluna, rotulo: str, r: pd.Series, saida: str, casas: int, unidade: str) -> None:
    coluna.metric(rotulo, _valor(r, saida, casas, unidade))  # sem delta: nenhuma seta sobre a faixa
    coluna.caption(_faixa(r, saida, casas, unidade))


def _resultado(r: pd.Series) -> None:
    st.subheader("Resultado")
    c1, c2, c3 = st.columns(3)
    _metrica(c1, "MJ/km combinado", r, "mj_km", 2, "MJ/km")
    _metrica(c2, "MJ/km urbano", r, "mj_km_urbano", 2, "MJ/km")
    _metrica(c3, "MJ/km estrada", r, "mj_km_estrada", 2, "MJ/km")

    aplicaveis = [c for c in NOMES_KML if pd.notna(r[f"kml_{c}"])]
    if aplicaveis:
        st.caption("Consumo em km/l (combinado urbano e estrada) para os combustíveis deste veículo")
        for coluna, comb in zip(st.columns(len(aplicaveis)), aplicaveis):
            _metrica(coluna, f"km/l {NOMES_KML[comb]}", r, f"kml_{comb}", 1, "km/l")
    for aviso in r["avisos"]:
        st.warning(aviso)


def _e_se(pacote, veiculo: dict, original: pd.Series) -> tuple[dict, pd.Series]:
    """Controles de variação; devolve o veículo ajustado e sua previsão (o original, se o ajuste falhar)."""
    st.subheader("E se…")
    st.caption("Mexa nos controles para ver quanto o consumo muda em relação ao veículo acima.")
    c1, c2 = st.columns(2)
    variacao = {
        "massa_om_kg": c1.slider("Variação da massa (%)", -20, 20, 0, 1, format="%d %%", key="es_massa"),
        "f0_n": c2.slider("Variação da F0 (%)", -20, 20, 0, 1, format="%d %%", key="es_f0"),
        "f2_n_kmh2": c1.slider("Variação da F2 (%)", -20, 20, 0, 1, format="%d %%", key="es_f2"),
        "potencia_kw": c2.slider(
            "Variação da potência (%)", -20, 20, 0, 1, format="%d %%", key="es_potencia"
        ),
    }
    alterado = dict(veiculo)
    for campo, pct in variacao.items():
        alterado[campo] = veiculo[campo] * (1 + pct / 100)
    if variacao["massa_om_kg"]:
        alterado["classe_inercia_kg"] = classe_inercia(alterado["massa_om_kg"])

    novo = _prever(pacote, alterado)
    if novo is None:
        return veiculo, original
    diferenca = novo.mj_km - original.mj_km
    st.metric(
        "MJ/km com as mudanças",
        f"{numero(novo.mj_km, 2)} MJ/km",
        f"{_sinal(diferenca, 2)} MJ/km ({_sinal(diferenca / original.mj_km * 100, 1)} %) em relação ao original",
        delta_color="inverse",
    )
    for aviso in novo["avisos"]:
        if aviso not in original["avisos"]:
            st.warning(aviso)
    return alterado, novo


def _salvar(nome_chave: str, cenario: dict) -> None:
    cenarios = st.session_state["cenarios"]
    nomes = [c["nome"] for c in cenarios]
    digitado = st.session_state[nome_chave].strip()
    cenarios.append({"nome": nome_livre(digitado or nome_padrao(nomes), nomes), **cenario})
    st.session_state[nome_chave] = nome_padrao([c["nome"] for c in cenarios])


def _remover(indice: int) -> None:
    st.session_state["cenarios"].pop(indice)


def _grafico(cenarios: list[dict]) -> None:
    base = cenarios[0]["resultado"]["mj_km"]
    valores = [c["resultado"]["mj_km"] for c in cenarios]
    rotulos = []
    for i, c in enumerate(cenarios):
        if i == 0:
            rotulos.append(f"{c['nome']}<br>base")
        else:
            diferenca = valores[i] - base
            rotulos.append(
                f"{c['nome']}<br>{_sinal(diferenca, 2)} MJ/km ({_sinal(diferenca / base * 100, 1)} %)"
            )
    fig = go.Figure(
        go.Bar(
            x=rotulos,
            y=valores,
            marker_color=["#7f8c8d"] + ["#2e86c1"] * (len(cenarios) - 1),
            error_y={
                "type": "data",
                "array": [c["resultado"]["mj_km_max"] - v for c, v in zip(cenarios, valores)],
                "arrayminus": [v - c["resultado"]["mj_km_min"] for c, v in zip(cenarios, valores)],
                "visible": True,
            },
        )
    )
    fig.update_layout(yaxis_title="MJ/km combinado", height=420, margin={"t": 30}, separators=",.")
    st.plotly_chart(fig)


def _cenarios() -> None:
    st.subheader("Cenários")
    st.caption("O primeiro cenário salvo é a referência (base) da comparação.")
    if "nome_cenario" not in st.session_state:
        st.session_state["nome_cenario"] = nome_padrao([c["nome"] for c in st.session_state["cenarios"]])
    st.text_input("Nome do cenário", key="nome_cenario")

    cenarios = st.session_state["cenarios"]
    if not cenarios:
        return
    _grafico(cenarios)
    for i, c in enumerate(cenarios):
        col_nome, col_mj, col_botao = st.columns([3, 2, 1])
        col_nome.write(f"**{c['nome']}**" + (" (base)" if i == 0 else ""))
        col_mj.write(f"{numero(c['resultado']['mj_km'], 2)} MJ/km")
        col_botao.button("Remover", key=f"remover_{i}", on_click=_remover, args=(i,))
    st.download_button(
        "Exportar cenários (Excel)",
        data=excel_cenarios(cenarios),
        file_name="cenarios_veiculo.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


st.set_page_config(page_title="Veículo", layout="wide")
st.title("Veículo")
try:
    pacote = modelo()
except FileNotFoundError:
    st.error("Nenhum modelo treinado. Rode: uv run eficiencia-treinar")
    st.stop()

if "cenarios" not in st.session_state:
    st.session_state["cenarios"] = []

veiculo = formulario_veiculo("veiculo")
original = _prever(pacote, veiculo)
if original is None:
    st.stop()
_resultado(original)

alterado, novo = _e_se(pacote, veiculo, original)
st.button(
    "Salvar cenário",
    key="salvar_cenario",
    on_click=_salvar,
    args=("nome_cenario", {"veiculo": alterado, "resultado": novo.to_dict()}),
)
st.caption("Salva o veículo com os ajustes atuais (sliders em 0 = veículo do formulário)")
_cenarios()
