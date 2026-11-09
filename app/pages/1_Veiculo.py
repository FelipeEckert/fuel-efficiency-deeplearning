"""Vehicle page: estimate one vehicle, try "what if" changes on it and save scenarios to compare."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # pasta app/, para importar formulario

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from formulario import excel_cenarios, formulario_veiculo, modelo, nome_livre, nome_padrao
from visual import CORES, NOMES, html_arvore, html_rede

from eficiencia.arvore import caminho_arvore
from eficiencia.entrada import validar
from eficiencia.esquema import ROTULOS, nome_valor
from eficiencia.explicar import explicar, explicavel
from eficiencia.fisica import classe_inercia
from eficiencia.formato import numero
from eficiencia.prever import prever

NOMES_KML = {
    "gasolina": "gasoline",
    "etanol": "ethanol",
    "eletrico_eq": "electric equivalent",
    "diesel": "diesel",
}
CICLOS = {"urbano": "city", "estrada": "highway"}


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


def _fixo(valor: float, casas: int) -> str:
    """Fixed decimals: 1.70, not 1.7."""
    return f"{valor:.{casas}f}"


def _valor(r: pd.Series, saida: str, casas: int, unidade: str) -> str:
    return "—" if pd.isna(r[saida]) else f"{_fixo(r[saida], casas)} {unidade}"


def _faixa(r: pd.Series, saida: str, casas: int) -> str:
    """The 90 % interval as "90 % range: 1.54 – 1.70"; empty when the output has no range."""
    if pd.isna(r[f"{saida}_min"]) or pd.isna(r[f"{saida}_max"]):
        return ""
    return f"90 % range: {_fixo(r[f'{saida}_min'], casas)} – {_fixo(r[f'{saida}_max'], casas)}"


def _metrica(coluna, rotulo: str, r: pd.Series, saida: str, casas: int, unidade: str) -> None:
    coluna.metric(rotulo, _valor(r, saida, casas, unidade))  # sem delta: nenhuma seta sobre a faixa
    coluna.caption(_faixa(r, saida, casas))


def _resultado(r: pd.Series) -> None:
    st.subheader("Result")
    c1, c2, c3 = st.columns(3)
    _metrica(c1, "MJ/km combined", r, "mj_km", 2, "MJ/km")
    _metrica(c2, "MJ/km city", r, "mj_km_urbano", 2, "MJ/km")
    _metrica(c3, "MJ/km highway", r, "mj_km_estrada", 2, "MJ/km")

    aplicaveis = [c for c in NOMES_KML if pd.notna(r[f"kml_{c}"])]
    if aplicaveis:
        st.caption("Consumption in km/l (combined city and highway) for the fuels this vehicle uses")
        for coluna, comb in zip(st.columns(len(aplicaveis)), aplicaveis):
            _metrica(coluna, f"km/l {NOMES_KML[comb]}", r, f"kml_{comb}", 1, "km/l")
    for aviso in r["avisos"]:
        st.warning(aviso)


def _explicacao(pacote, veiculo: dict) -> tuple[str, dict] | None:
    """How much each input raises or lowers consumption against the average vehicle in the data (SHAP).

    Returns (cycle, validated vehicle) so the tree uses the same values; None if the model is not XGBoost."""
    st.subheader("Why this value?")
    if not explicavel(pacote.modelo):
        st.caption("The per-input explanation is only available for the XGBoost model.")
        return None
    ciclo = st.radio("Cycle", list(CICLOS), format_func=CICLOS.get, horizontal=True, key="shap_ciclo")
    normalizado = (
        validar(pd.DataFrame([veiculo]))[0].iloc[0].to_dict()
    )  # completa a bateria_estimada, como o prever
    base, contribuicoes = explicar(pacote.modelo, normalizado, f"mj_km_{ciclo}")
    # entrar como eletrificado espalha o efeito por entradas que o validar zera; junta numa linha só
    junta = ["propulsao", "combustivel", "bateria_kwh", "bateria_estimada"]
    if normalizado["cilindrada_cm3"] == 0:
        junta.append("cilindrada_cm3")
    rotulos = ROTULOS | {"propulsao_grupo": "Powertrain type", "demais": "Other inputs"}
    log = contribuicoes.drop(junta)
    log["propulsao_grupo"] = contribuicoes[junta].sum()
    top = log.reindex(log.abs().nlargest(8).index)
    top["demais"] = log.drop(top.index).sum()  # o que sobra, para os efeitos fecharem na previsão
    efeitos = ((np.exp(top) - 1) * 100).iloc[::-1]  # maior no topo do gráfico, "Demais entradas" embaixo
    st.caption(
        f"Average vehicle in the data: {_fixo(math.exp(base), 2)} MJ/km × effects = "
        f"{_fixo(math.exp(base + contribuicoes.sum()), 2)} MJ/km. Each bar is how much the input "
        "raises (+) or lowers (−) consumption compared with it; the effects multiply. "
        '"Powertrain type" groups powertrain, fuel and battery (and the zero displacement of electric cars).'
    )
    fig = go.Figure(
        go.Bar(
            x=efeitos.values,
            y=[rotulos.get(e, e) for e in efeitos.index],
            orientation="h",
            marker_color=[CORES["primaria"] if v > 0 else CORES["ciano"] for v in efeitos.values],
            texttemplate="%{x:+.1f} %",
            hovertemplate="%{y}: %{x:+.1f} %<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="effect on MJ/km (%)", height=360, margin={"t": 30})
    st.plotly_chart(fig)
    return ciclo, normalizado


PRESETS = {  # chave do botão: (rótulo, variações em %)
    "preset_aliviar": ("10 % lighter", {"es_massa": -10}),
    "preset_pneus": ("Tires −15 % F0", {"es_f0": -15}),
    "preset_aero": ("Aero −10 % F2", {"es_f2": -10}),
    "preset_pacote": ("All three −10 %", {"es_massa": -10, "es_f0": -10, "es_f2": -10}),
}
SLIDERS = ["es_massa", "es_f0", "es_f2", "es_potencia"]


def _definir(valores: dict) -> None:
    st.session_state.update(valores)


def _controles(veiculo: dict) -> dict:
    """Preset buttons and change sliders; returns the adjusted vehicle."""
    st.subheader("What if")
    st.caption("Move the controls to see how much consumption changes against the original vehicle.")
    for coluna, (chave, (rotulo, valores)) in zip(st.columns(len(PRESETS)), PRESETS.items()):
        coluna.button(rotulo, key=chave, on_click=_definir, args=(valores,), width="stretch")
    c1, c2 = st.columns(2)
    variacao = {
        "massa_om_kg": c1.slider("Mass change (%)", -20, 20, 0, 5, format="%d %%", key="es_massa"),
        "f0_n": c2.slider("F0 change (%)", -20, 20, 0, 5, format="%d %%", key="es_f0"),
        "f2_n_kmh2": c1.slider("F2 change (%)", -20, 20, 0, 5, format="%d %%", key="es_f2"),
        "potencia_kw": c2.slider("Power change (%)", -20, 20, 0, 5, format="%d %%", key="es_potencia"),
    }
    alterado = dict(veiculo)
    for campo, pct in variacao.items():
        alterado[campo] = veiculo[campo] * (1 + pct / 100)
    if variacao["massa_om_kg"]:
        alterado["classe_inercia_kg"] = classe_inercia(alterado["massa_om_kg"])
    return alterado


def _medidor(original: pd.Series, novo: pd.Series) -> None:
    ref = original.mj_km
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=novo.mj_km,
            number={"valueformat": ".2f", "suffix": " MJ/km"},
            delta={
                "reference": ref,
                "valueformat": ".2f",
                "decreasing": {"color": CORES["ciano"]},
                "increasing": {"color": CORES["primaria"]},
            },
            gauge={
                "axis": {"range": [0.5 * ref, 1.5 * ref]},
                "bar": {"color": CORES["primaria"]},
                "steps": [{"range": [novo.mj_km_min, novo.mj_km_max], "color": "#DDD6FE"}],  # faixa de 90 %
            },
        )
    )
    fig.update_layout(height=260, margin={"t": 30, "b": 10})
    st.plotly_chart(fig)


def _e_se(pacote, alterado: dict, original: pd.Series) -> pd.Series | None:
    """Medidor, métrica e avisos do veículo ajustado; devolve sua previsão (None, se o ajuste falhar)."""
    novo = _prever(pacote, alterado)
    if novo is None:
        return None
    _medidor(original, novo)
    diferenca = novo.mj_km - original.mj_km
    delta = (
        f"{_sinal(diferenca, 2)} MJ/km ({_sinal(diferenca / original.mj_km * 100, 1)} %) against the original"
        if diferenca
        else None  # sem variação, sem seta
    )
    st.metric("MJ/km with the changes", f"{_fixo(novo.mj_km, 2)} MJ/km", delta, delta_color="inverse")
    for aviso in novo["avisos"]:
        if aviso not in original["avisos"]:
            st.warning(aviso)
    return novo


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
            rotulos.append(f"{c['nome']}<br>baseline")
        else:
            diferenca = valores[i] - base
            rotulos.append(
                f"{c['nome']}<br>{_sinal(diferenca, 2)} MJ/km ({_sinal(diferenca / base * 100, 1)} %)"
            )
    fig = go.Figure(
        go.Bar(
            x=rotulos,
            y=valores,
            marker_color=["#94A3B8"] + [CORES["primaria"]] * (len(cenarios) - 1),
            error_y={
                "type": "data",
                "array": [c["resultado"]["mj_km_max"] - v for c, v in zip(cenarios, valores)],
                "arrayminus": [v - c["resultado"]["mj_km_min"] for c, v in zip(cenarios, valores)],
                "visible": True,
            },
        )
    )
    fig.update_layout(yaxis_title="MJ/km combined", height=420, margin={"t": 30})
    st.plotly_chart(fig)


def _cenarios() -> None:
    st.subheader("Scenarios")
    st.caption("The first saved scenario is the baseline for the comparison.")
    if "nome_cenario" not in st.session_state:
        st.session_state["nome_cenario"] = nome_padrao([c["nome"] for c in st.session_state["cenarios"]])
    st.text_input("Scenario name", key="nome_cenario")

    cenarios = st.session_state["cenarios"]
    if not cenarios:
        return
    _grafico(cenarios)
    for i, c in enumerate(cenarios):
        col_nome, col_mj, col_botao = st.columns([3, 2, 1])
        col_nome.write(f"**{c['nome']}**" + (" (baseline)" if i == 0 else ""))
        col_mj.write(f"{_fixo(c['resultado']['mj_km'], 2)} MJ/km")
        col_botao.button("Remove", key=f"remover_{i}", on_click=_remover, args=(i,))
    st.download_button(
        "Export scenarios (Excel)",
        data=excel_cenarios(cenarios),
        file_name="vehicle_scenarios.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


st.set_page_config(page_title="Vehicle", layout="wide")
st.title("Vehicle")
try:
    pacote = modelo()
except FileNotFoundError:
    st.error("No trained model. Run: uv run eficiencia-treinar")
    st.stop()

if "cenarios" not in st.session_state:
    st.session_state["cenarios"] = []

resumo = st.container()  # preenchido depois do formulário, com os valores atuais
with st.expander("Edit the vehicle specifications", expanded=False):
    veiculo = formulario_veiculo("veiculo")
resumo.caption(
    f"{nome_valor('propulsao', veiculo['propulsao'])} · {nome_valor('combustivel', veiculo['combustivel'])}"
    f" · {nome_valor('transmissao', veiculo['transmissao'])}"
    f" · mass {numero(veiculo['massa_om_kg'], 0)} kg · F0 {numero(veiculo['f0_n'], 1)} N"
    f" · F2 {numero(veiculo['f2_n_kmh2'], 4)} N/(km/h)²"
)
original = _prever(pacote, veiculo)
if original is None:
    st.stop()

esquerda, direita = st.columns(2)
caixa = esquerda.container(border=True)
with caixa:
    alterado = _controles(veiculo)
with direita:
    novo = _e_se(pacote, alterado, original)
    if novo is None:  # invalid adjustment: the saved scenario is the original
        alterado, novo = veiculo, original
with caixa:
    salvar, zerar = st.columns(2)
    salvar.button(
        "Save scenario",
        key="salvar_cenario",
        on_click=_salvar,
        args=("nome_cenario", {"veiculo": alterado, "resultado": novo.to_dict()}),
    )
    zerar.button(
        "Back to original", key="zerar_simulador", on_click=_definir, args=({k: 0 for k in SLIDERS},)
    )
    st.caption("Saves the vehicle with the current changes (sliders at 0 = the vehicle in the form)")

_resultado(original)
contexto = _explicacao(pacote, veiculo)

st.subheader("Inside the model")
aba_arvore, aba_rede = st.tabs(["XGBoost tree", "Neural network"])
with aba_arvore:
    if contexto:
        ciclo, normalizado = contexto
        nos = caminho_arvore(pacote.modelo, normalizado, f"mj_km_{ciclo}")
        n = pacote.modelo.estimadores[f"mj_km_{ciclo}"].get_booster().num_boosted_rounds()
        st.caption(
            f"Tree 1 of {n} in the XGBoost model (target MJ/km {CICLOS[ciclo]}). The leaf value is a small "
            f"correction (on a log scale) added to those of the other {n - 1} trees. The signal follows the "
            "vehicle's path; the other nodes are dimmed."
        )
        st.iframe(html_arvore(nos), height=500)
    else:
        st.caption("The tree is only available when the selected model is XGBoost.")
with aba_rede:
    st.caption(
        "Illustrative sketch of a neural network, one of the training candidates; the number of neurons "
        f"is not the real one. The app uses the {NOMES.get(pacote.algoritmo, pacote.algoritmo)} model."
    )
    st.iframe(html_rede(), height=440)
_cenarios()
