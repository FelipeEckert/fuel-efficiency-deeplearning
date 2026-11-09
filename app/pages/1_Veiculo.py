"""Página Veículo: estimativa de um veículo, "e se" sobre ele e cenários salvos para comparação."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # pasta app/, para importar formulario

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from formulario import NOMES_VALORES, excel_cenarios, formulario_veiculo, modelo, nome_livre, nome_padrao
from visual import CORES, NOMES, html_arvore, html_rede

from eficiencia.arvore import caminho_arvore
from eficiencia.entrada import validar
from eficiencia.esquema import ROTULOS
from eficiencia.explicar import explicar, explicavel
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


def _fixo(valor: float, casas: int) -> str:
    """Decimais fixos em pt-BR: 1,70 e não 1,7."""
    return f"{valor:.{casas}f}".replace(".", ",")


def _valor(r: pd.Series, saida: str, casas: int, unidade: str) -> str:
    return "—" if pd.isna(r[saida]) else f"{_fixo(r[saida], casas)} {unidade}"


def _faixa(r: pd.Series, saida: str, casas: int) -> str:
    """Só o intervalo de 90 %, no formato "faixa de 90 %: 1,54 – 1,70"; vazio quando a saída não tem faixa."""
    if pd.isna(r[f"{saida}_min"]) or pd.isna(r[f"{saida}_max"]):
        return ""
    return f"faixa de 90 %: {_fixo(r[f'{saida}_min'], casas)} – {_fixo(r[f'{saida}_max'], casas)}"


def _metrica(coluna, rotulo: str, r: pd.Series, saida: str, casas: int, unidade: str) -> None:
    coluna.metric(rotulo, _valor(r, saida, casas, unidade))  # sem delta: nenhuma seta sobre a faixa
    coluna.caption(_faixa(r, saida, casas))


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


def _explicacao(pacote, veiculo: dict) -> tuple[str, dict] | None:
    """Quanto cada entrada aumenta ou reduz o consumo em relação ao veículo médio da base (SHAP).

    Devolve (ciclo, veículo validado) para a árvore usar os mesmos valores; None se não for XGBoost."""
    st.subheader("Por que este valor?")
    if not explicavel(pacote.modelo):
        st.caption("A explicação por entrada está disponível só para o modelo XGBoost.")
        return None
    ciclo = st.radio("Ciclo", ["urbano", "estrada"], horizontal=True, key="shap_ciclo")
    normalizado = validar(pd.DataFrame([veiculo]))[0].iloc[0].to_dict()  # zera o que não vale, como o prever
    base, contribuicoes = explicar(pacote.modelo, normalizado, f"mj_km_{ciclo}")
    # entrar como eletrificado espalha o efeito por entradas que o validar zera; junta numa linha só
    junta = ["propulsao", "combustivel", "bateria_kwh", "autonomia_eletrica_km"]
    if normalizado["cilindrada_cm3"] == 0:
        junta.append("cilindrada_cm3")
    rotulos = ROTULOS | {"propulsao_grupo": "Tipo de propulsão", "demais": "Demais entradas"}
    log = contribuicoes.drop(junta)
    log["propulsao_grupo"] = contribuicoes[junta].sum()
    top = log.reindex(log.abs().nlargest(8).index)
    top["demais"] = log.drop(top.index).sum()  # o que sobra, para os efeitos fecharem na previsão
    efeitos = ((np.exp(top) - 1) * 100).iloc[::-1]  # maior no topo do gráfico, "Demais entradas" embaixo
    st.caption(
        f"Veículo médio da base: {_fixo(math.exp(base), 2)} MJ/km × efeitos = "
        f"{_fixo(math.exp(base + contribuicoes.sum()), 2)} MJ/km. Cada barra é o quanto a entrada "
        "aumenta (+) ou reduz (−) o consumo em relação a ele; os efeitos se multiplicam. "
        '"Tipo de propulsão" junta propulsão, combustível, bateria e autonomia elétrica (e a cilindrada zero dos elétricos).'
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
    fig.update_layout(xaxis_title="efeito no MJ/km (%)", separators=",.", height=360, margin={"t": 30})
    st.plotly_chart(fig)
    return ciclo, normalizado


PRESETS = {  # chave do botão: (rótulo, variações em %)
    "preset_aliviar": ("Aliviar 10 %", {"es_massa": -10}),
    "preset_pneus": ("Pneus verdes −15 % F0", {"es_f0": -15}),
    "preset_aero": ("Aero −10 % F2", {"es_f2": -10}),
    "preset_pacote": ("Pacote −10 % em tudo", {"es_massa": -10, "es_f0": -10, "es_f2": -10}),
}
SLIDERS = ["es_massa", "es_f0", "es_f2", "es_potencia"]


def _definir(valores: dict) -> None:
    st.session_state.update(valores)


def _controles(veiculo: dict) -> dict:
    """Botões prontos e sliders de variação; devolve o veículo ajustado."""
    st.subheader("Simulador de cenários")
    st.caption("Mexa nos controles para ver quanto o consumo muda em relação ao veículo original.")
    for coluna, (chave, (rotulo, valores)) in zip(st.columns(len(PRESETS)), PRESETS.items()):
        coluna.button(rotulo, key=chave, on_click=_definir, args=(valores,), width="stretch")
    c1, c2 = st.columns(2)
    variacao = {
        "massa_om_kg": c1.slider("Variação da massa (%)", -20, 20, 0, 5, format="%d %%", key="es_massa"),
        "f0_n": c2.slider("Variação da F0 (%)", -20, 20, 0, 5, format="%d %%", key="es_f0"),
        "f2_n_kmh2": c1.slider("Variação da F2 (%)", -20, 20, 0, 5, format="%d %%", key="es_f2"),
        "potencia_kw": c2.slider(
            "Variação da potência (%)", -20, 20, 0, 5, format="%d %%", key="es_potencia"
        ),
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
    fig.update_layout(height=260, margin={"t": 30, "b": 10}, separators=",.")
    st.plotly_chart(fig)


def _e_se(pacote, alterado: dict, original: pd.Series) -> pd.Series | None:
    """Medidor, métrica e avisos do veículo ajustado; devolve sua previsão (None, se o ajuste falhar)."""
    novo = _prever(pacote, alterado)
    if novo is None:
        return None
    _medidor(original, novo)
    diferenca = novo.mj_km - original.mj_km
    delta = (
        f"{_sinal(diferenca, 2)} MJ/km ({_sinal(diferenca / original.mj_km * 100, 1)} %) em relação ao original"
        if diferenca
        else None  # sem variação, sem seta
    )
    st.metric("MJ/km com as mudanças", f"{_fixo(novo.mj_km, 2)} MJ/km", delta, delta_color="inverse")
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
            marker_color=["#94A3B8"] + [CORES["primaria"]] * (len(cenarios) - 1),
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
        col_mj.write(f"{_fixo(c['resultado']['mj_km'], 2)} MJ/km")
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

resumo = st.container()  # preenchido depois do formulário, com os valores atuais
with st.expander("Editar ficha técnica do veículo", expanded=False):
    veiculo = formulario_veiculo("veiculo")
resumo.caption(
    f"{veiculo['propulsao']} · {NOMES_VALORES['combustivel'].get(veiculo['combustivel'], veiculo['combustivel'])}"
    f" · {NOMES_VALORES['transmissao'].get(veiculo['transmissao'], veiculo['transmissao'])}"
    f" · massa {numero(veiculo['massa_om_kg'], 0)} kg · F0 {numero(veiculo['f0_n'], 1)} N"
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
    if novo is None:  # ajuste inválido: o cenário salvo é o original
        alterado, novo = veiculo, original
with caixa:
    salvar, zerar = st.columns(2)
    salvar.button(
        "Salvar cenário",
        key="salvar_cenario",
        on_click=_salvar,
        args=("nome_cenario", {"veiculo": alterado, "resultado": novo.to_dict()}),
    )
    zerar.button(
        "Voltar ao original", key="zerar_simulador", on_click=_definir, args=({k: 0 for k in SLIDERS},)
    )
    st.caption("Salva o veículo com os ajustes atuais (sliders em 0 = veículo do formulário)")

_resultado(original)
contexto = _explicacao(pacote, veiculo)

st.subheader("O modelo por dentro")
aba_arvore, aba_rede = st.tabs(["Árvore do XGBoost", "Rede neural"])
with aba_arvore:
    if contexto:
        ciclo, normalizado = contexto
        nos = caminho_arvore(pacote.modelo, normalizado, f"mj_km_{ciclo}")
        n = pacote.modelo.estimadores[f"mj_km_{ciclo}"].get_booster().num_boosted_rounds()
        st.caption(
            f"Árvore 1 de {n} do XGBoost (alvo MJ/km {ciclo}). O valor da folha é uma pequena correção "
            f"(em escala log) que se soma às das outras {n - 1} árvores. O sinal percorre o caminho do "
            "veículo; os demais nós ficam apagados."
        )
        st.iframe(html_arvore(nos), height=500)
    else:
        st.caption("A árvore está disponível só quando o modelo vencedor é o XGBoost.")
with aba_rede:
    st.caption(
        "Esquema ilustrativo de uma rede neural, uma das candidatas do treino; o número de neurônios "
        f"não é o real. O app usa o modelo {NOMES.get(pacote.algoritmo, pacote.algoritmo)}."
    )
    st.iframe(html_rede(), height=440)
_cenarios()
