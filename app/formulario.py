"""Formulário de um veículo: as entradas do modelo em widgets Streamlit, reutilizado pelas páginas do app."""

from io import BytesIO

import pandas as pd
import streamlit as st

from eficiencia.artefatos import Pacote, carregar
from eficiencia.esquema import CATEGORICAS, ENTRADAS, ROTULOS, VALORES, carregar_base
from eficiencia.fisica import classe_inercia
from eficiencia.formato import numero
from eficiencia.prever import SAIDAS

CATEGORIAS = {
    "SC": "Subcompacto",
    "C": "Compacto",
    "M": "Médio",
    "G": "Grande",
    "EXG": "Extra Grande",
    "SUVC": "Utilitário Esportivo Compacto",
    "SUVC4": "Utilitário Esportivo Compacto 4x4",
    "SUVG": "Utilitário Esportivo Grande",
    "SUVG4": "Utilitário Esportivo Grande 4x4",
    "FEC": "Fora de Estrada Compacto",
    "FEG": "Fora de Estrada Grande",
    "MIN": "Minivan",
    "CL": "Comercial",
    "PIC": "Picape",
    "PC": "Picape Compacta",
    "E": "Esportivo",
}
CAMPOS = [c for c in ENTRADAS if c != "bateria_estimada"]  # sempre False no app


@st.cache_resource
def modelo() -> Pacote:
    return carregar()


def _padrao_base() -> dict:
    """Veículo típico: mediana (numéricas) ou moda (categóricas) dos veículos de Combustão da base."""
    combustao = carregar_base().query('propulsao == "Combustão"')
    padrao = {}
    for campo in CAMPOS:
        if campo in CATEGORICAS:
            padrao[campo] = combustao[campo].mode().iloc[0]
        else:
            padrao[campo] = float(combustao[campo].median())
    padrao["classe_inercia_kg"] = classe_inercia(padrao["massa_om_kg"])
    return padrao


def _ao_mudar_massa(chave: str) -> None:
    st.session_state[f"{chave}_classe_inercia_kg"] = classe_inercia(st.session_state[f"{chave}_massa_om_kg"])


def _ao_mudar_propulsao(chave: str) -> None:
    if st.session_state[f"{chave}_propulsao"] == "Elétrico":
        st.session_state[f"{chave}_combustivel"] = "eletrico"
        st.session_state[f"{chave}_transmissao"] = "direta"
        st.session_state[f"{chave}_cilindrada_cm3"] = 0.0
    else:  # saiu do elétrico: combustível coerente e, se a cilindrada estava zerada, a típica
        if st.session_state[f"{chave}_combustivel"] == "eletrico":
            st.session_state[f"{chave}_combustivel"] = "gasolina"
        if st.session_state[f"{chave}_cilindrada_cm3"] == 0:
            st.session_state[f"{chave}_cilindrada_cm3"] = _padrao_base()["cilindrada_cm3"]


def formulario_veiculo(chave: str, padrao: dict | None = None) -> dict:
    """Desenha o formulário e devolve as entradas do veículo (sem bateria_estimada).

    `chave` prefixa as chaves dos widgets, para o formulário poder ser usado mais de uma vez na página.
    Os valores iniciais vêm de `padrao` na primeira execução; depois, o que estiver nos widgets vale.
    """
    padrao = padrao or _padrao_base()
    for campo in CAMPOS:
        st.session_state.setdefault(f"{chave}_{campo}", padrao[campo])
    pacote = modelo()

    def chave_de(campo: str) -> str:
        return f"{chave}_{campo}"

    def rotulo_categoria(codigo: str) -> str:
        return f"{codigo} – {CATEGORIAS[codigo]}"

    st.subheader("Identificação")
    c1, c2 = st.columns(2)
    propulsao = c1.selectbox(
        "Propulsão",
        VALORES["propulsao"],
        key=chave_de("propulsao"),
        on_change=_ao_mudar_propulsao,
        args=(chave,),
    )
    eletrico = propulsao == "Elétrico"
    combustivel = c2.selectbox(
        ROTULOS["combustivel"], VALORES["combustivel"], key=chave_de("combustivel"), disabled=eletrico
    )
    categoria = c1.selectbox(
        ROTULOS["categoria"],
        list(CATEGORIAS),
        key=chave_de("categoria"),
        format_func=rotulo_categoria,
    )
    proconve = c2.selectbox(
        ROTULOS["categoria_proconve"], VALORES["categoria_proconve"], key=chave_de("categoria_proconve")
    )
    transmissao = c1.selectbox(ROTULOS["transmissao"], VALORES["transmissao"], key=chave_de("transmissao"))
    direcao = c2.selectbox(ROTULOS["direcao"], VALORES["direcao"], key=chave_de("direcao"))

    st.subheader("Massa e inércia")
    c1, c2, c3 = st.columns(3)
    massa = c1.number_input(
        ROTULOS["massa_om_kg"],
        key=chave_de("massa_om_kg"),
        step=10.0,
        on_change=_ao_mudar_massa,
        args=(chave,),
    )
    inercia = c2.number_input(ROTULOS["classe_inercia_kg"], key=chave_de("classe_inercia_kg"), step=1.0)
    pbt = c3.number_input(ROTULOS["pbt_kg"], key=chave_de("pbt_kg"), step=10.0)

    st.subheader("Resistência ao rolamento e aerodinâmica")
    referencia = pacote.referencias.get(categoria, {})
    c1, c2 = st.columns(2)
    f0 = c1.number_input(f"{ROTULOS['f0_n']} (N)", key=chave_de("f0_n"), step=1.0)
    if "f0_n" in referencia:
        c1.caption(f"típico da categoria: {numero(referencia['f0_n'], 1)} N")
    f2 = c2.number_input(
        f"{ROTULOS['f2_n_kmh2']} (N/(km/h)²)", key=chave_de("f2_n_kmh2"), step=0.001, format="%.4f"
    )
    if "f2_n_kmh2" in referencia:
        c2.caption(f"típico da categoria: {numero(referencia['f2_n_kmh2'], 4)} N/(km/h)²")

    st.subheader("Motor, transmissão e bateria")
    c1, c2, c3 = st.columns(3)
    potencia = c1.number_input(ROTULOS["potencia_kw"], key=chave_de("potencia_kw"), step=1.0)
    cilindrada = c2.number_input(
        ROTULOS["cilindrada_cm3"], key=chave_de("cilindrada_cm3"), step=10.0, disabled=eletrico
    )
    marchas = c3.number_input(ROTULOS["n_marchas"], key=chave_de("n_marchas"), step=1.0)
    bateria = 0.0
    if propulsao != "Combustão":
        bateria = st.number_input(ROTULOS["bateria_kwh"], key=chave_de("bateria_kwh"), step=1.0)

    return {
        "propulsao": propulsao,
        "combustivel": "eletrico" if eletrico else combustivel,
        "categoria": categoria,
        "categoria_proconve": proconve,
        "transmissao": transmissao,
        "direcao": direcao,
        "massa_om_kg": massa,
        "classe_inercia_kg": inercia,
        "pbt_kg": pbt,
        "f0_n": f0,
        "f2_n_kmh2": f2,
        "potencia_kw": potencia,
        "cilindrada_cm3": 0.0 if eletrico else cilindrada,
        "n_marchas": marchas,
        "bateria_kwh": bateria,
    }


def nome_padrao(nomes: list[str]) -> str:
    """Próximo "Cenário N" ainda não usado."""
    n = len(nomes) + 1
    while f"Cenário {n}" in nomes:
        n += 1
    return f"Cenário {n}"


def nome_livre(nome: str, nomes: list[str]) -> str:
    """O nome, ou "nome (2)", "nome (3)"... se já existir, para as barras não se sobreporem."""
    if nome not in nomes:
        return nome
    n = 2
    while f"{nome} ({n})" in nomes:
        n += 1
    return f"{nome} ({n})"


def excel_cenarios(cenarios: list[dict]) -> bytes:
    """Uma linha por cenário: entradas que mudam em algum cenário e saídas com faixa."""
    base = cenarios[0]["veiculo"]
    alteradas = [
        campo
        for campo in ENTRADAS
        if campo != "bateria_estimada"
        and any(c["veiculo"][campo] != base[campo] and not pd.isna(c["veiculo"][campo]) for c in cenarios)
    ]
    linhas = []
    for c in cenarios:
        linha = {"Cenário": c["nome"]}
        linha.update({ROTULOS[campo]: c["veiculo"][campo] for campo in alteradas})
        for saida in SAIDAS:
            nome = ROTULOS.get(saida, saida)
            linha[nome] = c["resultado"][saida]
            linha[f"{nome} (mín.)"] = c["resultado"][f"{saida}_min"]
            linha[f"{nome} (máx.)"] = c["resultado"][f"{saida}_max"]
        linhas.append(linha)
    buffer = BytesIO()
    pd.DataFrame(linhas).to_excel(buffer, index=False)
    return buffer.getvalue()
