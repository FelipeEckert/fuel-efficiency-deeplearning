"""Vehicle form: the model inputs as Streamlit widgets, shared by the app pages."""

import json
import os
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from eficiencia.artefatos import PASTA_MODELO, Pacote, carregar
from eficiencia.esquema import (
    CATEGORICAS,
    ENTRADAS,
    ROTULOS,
    VALORES,
    caminho_base,
    carregar_base,
    nome_valor,
)
from eficiencia.fisica import classe_inercia
from eficiencia.formato import numero
from eficiencia.prever import SAIDAS

VEICULO_PADRAO = Path(__file__).with_name("veiculo_padrao.json")

CATEGORIAS = {
    "SC": "Subcompact",
    "C": "Compact",
    "M": "Midsize",
    "G": "Large",
    "EXG": "Extra large",
    "SUVC": "Compact SUV",
    "SUVC4": "Compact SUV 4x4",
    "SUVG": "Large SUV",
    "SUVG4": "Large SUV 4x4",
    "FEC": "Compact off-road",
    "FEG": "Large off-road",
    "MIN": "Minivan",
    "CL": "Light commercial",
    "PIC": "Pickup",
    "PC": "Compact pickup",
    "E": "Sports",
}
CAMPOS = list(ENTRADAS)
# medianas do PBEV 2026 por propulsão: a base não é versionada, então os valores típicos ficam aqui
TIPICOS = {
    "Híbrido": {"bateria_kwh": 0.9},
    "Plug-in": {"bateria_kwh": 19.4, "autonomia_eletrica_km": 61.2},
    "Elétrico": {"bateria_kwh": 66.4, "autonomia_eletrica_km": 332.4},
}


def _nome(campo: str):
    return lambda codigo: nome_valor(campo, codigo)


@st.cache_resource
def _carregar_pacote(pasta: str, mtime: float) -> Pacote:
    """O mtime entra só na chave do cache: um treino novo (arquivo regravado) carrega o modelo de novo."""
    return carregar(Path(pasta))


def modelo() -> Pacote:
    pasta = Path(os.environ.get("EFICIENCIA_MODELO", PASTA_MODELO))
    return _carregar_pacote(str(pasta), (pasta / "pacote.joblib").stat().st_mtime)


def _padrao_base() -> dict:
    """Veículo típico: mediana (numéricas) ou moda (categóricas) dos veículos de Combustão da base.

    Sem a base local (ela não é versionada), usa o veículo típico salvo em veiculo_padrao.json.
    """
    if not caminho_base().exists():
        return json.loads(VEICULO_PADRAO.read_text(encoding="utf-8"))
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
    propulsao = st.session_state[f"{chave}_propulsao"]
    faixas = modelo().faixas["numericas"].get(propulsao, {})
    for campo, mediana in TIPICOS.get(propulsao, {}).items():
        minimo, maximo = faixas.get(campo, (mediana, mediana))
        atual = st.session_state[f"{chave}_{campo}"]
        if atual == 0 or not minimo <= atual <= maximo:  # zerado ou fora da faixa da propulsão vira a mediana
            st.session_state[f"{chave}_{campo}"] = mediana
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
    """Desenha o formulário e devolve as entradas do veículo.

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

    st.subheader("Identification")
    c1, c2 = st.columns(2)
    propulsao = c1.selectbox(
        ROTULOS["propulsao"],
        VALORES["propulsao"],
        key=chave_de("propulsao"),
        format_func=_nome("propulsao"),
        on_change=_ao_mudar_propulsao,
        args=(chave,),
    )
    eletrico = propulsao == "Elétrico"
    combustivel = c2.selectbox(
        ROTULOS["combustivel"],
        VALORES["combustivel"],
        key=chave_de("combustivel"),
        disabled=eletrico,
        format_func=_nome("combustivel"),
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
    transmissao = c1.selectbox(
        ROTULOS["transmissao"],
        VALORES["transmissao"],
        key=chave_de("transmissao"),
        format_func=_nome("transmissao"),
    )
    direcao = c2.selectbox(
        ROTULOS["direcao"], VALORES["direcao"], key=chave_de("direcao"), format_func=_nome("direcao")
    )

    st.subheader("Mass and inertia")
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

    st.subheader("Road load")
    referencia = pacote.referencias.get(categoria, {})
    c1, c2 = st.columns(2)
    f0 = c1.number_input(f"{ROTULOS['f0_n']} (N)", key=chave_de("f0_n"), step=1.0)
    if "f0_n" in referencia:
        c1.caption(f"typical for the category: {numero(referencia['f0_n'], 1)} N")
    f2 = c2.number_input(
        f"{ROTULOS['f2_n_kmh2']} (N/(km/h)²)", key=chave_de("f2_n_kmh2"), step=0.001, format="%.4f"
    )
    if "f2_n_kmh2" in referencia:
        c2.caption(f"typical for the category: {numero(referencia['f2_n_kmh2'], 4)} N/(km/h)²")

    st.subheader("Powertrain and battery")
    c1, c2, c3 = st.columns(3)
    potencia = c1.number_input(ROTULOS["potencia_kw"], key=chave_de("potencia_kw"), step=1.0)
    cilindrada = c2.number_input(
        ROTULOS["cilindrada_cm3"], key=chave_de("cilindrada_cm3"), step=10.0, disabled=eletrico
    )
    marchas = c3.number_input(ROTULOS["n_marchas"], key=chave_de("n_marchas"), step=1.0)
    bateria = autonomia = 0.0
    if propulsao != "Combustão":
        c1, c2 = st.columns(2)
        bateria = c1.number_input(ROTULOS["bateria_kwh"], key=chave_de("bateria_kwh"), step=1.0)
        if propulsao in ("Plug-in", "Elétrico"):
            autonomia = c2.number_input(
                ROTULOS["autonomia_eletrica_km"], key=chave_de("autonomia_eletrica_km"), step=1.0
            )

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
        "autonomia_eletrica_km": autonomia,
    }


def nome_padrao(nomes: list[str]) -> str:
    """Next "Scenario N" not used yet."""
    n = len(nomes) + 1
    while f"Scenario {n}" in nomes:
        n += 1
    return f"Scenario {n}"


def nome_livre(nome: str, nomes: list[str]) -> str:
    """O nome, ou "nome (2)", "nome (3)"... se já existir, para as barras não se sobreporem."""
    if nome not in nomes:
        return nome
    n = 2
    while f"{nome} ({n})" in nomes:
        n += 1
    return f"{nome} ({n})"


_COMBUSTIVEIS = {
    "gasolina": "gasoline",
    "etanol": "ethanol",
    "eletrico_eq": "electric equivalent",
    "diesel": "diesel",
}
_CICLOS = {"": "combined", "urbano": "city", "estrada": "highway"}


def nome_saida(saida: str) -> str:
    """English name of a prediction output: mj_km_urbano -> MJ/km city, kml_etanol -> km/l ethanol combined."""
    if saida.startswith("mj_km"):
        return f"MJ/km {_CICLOS[saida.removeprefix('mj_km').lstrip('_')]}"
    resto = saida.removeprefix("kml_")
    for codigo, nome in _COMBUSTIVEIS.items():
        if resto.startswith(codigo):
            return f"km/l {nome} {_CICLOS[resto.removeprefix(codigo).lstrip('_')]}"
    return saida


def excel_cenarios(cenarios: list[dict]) -> bytes:
    """Uma linha por cenário: entradas que mudam em algum cenário e saídas com faixa."""
    base = cenarios[0]["veiculo"]
    alteradas = [
        campo
        for campo in ENTRADAS
        if any(c["veiculo"][campo] != base[campo] and not pd.isna(c["veiculo"][campo]) for c in cenarios)
    ]
    linhas = []
    for c in cenarios:
        linha = {"Scenario": c["nome"]}
        linha.update({ROTULOS[campo]: c["veiculo"][campo] for campo in alteradas})
        for saida in SAIDAS:
            nome = nome_saida(saida)
            linha[nome] = c["resultado"][saida]
            linha[f"{nome} (min.)"] = c["resultado"][f"{saida}_min"]
            linha[f"{nome} (max.)"] = c["resultado"][f"{saida}_max"]
        linhas.append(linha)
    buffer = BytesIO()
    pd.DataFrame(linhas).to_excel(buffer, index=False)
    return buffer.getvalue()
