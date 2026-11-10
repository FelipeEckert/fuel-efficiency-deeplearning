"""Ranges observed in the data per powertrain, and warnings for inputs outside them."""

from collections.abc import Mapping

import pandas as pd

from eficiencia.esquema import CATEGORICAS, NUMERICAS, ROTULOS, nome_valor
from eficiencia.formato import numero


def calcular_faixas(df: pd.DataFrame) -> dict:
    faixas: dict = {"numericas": {}, "categoricas": {}}
    for propulsao, grupo in df.groupby("propulsao"):
        faixas["numericas"][propulsao] = {
            c: (float(grupo[c].min()), float(grupo[c].max())) for c in NUMERICAS if grupo[c].notna().any()
        }
        faixas["categoricas"][propulsao] = {
            c: set(grupo[c].dropna()) for c in CATEGORICAS if grupo[c].notna().any()
        }
    return faixas


def avisos(veiculo: Mapping, faixas: dict) -> list[str]:
    propulsao = veiculo.get("propulsao")
    numericas = faixas["numericas"].get(propulsao, {})
    categoricas = faixas["categoricas"].get(propulsao, {})
    mensagens: list[str] = []
    for coluna in NUMERICAS + CATEGORICAS:  # numéricas primeiro: ordem exigida pelo teste da tarefa 7
        valor = veiculo.get(coluna)
        if valor is None or pd.isna(valor):
            continue
        rotulo = ROTULOS[coluna]
        grupo = nome_valor("propulsao", propulsao).lower()
        if coluna in numericas:
            minimo, maximo = numericas[coluna]
            if valor < minimo:
                mensagens.append(
                    f"{rotulo} = {numero(valor)} is below every {grupo} vehicle in the data (minimum {numero(minimo)})"
                )
            elif valor > maximo:
                mensagens.append(
                    f"{rotulo} = {numero(valor)} is above every {grupo} vehicle in the data (maximum {numero(maximo)})"
                )
        elif coluna in categoricas and coluna != "propulsao" and valor not in categoricas[coluna]:
            mensagens.append(
                f"{rotulo} = {nome_valor(coluna, valor)} was never seen in {grupo} vehicles in the data"
            )
    return mensagens
