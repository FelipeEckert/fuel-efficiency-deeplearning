"""Faixas de incerteza por quantil conformal dos erros em validação cruzada (out-of-fold).

O erro é relativo ao previsto, |real − previsto| / previsto, pois a faixa é previsto × (1 ± q):
dividir pelo real deixaria a faixa com cobertura menor que a nominal."""

import math

import pandas as pd

from eficiencia.esquema import ALVOS
from eficiencia.fisica import combinar_kml, combinar_mj

SAIDAS_COM_FAIXA = ALVOS + ["mj_km", "kml_gasolina", "kml_etanol"]


def _real_e_previsto(df: pd.DataFrame, oof: pd.DataFrame, saida: str) -> tuple[pd.Series, pd.Series] | None:
    if saida in ALVOS:
        if saida in df and saida in oof:
            return df[saida], oof[saida]
        return None
    if saida == "mj_km":
        if {"mj_km_urbano", "mj_km_estrada"} <= set(oof.columns) and "mj_km" in df:
            return df["mj_km"], combinar_mj(oof["mj_km_urbano"], oof["mj_km_estrada"])
        return None
    # kml_gasolina / kml_etanol: combinação harmônica de urbano e estrada
    urbano, estrada = f"{saida}_urbano", f"{saida}_estrada"
    if {urbano, estrada} <= set(oof.columns) and {urbano, estrada} <= set(df.columns):
        real = combinar_kml(df[urbano], df[estrada])
        return real, combinar_kml(oof[urbano], oof[estrada])
    return None


def calcular_quantis(df: pd.DataFrame, oof: pd.DataFrame, nivel: float = 0.9) -> dict[str, dict[str, float]]:
    quantis: dict[str, dict[str, float]] = {}
    for propulsao in df["propulsao"].unique():
        linhas = df["propulsao"] == propulsao
        quantis[propulsao] = {}
        for saida in SAIDAS_COM_FAIXA:
            par = _real_e_previsto(df[linhas], oof.loc[df.index[linhas]], saida)
            if par is None:
                continue
            real, previsto = par
            pares = pd.concat([real.rename("real"), previsto.rename("previsto")], axis=1).dropna()
            if pares.empty:
                continue
            erros = (
                ((pares["real"] - pares["previsto"]).abs() / pares["previsto"].abs()).sort_values().to_numpy()
            )
            n = len(erros)
            posicao = min(math.ceil(round((n + 1) * nivel, 9)), n)
            quantis[propulsao][saida] = float(erros[posicao - 1])
    return quantis


def erros_relativos(df: pd.DataFrame, oof: pd.DataFrame) -> dict[str, list[float]]:
    """Erros (real − previsto) / previsto do MJ/km combinado por propulsão, com sinal."""
    real, previsto = _real_e_previsto(df, oof, "mj_km")
    erros = ((real - previsto) / previsto.abs()).dropna()
    return {p: erros[df.loc[erros.index, "propulsao"] == p].tolist() for p in df["propulsao"].unique()}


def aplicar(previsto: float, q: float) -> tuple[float, float]:
    return previsto * (1 - q), previsto * (1 + q)
