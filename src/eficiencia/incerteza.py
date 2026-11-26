"""Faixas de incerteza por quantil conformal dos erros em validação cruzada (out-of-fold).

O erro é relativo ao previsto, |real − previsto| / previsto, pois a faixa é previsto × (1 ± q):
dividir pelo real deixaria a faixa com cobertura menor que a nominal."""

import math

import pandas as pd

from eficiencia.esquema import ALVOS
from eficiencia.fisica import combinar_kml, combinar_mj

SAIDAS_COM_FAIXA = ALVOS + [
    "mj_km",
    "kml_gasolina",
    "kml_etanol",
    "kml_diesel_urbano",
    "kml_diesel_estrada",
    "kml_diesel",
]
CICLOS = ("urbano", "estrada")


def _real_e_previsto(df: pd.DataFrame, oof: pd.DataFrame, saida: str) -> tuple[pd.Series, pd.Series] | None:
    if saida in df and saida in oof:  # saída por ciclo
        return df[saida], oof[saida]
    if saida in ALVOS or saida.endswith(CICLOS):
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


def _kml_do_app(df: pd.DataFrame, oof: pd.DataFrame, fatores: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Combustão e híbrido: o app mostra km/l = fator / MJ/km, então é esse km/l que tem o erro medido."""
    oof = oof.copy()
    linhas = df["propulsao"].isin(["Combustão", "Híbrido"]).to_numpy()
    for comb, por_ciclo in fatores.items():
        for c, fator in zip(CICLOS, por_ciclo):
            coluna = f"kml_{comb}_{c}"
            if coluna not in oof:
                oof[coluna] = float("nan")
            oof.loc[linhas, coluna] = fator / oof.loc[linhas, f"mj_km_{c}"]
    return oof


def calcular_quantis(
    df: pd.DataFrame,
    oof: pd.DataFrame,
    fatores: dict[str, tuple[float, float]] | None = None,
    nivel: float = 0.9,
) -> dict[str, dict[str, float]]:
    """Quantis por propulsão e saída. Com os fatores de km/l, o km/l de combustão e híbrido é o do app."""
    if fatores is not None:
        oof = _kml_do_app(df, oof, fatores)
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
