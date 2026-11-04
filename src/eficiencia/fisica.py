"""Conversões físicas: classe de inércia (NBR 6601), mistura urbano/estrada e equivalente elétrico."""

from bisect import bisect_right

import pandas as pd

# (mínimo da faixa em kg, classe de inércia) — spec §8.1
_FAIXAS_NBR_6601 = [
    (0, 454),
    (483, 510),
    (539, 567),
    (596, 624),
    (653, 680),
    (710, 737),
    (766, 794),
    (823, 850),
    (880, 907),
    (936, 964),
    (993, 1021),
    (1050, 1077),
    (1106, 1134),
    (1163, 1191),
    (1220, 1247),
    (1277, 1304),
    (1333, 1361),
    (1390, 1417),
    (1447, 1474),
    (1503, 1531),
    (1560, 1588),
    (1617, 1644),
    (1673, 1701),
    (1730, 1758),
    (1787, 1814),
    (1872, 1928),
    (1985, 2041),
    (2099, 2155),
    (2212, 2268),
    (2326, 2381),
    (2439, 2495),
]  # fora da tabela: a linha "2608→2495" da planilha é erro de digitação
_MINIMOS = [m for m, _ in _FAIXAS_NBR_6601]
MASSA_ENSAIO_EXTRA_KG = 136
FATOR_ELETRICO = 20.29


def classe_inercia(massa_om_kg: float) -> float:
    massa_ensaio = massa_om_kg + MASSA_ENSAIO_EXTRA_KG
    return _FAIXAS_NBR_6601[bisect_right(_MINIMOS, massa_ensaio) - 1][1]


def combinar_mj(urbano, estrada):
    return 0.55 * urbano + 0.45 * estrada


def combinar_kml(urbano, estrada):
    return 1 / (0.55 / urbano + 0.45 / estrada)


def kml_eletrico_eq(mj_km):
    return FATOR_ELETRICO / mj_km


def fatores_diesel(df: pd.DataFrame) -> tuple[float, float]:
    urbano = (df["kml_diesel_urbano"] * df["mj_km_urbano"]).median()
    estrada = (df["kml_diesel_estrada"] * df["mj_km_estrada"]).median()
    return float(urbano), float(estrada)


def kml_diesel(mj_km, fator: float):
    return fator / mj_km


def combustiveis_aplicaveis(propulsao: str, combustivel: str) -> set[str]:
    if propulsao == "Elétrico":
        return {"eletrico_eq"}
    aplicaveis = {"flex": {"gasolina", "etanol"}, "gasolina": {"gasolina"}, "diesel": {"diesel"}}
    aplicaveis = set(aplicaveis.get(combustivel, set()))
    if propulsao == "Plug-in":
        aplicaveis.add("eletrico_eq")
    return aplicaveis
