"""Conversões físicas: classe de inércia (NBR 6601), mistura urbano/estrada, equivalente elétrico e
fator de utilidade WLTP com fração a combustão."""

from bisect import bisect_right

import numpy as np
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
# Reg. (UE) 2017/1151, fator de utilidade de frota: UF(d) = 1 − exp(−Σ aᵢ·(d/dn)ⁱ), dn = 800 km
_COEF_UF_WLTP = [26.25, -38.94, -631.05, 5964.83, -25095, 60380.2, -87517, 75513.8, -35749, 7154.94]
_DN_WLTP_KM = 800


def classe_inercia(massa_om_kg: float) -> float:
    massa_ensaio = massa_om_kg + MASSA_ENSAIO_EXTRA_KG
    return _FAIXAS_NBR_6601[bisect_right(_MINIMOS, massa_ensaio) - 1][1]


def combinar_mj(urbano, estrada):
    return 0.55 * urbano + 0.45 * estrada


def combinar_kml(urbano, estrada):
    return 1 / (0.55 / urbano + 0.45 / estrada)


def kml_eletrico_eq(mj_km):
    return FATOR_ELETRICO / mj_km


def fatores_kml(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Mediana de km/l × MJ/km (urbano, estrada) por combustível, nas linhas Combustão e Híbrido."""
    d = df[df["propulsao"].isin(["Combustão", "Híbrido"])]
    return {
        comb: tuple(float((d[f"kml_{comb}_{c}"] * d[f"mj_km_{c}"]).median()) for c in ("urbano", "estrada"))
        for comb in ("gasolina", "etanol", "diesel")
    }


def kml_diesel(mj_km, fator: float):
    return fator / mj_km


def fator_utilidade(autonomia_km):
    """Fração da distância percorrida em modo elétrico (WLTP); aceita float ou pd.Series."""
    x = np.clip(autonomia_km, 0, _DN_WLTP_KM) / _DN_WLTP_KM
    return 1 - np.exp(-sum(a * x**i for i, a in enumerate(_COEF_UF_WLTP, start=1)))


def fracao_combustao(propulsao: pd.Series, autonomia_km: pd.Series) -> pd.Series:
    """Fração da distância a combustão: 1 em Combustão e Híbrido, 0 em Elétrico, 1 − UF no Plug-in."""
    fracao = np.select(
        [propulsao.isin(["Combustão", "Híbrido"]), propulsao.eq("Plug-in"), propulsao.eq("Elétrico")],
        [1.0, 1 - fator_utilidade(autonomia_km), 0.0],
        default=np.nan,
    )
    return pd.Series(fracao, index=propulsao.index)


def combustiveis_aplicaveis(propulsao: str, combustivel: str) -> set[str]:
    if propulsao == "Elétrico":
        return {"eletrico_eq"}
    aplicaveis = {"flex": {"gasolina", "etanol"}, "gasolina": {"gasolina"}, "diesel": {"diesel"}}
    aplicaveis = set(aplicaveis.get(combustivel, set()))
    if propulsao == "Plug-in":
        aplicaveis.add("eletrico_eq")
    return aplicaveis
