"""Model columns: inputs, targets, category values, and English labels for the interface."""

import os
from pathlib import Path

import pandas as pd

from eficiencia.dados import SAIDA

CAMINHO_BASE: Path = SAIDA

CATEGORICAS = ["propulsao", "combustivel", "categoria", "categoria_proconve", "transmissao", "direcao"]
NUMERICAS = [
    "massa_om_kg",
    "classe_inercia_kg",
    "pbt_kg",
    "f0_n",
    "f2_n_kmh2",
    "potencia_kw",
    "cilindrada_cm3",
    "n_marchas",
    "bateria_kwh",
    "autonomia_eletrica_km",
]
ENTRADAS = CATEGORICAS + NUMERICAS
DERIVADAS = ["fracao_combustao"]  # calculadas a partir de outras entradas, em preprocess
ALVOS = [
    "mj_km_urbano",
    "mj_km_estrada",
    "kml_gasolina_urbano",
    "kml_gasolina_estrada",
    "kml_etanol_urbano",
    "kml_etanol_estrada",
]
MONOTONAS = ["massa_om_kg", "classe_inercia_kg", "f0_n", "f2_n_kmh2", "fracao_combustao"]

VALORES: dict[str, list[str]] = {
    "propulsao": ["Combustão", "Elétrico", "Híbrido", "Plug-in"],
    "combustivel": ["gasolina", "flex", "eletrico", "diesel"],
    "categoria": [
        "EXG",
        "SUVC",
        "M",
        "PIC",
        "G",
        "E",
        "SUVG",
        "CL",
        "SUVG4",
        "FEG",
        "C",
        "PC",
        "SC",
        "MIN",
        "FEC",
        "SUVC4",
    ],
    "categoria_proconve": ["PL", "CL", "CL1", "CL2", "CL3"],
    "transmissao": ["automatica", "manual", "cvt", "dct", "direta", "dht"],
    "direcao": ["eletrica", "hidraulica", "eletro-hidraulica"],
}

ROTULOS: dict[str, str] = {
    "propulsao": "Powertrain",
    "combustivel": "Fuel",
    "categoria": "Category",
    "categoria_proconve": "PROCONVE category",
    "transmissao": "Transmission",
    "direcao": "Steering",
    "massa_om_kg": "Curb mass (kg)",
    "classe_inercia_kg": "Inertia class (kg)",
    "pbt_kg": "Gross vehicle weight (kg)",
    "f0_n": "F0",
    "f2_n_kmh2": "F2",
    "potencia_kw": "Power (kW)",
    "cilindrada_cm3": "Displacement (cm³)",
    "n_marchas": "Number of gears",
    "bateria_kwh": "Battery (kWh)",
    "autonomia_eletrica_km": "Electric range (km)",
    "fracao_combustao": "Share of distance on fuel",
    "mj_km_urbano": "MJ/km city",
    "mj_km_estrada": "MJ/km highway",
    "kml_gasolina_urbano": "km/l gasoline city",
    "kml_gasolina_estrada": "km/l gasoline highway",
    "kml_etanol_urbano": "km/l ethanol city",
    "kml_etanol_estrada": "km/l ethanol highway",
}

# English display names for the category codes; the data keeps the original PBEV codes
NOMES_VALORES: dict[str, dict[str, str]] = {
    "propulsao": {
        "Combustão": "Combustion",
        "Elétrico": "Electric",
        "Híbrido": "Hybrid",
        "Plug-in": "Plug-in hybrid",
    },
    "combustivel": {
        "gasolina": "Gasoline",
        "flex": "Flex (gasoline/ethanol)",
        "diesel": "Diesel",
        "eletrico": "Electric",
    },
    "transmissao": {
        "automatica": "Automatic",
        "manual": "Manual",
        "cvt": "CVT",
        "dct": "Dual clutch (DCT)",
        "dht": "Dedicated hybrid (DHT)",
        "direta": "Direct drive (1 gear)",
    },
    "direcao": {"eletrica": "Electric", "hidraulica": "Hydraulic", "eletro-hidraulica": "Electro-hydraulic"},
}


def nome_valor(coluna: str, codigo) -> str:
    """English name of a category code, or the code itself when it has none."""
    return NOMES_VALORES.get(coluna, {}).get(codigo, str(codigo))


def caminho_base() -> Path:
    """Clean table path; EFICIENCIA_BASE points elsewhere (the tests use a synthetic table)."""
    return Path(os.environ.get("EFICIENCIA_BASE", CAMINHO_BASE))


def carregar_base(caminho: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(caminho or caminho_base(), dtype={"lcvm": str})
