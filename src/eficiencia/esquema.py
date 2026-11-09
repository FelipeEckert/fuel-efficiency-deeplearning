"""Colunas do modelo: entradas, alvos, valores categóricos e rótulos em pt-BR."""

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
ALVOS = [
    "mj_km_urbano",
    "mj_km_estrada",
    "kml_gasolina_urbano",
    "kml_gasolina_estrada",
    "kml_etanol_urbano",
    "kml_etanol_estrada",
]
MONOTONAS = ["massa_om_kg", "classe_inercia_kg", "f0_n", "f2_n_kmh2"]

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
    "propulsao": "Propulsão",
    "combustivel": "Combustível",
    "categoria": "Categoria",
    "categoria_proconve": "Categoria PROCONVE",
    "transmissao": "Transmissão",
    "direcao": "Direção",
    "massa_om_kg": "Massa em ordem de marcha (kg)",
    "classe_inercia_kg": "Classe de inércia (kg)",
    "pbt_kg": "PBT (kg)",
    "f0_n": "F0",
    "f2_n_kmh2": "F2",
    "potencia_kw": "Potência (kW)",
    "cilindrada_cm3": "Cilindrada (cm³)",
    "n_marchas": "Número de marchas",
    "bateria_kwh": "Bateria (kWh)",
    "autonomia_eletrica_km": "Autonomia elétrica (km)",
    "mj_km_urbano": "MJ/km urbano",
    "mj_km_estrada": "MJ/km estrada",
    "kml_gasolina_urbano": "km/l gasolina urbano",
    "kml_gasolina_estrada": "km/l gasolina estrada",
    "kml_etanol_urbano": "km/l etanol urbano",
    "kml_etanol_estrada": "km/l etanol estrada",
}


def carregar_base(caminho: Path = CAMINHO_BASE) -> pd.DataFrame:
    return pd.read_csv(caminho, dtype={"lcvm": str})
