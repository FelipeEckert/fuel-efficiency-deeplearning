import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from visual import html_arvore, html_rede, pergunta

NOS = [
    {
        "id": 0,
        "pai": None,
        "nivel": 0,
        "folha": False,
        "coluna": "bateria_kwh",
        "categoria": None,
        "limite": 12.0,
        "valor": None,
        "carros": 320,
        "no_caminho": True,
        "sim": True,
    },
    {
        "id": 1,
        "pai": 0,
        "nivel": 1,
        "folha": False,
        "coluna": "propulsao",
        "categoria": "Elétrico",
        "limite": None,
        "valor": None,
        "carros": 219,
        "no_caminho": True,
        "sim": False,
    },
    {
        "id": 2,
        "pai": 0,
        "nivel": 1,
        "folha": True,
        "coluna": None,
        "categoria": None,
        "limite": None,
        "valor": 0.01,
        "carros": 101,
        "no_caminho": False,
        "sim": None,
    },
    {
        "id": 4,
        "pai": 1,
        "nivel": 2,
        "folha": True,
        "coluna": None,
        "categoria": None,
        "limite": None,
        "valor": 0.0022,
        "carros": 32,
        "no_caminho": True,
        "sim": None,
    },
]


def test_pergunta_legivel():
    assert pergunta(NOS[0]) == "Bateria (kWh) < 12?"
    assert pergunta(NOS[1]) == "Propulsão é Elétrico?"


def test_html_arvore_anima_so_o_caminho():
    h = html_arvore(NOS)
    assert h.count('class="part"') == 2  # duas arestas no caminho
    assert "Propulsão é Elétrico?" in h and "32 carros" in h
    assert "prefers-reduced-motion" in h and "overflow-x:auto" in h


def test_html_rede_uma_particula_por_aresta():
    h = html_rede()
    assert h.count('class="part"') == 6 * 7 + 7 * 7 + 7 * 2
    assert "esquema" in h.lower() and "prefers-reduced-motion" in h
