import re
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
    assert pergunta(NOS[0]) == "Battery (kWh) < 12?"
    assert pergunta(NOS[1]) == "Powertrain is Electric?"
    no = {"coluna": "autonomia_eletrica_km", "categoria": None, "limite": 42.4}
    assert pergunta(no) == "Electric range (km) < 42?"


def test_html_arvore_anima_so_o_caminho():
    h = html_arvore(NOS)
    assert h.count('class="part"') == 2  # duas arestas no caminho
    assert "Powertrain is Electric?" in h and "32 cars" in h
    # NOS[1] é o Yes da pergunta numérica (sim), NOS[2] o No (não); NOS[4] é o No da pergunta de categoria (sim)
    assert h.count(">yes<") == 2 and h.count(">no<") == 1
    assert "prefers-reduced-motion" in h and "overflow-x:auto" in h


def test_html_rede_uma_particula_por_aresta():
    h = html_rede()
    assert h.count('class="part"') == 6 * 7 + 7 * 7 + 7 * 2
    assert "prefers-reduced-motion" in h


def test_filhos_de_no_fora_do_caminho_nao_se_sobrepoem():
    def no(id_, pai, nivel, caminho, sim):
        return {
            "id": id_,
            "pai": pai,
            "nivel": nivel,
            "folha": False,
            "coluna": "pbt_kg",
            "categoria": None,
            "limite": 2000.0 + id_,
            "valor": None,
            "carros": 10,
            "no_caminho": caminho,
            "sim": sim,
        }

    nos = [
        no(0, None, 0, True, True),
        no(1, 0, 1, True, True),
        no(2, 0, 1, False, None),
        no(5, 2, 2, False, None),
        no(6, 2, 2, False, None),
    ]
    h = html_arvore(nos)
    assert len(set(re.findall(r"left:([\d.]+)px;top:155.0px", h))) == 2
