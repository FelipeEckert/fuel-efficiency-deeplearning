from eficiencia.faixa import avisos
from eficiencia.formato import numero


def test_avisos_texto():
    faixas = {
        "numericas": {"Elétrico": {"f2_n_kmh2": (0.029, 0.08)}},
        "categoricas": {"Elétrico": {"categoria": {"SUVC"}}},
    }
    v = {"propulsao": "Elétrico", "f2_n_kmh2": 0.025, "categoria": "PIC"}
    assert avisos(v, faixas) == [
        "F2 = 0,025 abaixo de qualquer Elétrico da base (mínimo 0,029)",
        "Categoria = PIC nunca vista em Elétrico na base",
    ]
    assert avisos({"propulsao": "Elétrico", "f2_n_kmh2": 0.09, "categoria": "SUVC"}, faixas) == [
        "F2 = 0,09 acima de qualquer Elétrico da base (máximo 0,08)"
    ]


def test_numero_formato():
    assert numero(0.025) == "0,025"
    assert numero(1644.0) == "1644"
    assert numero(1.5, 2) == "1,5"
