"""Number formatting for messages shown to the user."""


def numero(v: float, casas: int | None = None) -> str:
    """Shortest form of v with at most `casas` decimals (6 by default): 1.5, not 1.500000."""
    texto = f"{v:.{casas}f}" if casas is not None else f"{v:.6f}"
    if "." in texto:
        texto = texto.rstrip("0").rstrip(".")
    return texto
