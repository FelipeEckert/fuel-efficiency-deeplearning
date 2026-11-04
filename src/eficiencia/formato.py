"""Formatação de números em pt-BR para as mensagens ao usuário."""


def numero(v: float, casas: int | None = None) -> str:
    texto = f"{v:.{casas}f}" if casas is not None else f"{v:.6f}"
    if "." in texto:
        texto = texto.rstrip("0").rstrip(".")
    return texto.replace(".", ",")
