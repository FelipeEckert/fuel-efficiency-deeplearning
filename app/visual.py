"""Animated (CSS) HTML of the real XGBoost tree and of the neural network sketch, for st.iframe. Captions live on the page."""

import math
from html import escape

from eficiencia.esquema import ROTULOS, nome_valor
from eficiencia.formato import numero
from eficiencia.modelos import NOMES_MODELOS

CORES = {
    "primaria": "#7C3AED",
    "ciano": "#0891B2",
    "tinta": "#1E1B4B",
    "fundo": "#FAF5FF",
    "borda": "#DDD6FE",
    "escuro": "#1E1B4B",
    "destaque": "#22D3EE",
    "lilas": "#A78BFA",
}

NOMES = NOMES_MODELOS  # display name of each candidate

_CASAS = {
    "f0_n": 1,
    "bateria_kwh": 1,
    "f2_n_kmh2": 4,
}  # as demais colunas numéricas (kg, kW, cm³...) usam 0 casas

# Keyframes e classes do protótipo V6. Só transform/opacity; o .part sai do HTML como class="part".
_CSS = """
body{margin:0;padding:12px;background:#1E1B4B;color:#fff;font-family:Inter,system-ui,sans-serif}
@keyframes viaja{0%{transform:translateX(0);opacity:0}3%{opacity:1}15%{transform:translateX(var(--L));opacity:1}
17%,100%{transform:translateX(var(--L));opacity:0}}
.part{position:absolute;left:-6px;top:50%;width:12px;height:12px;margin-top:-6px;border-radius:50%;
background:radial-gradient(circle,#FFFFFF 0 28%,#22D3EE 55%,rgba(34,211,238,0) 72%);
animation:viaja var(--T) cubic-bezier(.45,.05,.25,1) infinite;animation-delay:var(--D);will-change:transform,opacity}
@keyframes anel{0%{transform:scale(.9);opacity:0}4%{opacity:.95}22%{transform:scale(1.3);opacity:0}
100%{transform:scale(1.3);opacity:0}}
.anel{position:absolute;left:-5px;right:-5px;top:-5px;bottom:-5px;border-radius:10px;border:2px solid #22D3EE;
opacity:0;animation:anel var(--T) cubic-bezier(.22,1,.36,1) infinite;animation-delay:var(--D);pointer-events:none}
@keyframes neur{0%{transform:scale(1)}6%{transform:scale(1.45)}20%,100%{transform:scale(1)}}
.neur{animation:neur var(--T) cubic-bezier(.34,1.56,.64,1) infinite;animation-delay:var(--D)}
.pausado .part,.pausado .anel,.pausado .neur{animation-play-state:paused}
button{font:inherit;font-size:14px;font-weight:600;border:0;border-radius:9px;padding:10px 14px;min-height:44px;
cursor:pointer;background:#fff;color:#1E1B4B;margin-bottom:10px}
@media (prefers-reduced-motion: reduce){.part,.anel{animation:none;opacity:0}.neur{animation:none}}
"""


def pergunta(no: dict) -> str:
    """Texto da pergunta de um nó interno da árvore."""
    rotulo = ROTULOS[no["coluna"]]
    if no["categoria"] is not None:
        return f"{rotulo} is {nome_valor(no['coluna'], no['categoria'])}?"
    return f"{rotulo} < {numero(no['limite'], _CASAS.get(no['coluna'], 0))}?"


def _linha(x1, y1, x2, y2, cor="#3730A3", esp=1.5, filho="", vars_=""):
    """Aresta como um div girado; devolve o HTML (o `filho` anda ao longo dela)."""
    comp = math.hypot(x2 - x1, y2 - y1)
    ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
    estilo = (
        f"position:absolute;left:{x1:.1f}px;top:{y1:.1f}px;width:{comp:.1f}px;height:{esp}px;background:{cor};"
        f"border-radius:2px;transform-origin:0 50%;transform:rotate({ang:.2f}deg);--L:{comp:.1f}px;{vars_}"
    )
    return f'<div style="{estilo}">{filho}</div>'


def _documento(rotulo: str, largura: int, altura: int, miolo: str) -> str:
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><style>{_CSS}</style></head><body>'
        "<button onclick=\"document.body.classList.toggle('pausado')\">Pause animation</button>"
        f'<div style="overflow-x:auto" role="img" aria-label="{escape(rotulo)}">'
        f'<div style="position:relative;width:{largura}px;height:{altura}px">{miolo}</div></div>'
        "</body></html>"
    )


def html_arvore(nos: list[dict]) -> str:
    """Árvore 1000 × 400: níveis 0-2 completos, mais o caminho do veículo com o sinal percorrendo as arestas."""
    por_id = {n["id"]: n for n in nos}
    ult = max(n["nivel"] for n in nos)
    passo_fundo = min(70, 132 / max(1, ult - 3))  # níveis fundos (só o caminho) cabem até y = 372
    yes, pos, col = (
        {},
        {},
        {},
    )  # yes: o nó é o filho "Yes" do XGBoost?  # centro (x, y) e posição binária no nível: 2 * posição do pai + 0 (sim) ou 1 (não)
    for n in nos:
        if n["nivel"] > 2:  # desce em coluna, no x do pai
            pos[n["id"]] = (pos[n["pai"]][0], 170 + (n["nivel"] - 2) * passo_fundo)
            yes[n["id"]] = por_id[n["pai"]]["sim"]  # nível fundo: só o filho do caminho
            continue
        if n["nivel"] == 0:
            col[n["id"]] = 0
        else:
            pai = por_id[n["pai"]]
            if pai["sim"] is None:  # pai fora do caminho: no XGBoost o filho "Yes" tem o menor id
                vai_sim = n["id"] == min(f["id"] for f in nos if f["pai"] == pai["id"])
            else:
                vai_sim = pai["sim"] == n["no_caminho"]
            yes[n["id"]] = vai_sim
            col[n["id"]] = 2 * col[pai["id"]] + (0 if vai_sim else 1)
        pos[n["id"]] = ((col[n["id"]] + 0.5) / 2 ** n["nivel"] * 1000, 30 + 70 * n["nivel"])

    ordem = {n["id"]: k for k, n in enumerate(n for n in nos if n["no_caminho"])}  # k = passo do sinal
    arestas, caixas, rotulos_aresta = [], [], []
    for n in nos:
        x, y = pos[n["id"]]
        k = ordem.get(n["id"])
        if n["pai"] is not None:
            px, py = pos[n["pai"]]
            # categoria: o Yes do XGBoost é x < 1, ou seja, "não é a categoria"
            sim_ou_nao = "yes" if yes[n["id"]] != (por_id[n["pai"]]["categoria"] is not None) else "no"
            rotulos_aresta.append(
                f'<div style="position:absolute;left:{(px + x) / 2 - 15:.1f}px;top:{(py + y) / 2 - 8:.1f}px;'
                f'width:30px;text-align:center;font-size:10px;color:#C7D2FE">{sim_ou_nao}</div>'
            )
            if k is None:
                arestas.append(_linha(px, py + 15, x, y - 15))
            else:
                vars_ = f"--T:4.8s;--D:{(k - 1) * 0.8:.2f}s;"
                part = f'<div class="part" style="{vars_}"></div>'
                arestas.append(_linha(px, py + 15, x, y - 15, "rgba(34,211,238,.55)", 3, part, vars_))
        if n["folha"]:
            sinal = "+" if n["valor"] >= 0 else "−"
            texto = f"leaf · {n['carros']} cars · {sinal}{numero(abs(n['valor']), 4)}"
            fundo, larg = "background:#A78BFA;color:#1E1B4B;font-weight:700;border-radius:999px;", 210
        else:
            texto, larg = pergunta(n), 150
            fundo = (
                "background:#0E7490;color:#FFFFFF;font-weight:600;border-radius:8px;"
                if k is not None
                else "background:#312E81;color:#C7D2FE;border-radius:8px;"  # fora do caminho: apagado
            )
        anel = ""
        if k is not None:
            anel = f'<div class="anel" style="--T:4.8s;--D:{max(k * 0.8 - 0.12, 0):.2f}s"></div>'
        caixas.append(
            f'<div style="position:absolute;left:{x - larg / 2:.1f}px;top:{y - 15:.1f}px;width:{larg}px;'
            f'box-sizing:border-box;text-align:center;font-size:12px;padding:7px 4px;{fundo}">'
            f"{escape(texto)}{anel}</div>"
        )
    return _documento("Vehicle path through the tree", 1000, 400, "".join(arestas + caixas + rotulos_aresta))


def html_rede() -> str:
    """Esquema 6-7-7-2 da rede neural (ilustrativo): uma onda de sinal camada por camada."""
    camadas, xs, onda = [6, 7, 7, 2], [120, 380, 640, 880], 0.55
    pts = [[(xs[k], 30 + (i + 0.5) * 270 / n) for i in range(n)] for k, n in enumerate(camadas)]
    arestas = []
    for k in range(3):
        for i, (ax, ay) in enumerate(pts[k]):
            for j, (bx, by) in enumerate(pts[k + 1]):
                vars_ = f"--T:2.6s;--D:{k * onda + ((i * 3 + j) % 5) * 0.04:.2f}s;"
                part = f'<div class="part" style="{vars_}width:8px;height:8px;margin-top:-4px"></div>'
                arestas.append(_linha(ax, ay, bx, by, "rgba(165,180,252,.22)", 1, part, vars_))
    cor = ["#22D3EE", "#E0E7FF", "#E0E7FF", "#A78BFA"]
    neuronios = [
        f'<div class="neur" style="position:absolute;left:{x - 10:.1f}px;top:{y - 10:.1f}px;width:20px;'
        f'height:20px;border-radius:50%;--T:2.6s;--D:{k * onda + 0.3:.2f}s;background:{cor[k]}"></div>'
        for k, camada in enumerate(pts)
        for x, y in camada
    ]
    rotulos = [
        f'<div style="position:absolute;left:{x - 80}px;top:310px;width:160px;text-align:center;'
        f'font-size:12px;color:#C7D2FE">{t}</div>'
        for x, t in zip(xs, ["48 inputs", "hidden 1", "hidden 2", "city · highway"])
    ]
    return _documento("Neural network sketch", 1000, 340, "".join(arestas + neuronios + rotulos))
