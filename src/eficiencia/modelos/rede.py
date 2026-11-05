"""Redes neurais PyTorch monotônicas: uma rede por alvo e uma rede com uma saída por alvo, perda MSE mascarada."""

import itertools
import math
from typing import ClassVar, Self

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from eficiencia.esquema import ALVOS, MONOTONAS
from eficiencia.modelos.base import Candidato, CandidatoPorAlvo
from eficiencia.modelos.restricoes import sinal_alvo
from eficiencia.preprocess import Preprocessador

PACIENCIA = 30  # épocas sem melhora na validação até parar
FRACAO_VALIDACAO = 0.15


def _sugerir_mlp(trial: optuna.Trial) -> dict:
    return {
        "camadas": trial.suggest_int("camadas", 1, 5),
        "largura": trial.suggest_categorical("largura", [32, 64, 128, 256, 512]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "tamanho_lote": trial.suggest_categorical("tamanho_lote", [16, 32, 64]),
    }


def _perda(previsto: torch.Tensor, alvo: torch.Tensor, mascara: torch.Tensor) -> torch.Tensor:
    # alvo ausente (NaN) já foi trocado por 0; a máscara zera a contribuição dele na perda e no gradiente
    return ((previsto - alvo) ** 2 * mascara).sum() / mascara.sum().clamp(min=1)


def _mlp_livre(d_entrada: int, d_saida: int, camadas: int, largura: int, dropout: float) -> nn.Sequential:
    blocos: list[nn.Module] = []
    d = d_entrada
    for _ in range(camadas):
        blocos += [nn.Linear(d, largura), nn.ReLU(), nn.Dropout(dropout)]
        d = largura
    blocos.append(nn.Linear(d, d_saida))
    return nn.Sequential(*blocos)


class _MlpPositivo(nn.Module):
    """Peso efetivo softplus(w) >= 0 e viés livre: não decrescente em cada entrada.

    Entre camadas, metade dos neurônios usa ReLU (convexa) e metade -ReLU(-x) (côncava). Com só ReLU,
    a rede seria convexa nas entradas monótonas; a mistura permite curvas que crescem cada vez menos.
    """

    def __init__(self, d_entrada: int, d_saida: int, camadas: int, largura: int) -> None:
        super().__init__()
        dims = [d_entrada] + [largura] * camadas + [d_saida]
        # softplus(w) ~ 1/fan_in no início: sem isso as ativações crescem ~width por camada
        self.pesos = nn.ParameterList(
            nn.Parameter(math.log(math.expm1(1 / din)) + 0.1 * torch.randn(dout, din))
            for din, dout in itertools.pairwise(dims)
        )
        self.vies = nn.ParameterList(nn.Parameter(torch.zeros(dout)) for dout in dims[1:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ultima = len(self.pesos) - 1
        for i, (w, b) in enumerate(zip(self.pesos, self.vies, strict=True)):
            x = x @ F.softplus(w).T + b
            if i < ultima:
                metade = x.shape[1] // 2
                x = torch.cat([torch.relu(x[:, :metade]), -torch.relu(-x[:, metade:])], dim=1)
        return x


class _RedeMonotona(nn.Module):
    """y_j = g_j(x_livre) + sinal_j * sum_k softplus(s_jk(x_livre)) * m_k(x_mono), com g, s livres e m positivo."""

    def __init__(self, nomes: list[str], sinais: list[int], camadas: int, largura: int, dropout: float):
        super().__init__()
        self.mono = torch.as_tensor([i for i, n in enumerate(nomes) if n in MONOTONAS], dtype=torch.long)
        self.livre = torch.as_tensor([i for i, n in enumerate(nomes) if n not in MONOTONAS], dtype=torch.long)
        self.sinais = torch.as_tensor(sinais, dtype=torch.float32)
        self.k = len(sinais)
        self.m = _MlpPositivo(len(self.mono), self.k, camadas, largura)
        self.g = _mlp_livre(len(self.livre), self.k, camadas, largura, dropout)
        self.s = _mlp_livre(len(self.livre), self.k * self.k, camadas, largura, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xm, xl = x[:, self.mono], x[:, self.livre]
        pesos = F.softplus(self.s(xl)).view(-1, self.k, self.k)  # (n, j, k) >= 0
        misturado = torch.einsum("njk,nk->nj", pesos, self.m(xm))
        return self.g(xl) + self.sinais * misturado


class _Mlp:
    """Rede monotônica com saídas padronizadas por coluna (média/desvio das linhas não NaN) e parada antecipada."""

    def __init__(
        self,
        semente: int = 0,
        nomes: list[str] | None = None,
        sinais: list[int] | None = None,
        camadas: int = 2,
        largura: int = 64,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epocas: int = 500,
        tamanho_lote: int = 32,
    ) -> None:
        self.semente = semente
        self.nomes = nomes or []
        self.sinais = sinais or []
        self.camadas = camadas
        self.largura = largura
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epocas = max_epocas
        self.tamanho_lote = tamanho_lote

    def _construir(self) -> _RedeMonotona:
        return _RedeMonotona(self.nomes, self.sinais, self.camadas, self.largura, self.dropout)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> Self:
        """X: float32 (n, d); Y: float (n, k) com NaN onde o alvo não se aplica."""
        torch.manual_seed(self.semente)
        rng = np.random.default_rng(self.semente)
        self.media = np.nanmean(Y, axis=0)
        desvio = np.nanstd(Y, axis=0)
        self.desvio = np.where(desvio > 0, desvio, 1.0)  # coluna constante não divide por zero
        presente = ~np.isnan(Y)
        Ys = np.where(presente, (Y - self.media) / self.desvio, 0.0).astype(np.float32)

        ordem = rng.permutation(len(X))
        n_val = max(1, round(FRACAO_VALIDACAO * len(X)))
        val, tr = ordem[:n_val], ordem[n_val:]
        Xt, Yt, Mt = (torch.as_tensor(a[tr], dtype=torch.float32) for a in (X, Ys, presente))
        Xv, Yv, Mv = (torch.as_tensor(a[val], dtype=torch.float32) for a in (X, Ys, presente))

        self.rede = self._construir()
        otimizador = torch.optim.Adam(self.rede.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # pesos iniciais como salvaguarda: restaurados se a validação nunca for finita e menor
        melhor, melhor_epoca = np.inf, 0
        melhores_pesos = {k: v.clone() for k, v in self.rede.state_dict().items()}
        for epoca in range(self.max_epocas):
            self.rede.train()
            perm = torch.as_tensor(rng.permutation(len(tr)))
            for i in range(0, len(tr), self.tamanho_lote):
                idx = perm[i : i + self.tamanho_lote]
                perda = _perda(self.rede(Xt[idx]), Yt[idx], Mt[idx])
                otimizador.zero_grad()
                perda.backward()
                otimizador.step()
            self.rede.eval()
            with torch.no_grad():
                validacao = _perda(self.rede(Xv), Yv, Mv).item()
            if validacao < melhor:
                melhor, melhor_epoca = validacao, epoca
                melhores_pesos = {k: v.clone() for k, v in self.rede.state_dict().items()}
            elif epoca - melhor_epoca >= PACIENCIA:
                break
        self.rede.load_state_dict(melhores_pesos)
        self.rede.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Previsões nas unidades de Y (desfeita a padronização)."""
        with torch.no_grad():
            z = self.rede(torch.as_tensor(X, dtype=torch.float32)).numpy()
        return z * self.desvio + self.media


class _EstimadorMlp:
    """Adaptador numpy de um alvo para CandidatoPorAlvo: fit(X, y) e predict(X)."""

    def __init__(self, semente: int, nomes: list[str], alvo: str, **params) -> None:
        self.mlp = _Mlp(semente, nomes=nomes, sinais=[sinal_alvo(alvo)], **params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        self.mlp.fit(X, y.reshape(-1, 1))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.mlp.predict(X)[:, 0]


class RedePorAlvo(CandidatoPorAlvo):
    VERSAO: ClassVar[int] = 2  # 2: viés e ativação côncava na parte monótona
    nome = "rede_por_alvo"
    PARAMS_RAPIDOS: ClassVar[dict] = {"max_epocas": 5}
    sugerir = staticmethod(_sugerir_mlp)

    def _novo_estimador(self, alvo: str, nomes: list[str]):
        return _EstimadorMlp(self.semente, nomes, alvo, **self.params)


class RedeMulti(Candidato):
    VERSAO: ClassVar[int] = 2  # 2: viés e ativação côncava na parte monótona
    nome = "rede_multi"
    PARAMS_RAPIDOS: ClassVar[dict] = {"max_epocas": 5}
    sugerir = staticmethod(_sugerir_mlp)

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> Self:
        self.prep = Preprocessador().fit(X)
        sinais = [sinal_alvo(alvo) for alvo in ALVOS]
        mlp = _Mlp(self.semente, nomes=self.prep.nomes, sinais=sinais, **self.params)
        self.mlp = mlp.fit(self.prep.transform(X), np.log(Y[ALVOS].to_numpy()))
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = self.mlp.predict(self.prep.transform(X))
        return pd.DataFrame(np.exp(Z), columns=ALVOS, index=X.index)
