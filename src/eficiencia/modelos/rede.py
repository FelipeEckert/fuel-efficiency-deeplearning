"""Redes neurais PyTorch (MLP): uma rede por alvo e uma rede com uma saída por alvo, perda MSE mascarada."""

from typing import ClassVar, Self

import numpy as np
import optuna
import pandas as pd
import torch
from torch import nn

from eficiencia.esquema import ALVOS
from eficiencia.modelos.base import Candidato, CandidatoPorAlvo
from eficiencia.preprocess import Preprocessador

PACIENCIA = 30  # épocas sem melhora na validação até parar
FRACAO_VALIDACAO = 0.15


def _sugerir_mlp(trial: optuna.Trial) -> dict:
    return {
        "camadas": trial.suggest_int("camadas", 1, 3),
        "largura": trial.suggest_categorical("largura", [16, 32, 64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
    }


def _perda(previsto: torch.Tensor, alvo: torch.Tensor, mascara: torch.Tensor) -> torch.Tensor:
    # alvo ausente (NaN) já foi trocado por 0; a máscara zera a contribuição dele na perda e no gradiente
    return ((previsto - alvo) ** 2 * mascara).sum() / mascara.sum().clamp(min=1)


class _Mlp:
    """MLP com saídas padronizadas por coluna (média/desvio das linhas não NaN) e parada antecipada."""

    def __init__(
        self,
        semente: int = 0,
        camadas: int = 2,
        largura: int = 64,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epocas: int = 500,
    ) -> None:
        self.semente = semente
        self.camadas = camadas
        self.largura = largura
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epocas = max_epocas

    def _construir(self, d_entrada: int, d_saida: int) -> nn.Sequential:
        blocos: list[nn.Module] = []
        d = d_entrada
        for _ in range(self.camadas):
            blocos += [nn.Linear(d, self.largura), nn.ReLU(), nn.Dropout(self.dropout)]
            d = self.largura
        blocos.append(nn.Linear(d, d_saida))
        return nn.Sequential(*blocos)

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

        self.rede = self._construir(X.shape[1], Y.shape[1])
        otimizador = torch.optim.Adam(self.rede.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        melhor, melhor_epoca, melhores_pesos = np.inf, 0, None
        for epoca in range(self.max_epocas):
            # lote completo (base pequena): um passo por época, mais rápido que minilotes na CPU
            self.rede.train()
            perda = _perda(self.rede(Xt), Yt, Mt)
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

    def __init__(self, semente: int, **params) -> None:
        self.mlp = _Mlp(semente, **params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        self.mlp.fit(X, y.reshape(-1, 1))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.mlp.predict(X)[:, 0]


class RedePorAlvo(CandidatoPorAlvo):
    nome = "rede_por_alvo"
    PARAMS_RAPIDOS: ClassVar[dict] = {"max_epocas": 5}
    sugerir = staticmethod(_sugerir_mlp)

    def _novo_estimador(self, alvo: str, nomes: list[str]):
        return _EstimadorMlp(self.semente, **self.params)


class RedeMulti(Candidato):
    nome = "rede_multi"
    PARAMS_RAPIDOS: ClassVar[dict] = {"max_epocas": 5}
    sugerir = staticmethod(_sugerir_mlp)

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> Self:
        self.prep = Preprocessador().fit(X)
        self.mlp = _Mlp(self.semente, **self.params).fit(self.prep.transform(X), np.log(Y[ALVOS].to_numpy()))
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = self.mlp.predict(self.prep.transform(X))
        return pd.DataFrame(np.exp(Z), columns=ALVOS, index=X.index)
