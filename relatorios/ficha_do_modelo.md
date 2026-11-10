# Ficha do modelo: eficiência energética de veículos leves

Data: 2026-11-10 · modelo: XGBoost monotônico · base: PBEV 2026 (INMETRO), hash `ad3d96d0`

## O que faz

A partir da ficha técnica de um veículo, estima o consumo de energia em **MJ/km** (urbano, estrada e combinado) e o
**km/l** de cada combustível aplicável, sempre com uma **faixa de 90 %**. Serve para comparar versões e montar
cenários de portfólio antes de existir o carro. **Não substitui o ensaio de homologação.**

As entradas são a propulsão, o combustível, a categoria, a fase do PROCONVE, a transmissão e a direção. Também entram a
massa em ordem de marcha, a classe de inércia, o PBT, os coeficientes F0 e F2 de resistência ao rolamento, a potência, a
cilindrada e o número de marchas. Para elétricos e plug-ins entram ainda a bateria e a autonomia elétrica.

## Dados

São 510 versões do PBEV 2026: 288 a combustão, 109 elétricas, 63 híbridas e 50 plug-ins. Os valores "reais" são os
de homologação (ciclos urbano e estrada do INMETRO), não o consumo na rua.

## Quanto erra

A validação é cruzada e aninhada, por família de veículo: versões irmãs nunca ficam ao mesmo tempo no treino e no
teste.

- **Esquema:** 5 dobras externas e 3 internas.
- **Busca de hiperparâmetros:** 100 tentativas do Optuna por dobra.
- **Candidatos:** 5, comparados nas mesmas dobras.

O vencedor é o de menor erro entre os que respeitam a física: mais massa, mais F0 ou mais F2 nunca podem reduzir o
consumo.

| Candidato | Erro médio MJ/km | Combustão | Elétrico | Híbrido | Plug-in |
|---|---|---|---|---|---|
| **XGBoost (vencedor)** | **5,71 %** | 4,71 % | 4,57 % | 7,20 % | 12,07 % |
| Floresta aleatória | 6,55 % | 5,31 % | 4,64 % | 11,61 % | 11,46 % |
| Rede neural por alvo | 6,77 % | 5,51 % | 5,54 % | 7,32 % | 16,02 % |
| Rede neural multi | 6,95 % | 6,12 % | 5,71 % | 7,16 % | 14,21 % |
| Regressão linear | 7,46 % | 5,93 % | 7,38 % | 7,42 % | 16,52 % |

Os candidatos estão em ordem de erro médio. O modelo anterior errava 5,77 %. Detalhes em
[relatorio.md](relatorio.md); os experimentos que levaram às entradas atuais estão em
[experimentos/conclusoes.md](experimentos/conclusoes.md).

## A faixa de 90 % cumpre o que promete

A faixa vem das previsões fora da amostra: cada carro é previsto por um modelo que não o viu no treino. A cobertura é a
fração de carros cujo valor real ficou dentro da faixa mostrada pelo app.

| Propulsão | Cobertura MJ/km | Cobertura km/l | Meia-largura típica (MJ/km) |
|---|---|---|---|
| Combustão | 90,6 % | 93,5 % | ±10,5 % |
| Elétrico | 90,5 % | — | ±10,7 % |
| Híbrido | 91,5 % | 96,2 % | ±15,5 % |
| Plug-in | 92,0 % | 92,2 % | ±27,3 % |

Em combustão e híbridos, o km/l sai do MJ/km pelo fator físico de cada combustível, então os dois números nunca se
contradizem. Nos plug-ins essa relação varia muito de carro para carro, e o km/l vem de modelos próprios.

## Exemplo do simulador

Citroën Aircross flex, 1210 kg. É um carro da base, por isso a previsão bate quase exatamente com o real.

- **Previsão:** 1,80 MJ/km (faixa de 1,61 a 1,99); real de homologação: 1,80.
- **Gasolina:** 12,0 km/l (faixa de 10,8 a 13,6).
- **Etanol:** 8,3 km/l.

| Mudança | Efeito no MJ/km | Gasolina |
|---|---|---|
| Massa −10 % | −2,3 % | 12,0 → 12,3 km/l |
| F0 −10 % | −1,7 % | 12,0 → 12,2 km/l |
| F2 (aerodinâmica) −10 % | −5,0 % | 12,0 → 12,7 km/l |

O simulador tira a média das previsões em ±5 % em torno de cada ajuste. Sem isso, modelos de árvore respondem em
degraus.

## Onde não usar, ou usar com cuidado

- **Fora das faixas da base.** Se uma entrada passa do mínimo ou do máximo visto para aquela propulsão, o app mostra
  um aviso, e a previsão vira extrapolação.
- **Plug-ins.** O erro é de 12 % e a faixa tem cerca de ±27 %, porque são só 50 carros e o consumo depende muito do
  uso da bateria. Use para ordem de grandeza, não para decidir entre versões parecidas.
- **Carros de anos futuros.** Ainda não houve validação com um PBEV de outro ano. O teste temporal fica pendente até
  sair o PBEV 2027.
- **Consumo na rua.** O modelo aprende o consumo de homologação, que costuma ser menor que o real.
- **Portfólio.** A faixa da média ponderada supõe erros independentes entre os carros. Versões irmãs tendem a errar
  juntas, então a faixa real pode ser mais larga.

## Como reproduzir

```bash
uv run eficiencia-treinar --tentativas 100
```

Leva cerca de 21 h num i7-9750H. O processo acumula memória ao longo de milhares de tentativas, e na rodada de
2026-11-09 parou por falta de memória depois de ~11 h. Como o Optuna salva cada tentativa em `optuna/estudos.db`, rodar
o mesmo comando de novo retoma de onde parou.
