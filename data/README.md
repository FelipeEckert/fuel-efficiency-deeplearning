# Dados

```
data/
  raw/        planilhas originais (não editar) + complemento_baterias.csv
  processed/  pbev_2026_limpo.csv — gerado por scripts/limpar_pbev.py
```

`raw/complemento_baterias.csv`: capacidade de bateria pesquisada em fontes públicas (nov/2026)
para os veículos em que o PBEV não informa. Uma linha por marca + modelo + potência + massa,
com link da fonte, tipo de capacidade e confiança (alta / media / baixa / sem_fonte).
Só preenche lacunas — nunca sobrescreve o PBEV.

Para regenerar: `python scripts/limpar_pbev.py`

## Origem

Aba **"Dados de Entrada"** de `PED - PLANILHA DE ENTRADA DE DADOS PBEV 2026_05-05-2026.xlsm`.
São 844 versões que viram **510 homologações únicas** (uma linha por ensaio).

A planilha `ESPECIFICAÇÃO TÉCNICA - FEVEREIRO DE 2026.XLSM` não é usada por enquanto.

## Regras de limpeza

| Problema | O que foi feito |
|---|---|
| Propulsão escrita de vários jeitos ("Hibrido", "Plug-In", "Gasolina"…) | padronizada em Combustão / Híbrido / Plug-in / Elétrico |
| Marca com espaço sobrando ou caixa diferente; "VW", "CAOACHANGAN" | maiúsculas, sem espaços extras; VOLKSWAGEN, CAOA CHANGAN |
| Potência em 200+ formatos (kW, cv, "176 cv (G) e 176 cv (E)", "120 + 10", "150/110") | convertida para kW. Flex: maior valor entre G e E. Mercedes "A + B" = motor + elétrico. JAC "cv/kW". Número sem unidade: VW = cv, Mercedes = kW |
| F2 > 0,12 em 10 linhas (Oroch, Duster, MINI JCW, Ferrari 296) | declarado em N/(m/s)²; dividido por 3,6² |
| PRR80 errado (Duster) | coluna PRR80 removida: é derivada de F0/F2 |
| F1 sempre 0 | removida |
| Cilindrada "Elétrico", "N.A.", "1.798" ou vazia | elétrico = 0; "1.798" = 1798; vazia usa o nome do motor ("2.0-16V" = 2000, GAC GS4) |
| Honda informou 60 (Ah da bateria 12 V) como kWh | bateria = 0 para todo carro a combustão |
| Bateria faltando em 57 homologações | preenchida por `complemento_baterias.csv`; confiança baixa ou sem fonte → `bateria_estimada = True` (sem fonte usa a mediana da propulsão) |
| LCVM com espaço, quebra de linha, "N.A.", "Isento", "xxxx" ou número corrompido (Tank 300) | aparado; aceita `LC12233`, `19437`, `10945/24`, `SL64402`; o resto vira vazio (118 homologações sem LCVM) |
| "#N/A", "#VALUE!", "\", "-", "NA" | vazio |
| 3 linhas sem F0/F2/massa/MJ/km | removidas |
| Versões com dados idênticos (mesma homologação), inclusive entre marcas (vans Stellantis) | uma linha só; todas listadas em `versoes` |

## Colunas

**Identificação** (não entram no modelo)

| coluna | descrição |
|---|---|
| `lcvm`, `nome_homologado` | número da LCVM e nome homologado |
| `marca`, `modelo` | da primeira versão do grupo |
| `versoes` | todas as versões (marca + modelo + versão) que compartilham a homologação |
| `n_versoes` | quantas versões |
| `grupo_cv` | MARCA MODELO sem sufixos de variante (4X2/4X4, AT/MT, DIESEL, FLEX, 4M, (MOD. 26), acabamento); usar como grupo na validação cruzada (258 grupos). Carrocerias diferentes (Onix × Onix Plus) ficam separadas |

**Entradas**

| coluna | unidade / valores |
|---|---|
| `propulsao` | Combustão, Híbrido, Plug-in, Elétrico |
| `combustivel` | gasolina, flex, diesel, eletrico |
| `categoria` | categoria PBEV (SC, C, M, G, EXG, SUVC, SUVG, PIC, CL…) |
| `categoria_proconve` | PL, CL, CL1, CL2, CL3 |
| `massa_om_kg` | massa em ordem de marcha (kg) |
| `classe_inercia_kg` | classe de inércia do ensaio (kg) |
| `pbt_kg` | peso bruto total (kg) |
| `f0_n` | coeficiente F0 de pista (N) |
| `f2_n_kmh2` | coeficiente F2 de pista (N/(km/h)²) |
| `area_m2` | área declarada (m²); é comprimento × largura, não área frontal. **Não usada no modelo** (teste: erro 7,82% → 7,79% com ela) |
| `potencia_kw` | potência máxima (kW) |
| `cilindrada_cm3` | cm³; 0 em elétricos |
| `transmissao` | automatica, manual, cvt, dct, dht, direta |
| `n_marchas` | vazio em CVT/DHT |
| `direcao` | eletrica, hidraulica, eletro-hidraulica |
| `bateria_kwh` | capacidade nominal (kWh); 0 em combustão; sem lacunas |
| `bateria_estimada` | True quando a bateria veio de inferência ou mediana (17 híbridos) |

**Alvos** (valores corrigidos, os da etiqueta; vazio quando não se aplica)

| coluna | descrição |
|---|---|
| `mj_km` | consumo energético médio = 0,55 × urbano + 0,45 × estrada (todas as linhas) |
| `mj_km_urbano`, `mj_km_estrada` | |
| `kml_gasolina_urbano/estrada` | km/l com gasolina E22 |
| `kml_etanol_urbano/estrada` | km/l com etanol (só flex) |
| `kml_diesel_urbano/estrada` | km/l diesel |
| `kml_eletrico_eq_urbano/estrada` | km/l equivalente (BEV e PHEV); ≈ 20,29 / MJ/km (exato exceto no único PHEV flex: 19,95) |

## Limitações conhecidas

- Não há tração (4x2/4x4): o campo da planilha descreve o dinamômetro, não o veículo.
- "Híbrido" mistura mild hybrid (ex.: Mercedes "120 + 10") com híbrido completo.
- A regra de unidade da potência sem unidade vale para VW e Mercedes; marca nova nesse formato precisa de revisão.
- Baterias de mild hybrid vêm em boa parte de bases de dados (CarExpert, auto-data), não da montadora;
  os 0,9 kWh da Mercedes 48 V podem ser um valor genérico. A mediana dos híbridos (0,9) é puxada pelos
  mild hybrids, então o Lexus UX 300h (híbrido completo, sem fonte) ficou subestimado — marcado como estimado.
- O complemento é ligado por marca + modelo + potência + massa; se uma planilha PBEV nova mudar esses
  valores, a linha cai para a mediana (e fica marcada como estimada).
