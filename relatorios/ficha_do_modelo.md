# Model card: light-vehicle energy efficiency

Date: 2026-11-26 · model: monotonic XGBoost · data: PBEV 2026 (Inmetro), hash `7f9a2a41`

## What it does

From a vehicle's technical specifications, it estimates energy consumption in **MJ/km** (city, highway and combined)
and the **km/l** for each applicable fuel, always with a **90 % range**. It is meant for comparing versions and
building portfolio scenarios before the car exists. **It does not replace the homologation test.**

The inputs are powertrain, hybrid type (full, 48 V mild or 12 V mild), fuel, category, PROCONVE phase, transmission
and steering. The model also uses curb mass,
inertia class, gross vehicle weight, the road-load coefficients F0 and F2, power, displacement and number of gears.
Electric and plug-in vehicles add battery capacity and electric range.

## Data

The data has 510 versions from PBEV 2026: 286 combustion, 109 electric, 65 hybrid and 50 plug-in hybrids. The "true"
values are the homologation results (Inmetro city and highway cycles), not on-road consumption.

## How wrong it is

Validation is nested cross-validation grouped by vehicle family. Sibling versions never appear in both training and
test.

- **Folds:** 5 outer and 3 inner.
- **Hyperparameter search:** 100 Optuna trials per fold.
- **Candidates:** 5, compared on the same folds.

The selected model is the one with the lowest error among those that respect the physics: more mass, F0 or F2 can
never lower consumption.

| Candidate | Mean MJ/km error | Combustion | Electric | Hybrid | Plug-in |
|---|---|---|---|---|---|
| **XGBoost (selected)** | **5.40 %** | 4.04 % | 4.43 % | 7.49 % | 12.65 % |
| Random forest | 6.36 % | 4.97 % | 4.68 % | 11.51 % | 11.30 % |
| Neural network, one per target | 6.62 % | 5.03 % | 5.32 % | 6.80 % | 18.29 % |
| Neural network, multi-output | 6.75 % | 5.48 % | 5.66 % | 6.77 % | 16.34 % |
| Linear regression | 6.92 % | 5.36 % | 7.33 % | 6.01 % | 16.12 % |

The candidates are ordered by mean error. The previous model had 5.64 %; the gain comes from the new hybrid type
input, which separates full hybrids from mild ones (see H12 in the experiments). Before that, fixing the label of two
Toyota RAV4 Hybrid rows that the spreadsheet entered as combustion took the error from 5.71 % to 5.64 %. Details are in [relatorio.md](relatorio.md),
and the experiments behind the current inputs are in
[experimentos/conclusoes.md](experimentos/conclusoes.md).

## The 90 % range

The range comes from out-of-sample predictions: each car is predicted by a model that did not see it in training.
Coverage is the share of cars whose true value fell inside the range the app shows.

**Coverage on families the range never saw.** For each of the 5 outer folds, the range is rebuilt from the errors of
the other 4 folds only, then checked on the held-out fold. The held-out fold holds whole model families, so this is
the coverage a new family gets. The km/l column goes through the same code as the app.

| Powertrain | Cars | MJ/km coverage | 90 % interval | Gasoline km/l coverage |
|---|---|---|---|---|
| Combustion | 286 | 89.2 % | 85.8–91.8 % | 92.4 % |
| Electric | 109 | 91.7 % | 86.3–95.1 % | — |
| Hybrid | 65 | 87.7 % | 79.4–92.9 % | 93.1 % |
| Plug-in | 50 | 92.0 % | 83.3–96.4 % | 94.0 % |
| **All** | **510** | **89.8 %** | **87.4–91.8 %** | **92.7 %** |

- **Per fold:** MJ/km coverage goes from 85.7 % to 92.4 %.
- **Ethanol and diesel:** km/l coverage is 96.4 % and 98.2 %, so the range there is wider than it needs to be.
- **Still open:** all of this is one model year. Cars from the next PBEV may be farther from the data.

**Half-widths the app uses**, read from the shipped model package:

| Powertrain | Combined | City | Highway |
|---|---|---|---|
| Combustion | ±8.8 % | ±9.8 % | ±9.6 % |
| Electric | ±9.7 % | ±12.5 % | ±8.3 % |
| Hybrid | ±18.6 % | ±22.6 % | ±17.8 % |
| Plug-in | ±25.3 % | ±33.0 % | ±28.6 % |

For combustion and hybrids, km/l is derived from MJ/km with each fuel's physical factor, so the two numbers never
contradict each other. For plug-ins that ratio varies a lot from car to car, so km/l comes from dedicated models.

## Simulator example

Citroën Aircross flex, 1210 kg. This car is in the data, which is why the prediction matches the true value almost
exactly.

- **Prediction:** 1.80 MJ/km (range 1.64 to 1.96); homologation value: 1.80.
- **Gasoline:** 12.0 km/l (range 11.0 to 13.3).
- **Ethanol:** 8.4 km/l.

| Change | Effect on MJ/km | Gasoline |
|---|---|---|
| Mass −10 % | −2.1 % | 12.0 → 12.3 km/l |
| F0 −10 % | −1.3 % | 12.0 → 12.2 km/l |
| F2 (aerodynamics) −10 % | −4.7 % | 12.0 → 12.6 km/l |

The simulator averages the predictions within ±5 % around each adjustment. Without that, tree models respond in steps. The scenario export marks these rows as smoothed.

## Where not to use it, or use it with care

- **Outside the data ranges.** If an input goes below the minimum or above the maximum seen for that powertrain, the
  app shows a warning, and the prediction becomes an extrapolation.
- **Plug-in hybrids.** The error is 12.6 % and the combined range is about ±25 %, because there are only 50 cars and consumption
  depends heavily on how the battery is used. Use it for orders of magnitude, not to choose between similar versions.
- **Before the electric range is known.** The electric range in the data is the one measured in the homologation test.
  Leaving it out raises the error from 4.5 % to 5.1 % for electric cars and from 12.9 % to 15.0 % for plug-ins. A design
  target 15 % above the real range barely matters (4.6 % and 12.4 %), so a reasonable target is better than none.
- **Hybrids whose battery was estimated.** The 19 hybrids with no battery capacity in the spreadsheet err 9.7 %,
  against 6.6 % for the other hybrids.
- **Full hybrids.** In the paired test the hybrid type took their error from 12.3 % to 9.5 %. In the shipped model it
  is 9.8 %, still the highest of the three types (48 V mild 7.2 %, 12 V mild 2.5 %). A new hybrid brand needs its type added to
  `TIPO_HIBRIDO_MARCA` in `eficiencia.dados`, or the data check stops.
- **Hybrid km/l.** The km/l the app shows for gasoline hybrids comes from MJ/km, so the two always agree. That costs
  about 1.3 points: 7.4 % against 6.1 % for the model's own km/l output.
- **Plug-in diesel.** There is no plug-in diesel in the data. Its km/l comes from MJ/km with the diesel factor of
  combustion and hybrid cars, not from a dedicated model. For gasoline plug-ins that factor would overstate km/l
  about 2.5–3 times (median km/l × MJ/km of 8 for plug-ins against 21 to 22 for the others), so the app shows it
  without a range and warns not to use it for decisions.
- **Cars from future years.** There has been no validation on a PBEV from another year yet. The temporal test waits for
  PBEV 2027.
- **On-road consumption.** The model learns homologation consumption, which is usually lower than real-world use.
- **Portfolio.** The range of the weighted average assumes independent errors across cars. Sibling versions tend to be
  wrong together, so the real range can be wider.

## How to reproduce

```bash
uv run eficiencia-treinar --tentativas 100
```

It took about 26 h on an i7-9750H in the 2026-11-25 run. The process accumulates memory over thousands of trials, and the 2026-11-09 run
stopped for lack of memory after about 11 h. Optuna saves every trial in `optuna/estudos.db`, so running the same
command again resumes where it stopped.
