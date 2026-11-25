# Model card: light-vehicle energy efficiency

Date: 2026-11-13 · model: monotonic XGBoost · data: PBEV 2026 (Inmetro), hash `7084f5b9`

## What it does

From a vehicle's technical specifications, it estimates energy consumption in **MJ/km** (city, highway and combined)
and the **km/l** for each applicable fuel, always with a **90 % range**. It is meant for comparing versions and
building portfolio scenarios before the car exists. **It does not replace the homologation test.**

The inputs are powertrain, fuel, category, PROCONVE phase, transmission and steering. The model also uses curb mass,
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
| **XGBoost (selected)** | **5.64 %** | 4.17 % | 4.76 % | 7.93 % | 13.05 % |
| Neural network, one per target | 6.52 % | 5.17 % | 5.67 % | 7.14 % | 15.28 % |
| Random forest | 6.53 % | 5.01 % | 4.56 % | 12.14 % | 12.25 % |
| Neural network, multi-output | 6.62 % | 5.41 % | 5.29 % | 6.73 % | 16.33 % |
| Linear regression | 7.36 % | 5.63 % | 7.43 % | 7.88 % | 16.48 % |

The candidates are ordered by mean error. The previous model had 5.71 %; the gain comes from fixing the label of two
Toyota RAV4 Hybrid rows that the spreadsheet entered as combustion. Details are in [relatorio.md](relatorio.md),
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
| Combustion | 286 | 90.6 % | 87.3–93.0 % | 94.1 % |
| Electric | 109 | 90.8 % | 85.2–94.4 % | — |
| Hybrid | 65 | 92.3 % | 85.0–96.2 % | 91.4 % |
| Plug-in | 50 | 92.0 % | 83.3–96.4 % | 94.0 % |
| **All** | **510** | **91.0 %** | **88.7–92.9 %** | **93.6 %** |

- **Per fold:** MJ/km coverage goes from 87.6 % to 95.5 %.
- **Ethanol and diesel:** km/l coverage is 97.0 % and 98.2 %, so the range there is wider than it needs to be.
- **Still open:** all of this is one model year. Cars from the next PBEV may be farther from the data.

**Half-widths the app uses**, read from the shipped model package:

| Powertrain | Combined | City | Highway |
|---|---|---|---|
| Combustion | ±9.0 % | ±12.4 % | ±9.8 % |
| Electric | ±10.4 % | ±14.3 % | ±9.6 % |
| Hybrid | ±20.0 % | ±24.6 % | ±15.7 % |
| Plug-in | ±24.6 % | ±33.3 % | ±30.8 % |

For combustion and hybrids, km/l is derived from MJ/km with each fuel's physical factor, so the two numbers never
contradict each other. For plug-ins that ratio varies a lot from car to car, so km/l comes from dedicated models.

## Simulator example

Citroën Aircross flex, 1210 kg. This car is in the data, which is why the prediction matches the true value almost
exactly.

- **Prediction:** 1.80 MJ/km (range 1.64 to 1.96); homologation value: 1.80.
- **Gasoline:** 12.0 km/l (range 10.8 to 13.6).
- **Ethanol:** 8.4 km/l.

| Change | Effect on MJ/km | Gasoline |
|---|---|---|
| Mass −10 % | −2.5 % | 12.0 → 12.3 km/l |
| F0 −10 % | −1.8 % | 12.0 → 12.3 km/l |
| F2 (aerodynamics) −10 % | −4.8 % | 12.0 → 12.6 km/l |

The simulator averages the predictions within ±5 % around each adjustment. Without that, tree models respond in steps. The scenario export marks these rows as smoothed.

## Where not to use it, or use it with care

- **Outside the data ranges.** If an input goes below the minimum or above the maximum seen for that powertrain, the
  app shows a warning, and the prediction becomes an extrapolation.
- **Plug-in hybrids.** The error is 13 % and the combined range is about ±25 %, because there are only 50 cars and consumption
  depends heavily on how the battery is used. Use it for orders of magnitude, not to choose between similar versions.
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

It took about 36 h on an i7-9750H in the 2026-11-12 run. The process accumulates memory over thousands of trials, and the 2026-11-09 run
stopped for lack of memory after about 11 h. Optuna saves every trial in `optuna/estudos.db`, so running the same
command again resumes where it stopped.
