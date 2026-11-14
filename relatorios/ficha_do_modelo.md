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

## The 90 % range delivers what it promises

The range comes from out-of-sample predictions: each car is predicted by a model that did not see it in training.
Coverage is the share of cars whose true value fell inside the range the app shows.

| Powertrain | MJ/km coverage | km/l coverage | Typical half-width (MJ/km) |
|---|---|---|---|
| Combustion | 90.6 % | 93.7 % | ±9.8 % |
| Electric | 90.8 % | — | ±10.4 % |
| Hybrid | 92.3 % | 94.6 % | ±20.0 % |
| Plug-in | 92.0 % | 92.2 % | ±30.8 % |

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

The simulator averages the predictions within ±5 % around each adjustment. Without that, tree models respond in steps.

## Where not to use it, or use it with care

- **Outside the data ranges.** If an input goes below the minimum or above the maximum seen for that powertrain, the
  app shows a warning, and the prediction becomes an extrapolation.
- **Plug-in hybrids.** The error is 13 % and the range is about ±31 %, because there are only 50 cars and consumption
  depends heavily on how the battery is used. Use it for orders of magnitude, not to choose between similar versions.
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
