# Model card: light-vehicle energy efficiency

Date: 2026-11-10 · model: monotonic XGBoost · data: PBEV 2026 (Inmetro), hash `ad3d96d0`

## What it does

From a vehicle's technical specifications, it estimates energy consumption in **MJ/km** (city, highway and combined)
and the **km/l** for each applicable fuel, always with a **90 % range**. It is meant for comparing versions and
building portfolio scenarios before the car exists. **It does not replace the homologation test.**

The inputs are powertrain, fuel, category, PROCONVE phase, transmission and steering. The model also uses curb mass,
inertia class, gross vehicle weight, the road-load coefficients F0 and F2, power, displacement and number of gears.
Electric and plug-in vehicles add battery capacity and electric range.

## Data

The data has 510 versions from PBEV 2026: 288 combustion, 109 electric, 63 hybrid and 50 plug-in hybrids. The "true"
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
| **XGBoost (selected)** | **5.71 %** | 4.71 % | 4.57 % | 7.20 % | 12.07 % |
| Random forest | 6.55 % | 5.31 % | 4.64 % | 11.61 % | 11.46 % |
| Neural network, one per target | 6.77 % | 5.51 % | 5.54 % | 7.32 % | 16.02 % |
| Neural network, multi-output | 6.95 % | 6.12 % | 5.71 % | 7.16 % | 14.21 % |
| Linear regression | 7.46 % | 5.93 % | 7.38 % | 7.42 % | 16.52 % |

The candidates are ordered by mean error. The previous model had 5.77 %. Details are in [relatorio.md](relatorio.md),
and the experiments behind the current inputs are in
[experimentos/conclusoes.md](experimentos/conclusoes.md).

## The 90 % range delivers what it promises

The range comes from out-of-sample predictions: each car is predicted by a model that did not see it in training.
Coverage is the share of cars whose true value fell inside the range the app shows.

| Powertrain | MJ/km coverage | km/l coverage | Typical half-width (MJ/km) |
|---|---|---|---|
| Combustion | 90.6 % | 93.5 % | ±10.5 % |
| Electric | 90.5 % | — | ±10.7 % |
| Hybrid | 91.5 % | 96.2 % | ±15.5 % |
| Plug-in | 92.0 % | 92.2 % | ±27.3 % |

For combustion and hybrids, km/l is derived from MJ/km with each fuel's physical factor, so the two numbers never
contradict each other. For plug-ins that ratio varies a lot from car to car, so km/l comes from dedicated models.

## Simulator example

Citroën Aircross flex, 1210 kg. This car is in the data, which is why the prediction matches the true value almost
exactly.

- **Prediction:** 1.80 MJ/km (range 1.61 to 1.99); homologation value: 1.80.
- **Gasoline:** 12.0 km/l (range 10.8 to 13.6).
- **Ethanol:** 8.3 km/l.

| Change | Effect on MJ/km | Gasoline |
|---|---|---|
| Mass −10 % | −2.3 % | 12.0 → 12.3 km/l |
| F0 −10 % | −1.7 % | 12.0 → 12.2 km/l |
| F2 (aerodynamics) −10 % | −5.0 % | 12.0 → 12.7 km/l |

The simulator averages the predictions within ±5 % around each adjustment. Without that, tree models respond in steps.

## Where not to use it, or use it with care

- **Outside the data ranges.** If an input goes below the minimum or above the maximum seen for that powertrain, the
  app shows a warning, and the prediction becomes an extrapolation.
- **Plug-in hybrids.** The error is 12 % and the range is about ±27 %, because there are only 50 cars and consumption
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

It takes about 21 h on an i7-9750H. The process accumulates memory over thousands of trials, and the 2026-11-09 run
stopped for lack of memory after about 11 h. Optuna saves every trial in `optuna/estudos.db`, so running the same
command again resumes where it stopped.
