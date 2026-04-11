# Dynamic Disequilibrium Input–Output Model

A Python implementation of a simple single-region Dynamic Disequilibrium Input-Output (IO) model with multiple production rules, inventory dynamics, labour adjustment, and two shock types: a consumption shock and a supplier-side input-availability shock.

## Overview

In this setup of the Dynamic Disequilibrium[^d] IO model it is strictly single-region: one set of sectors, one technical-coefficient matrix, and one final-demand vector per period. It is discrete-time and sector-level. Production can be Leontief, adapted Leontief, linear, or CES. Inventories buffer production through beginning-of-period stocks and rebuild through a damped positive restocking rule rather than an unconstrained stock-gap correction. Labour adjusts gradually toward the output level implied by the previous period's non-labour constraints, subject to disruption and capacity bounds. Household demand is benchmark-consistent by construction: the no-shock baseline is an exact fixed point, demand is formed from a single lagged household-income concept, and savings are reported as an accounting outcome rather than fed back through an endogenous wealth-gap loop. When `savings_rate` is not supplied, the default rate is inferred from the observed base-year household accounts so the data-calibrated `cons_vec` remains the benchmark. The repository distinguishes between mild example shocks and tighter stress cases, especially for input-availability comparisons where structural differences between production rules only become visible when inventories are low enough for supply-chain bottlenecks to bind. Monte Carlo uncertainty analysis over parameter distributions is supported, and results can be plotted as total output or percentage change from baseline, with optional uncertainty bands. Theoretical underpinnings and data sources are described in references [1-5] below; reference 4 gives the data source and licence for the example inputs (EXIOBASE).

[^d]: Persistent possibility of excess demand or supply, with quantity adjustment and rationing rather than instantaneous price-mediated market clearing.

This project provides:

- **Core model**: `SingleRegionInputOutputModel` with configurable production (`leontief` / `leontief.adapted` / `linear` / `ces`), inventories, labour adjustment, selectable household closures, consumption shocks, and supplier-side input-availability shocks
- **Scenarios**: `ScenarioManager`, `Scenario`, and `ScenarioRunResult` for baseline and shocked runs; comparison to baseline (GDP and realised consumption)
- **Uncertainty**: `MonteCarloUncertaintyAnalysis` for parameter sampling, run ensembles, and metrics (mean, quantiles) for GDP, consumption, and gross output
- **Configuration**: `ModelConfig` with validation (`n_periods` > 0, `time_frequency` in `"daily"` or `"quarterly"`)

Further equations and implementation choices for the simple plain-vanilla single-region Dynamic Disequilibrium Input–Output (IO) model are documented in `docs/Mathematical_summary.pdf`.


## Requirements

- Python 3.8+
- NumPy
- Matplotlib

No formal installation is required. The repository may be cloned or copied and the application run from the project root so that the data file is found (default: `data/example_data.pkl`; or `ModelConfig.data_path` may be set to the correct path). The example data are derived from a subset of EXIOBASE data (see References, item 4).

## Quick Start

### Baseline run and plotting

```python
from pathlib import Path
from pyMacroIO import (
    ModelConfig,
    ScenarioManager,
    MonteCarloUncertaintyAnalysis,
    ENABLE_PLOTTING,
)

config = ModelConfig(
    n_periods=30,
    time_frequency="daily",
    prod_function="leontief",  # Explicit example choice; the library default is "leontief.adapted".
    savings_rate=0.05,
)
manager = ScenarioManager(config)
baseline_run = manager.run_baseline(force=True)

figures_dir = Path("figures")
figures_dir.mkdir(parents=True, exist_ok=True)

mc = MonteCarloUncertaintyAnalysis(baseline_run.model, n_simulations=50)
mc.run_uncertainty_analysis(shock_scenario="baseline", seed=42)
uncertainty = mc.get_uncertainty_data_for_plotting()

if ENABLE_PLOTTING:
    baseline_run.model.plot_results(
        baseline_run.results,
        baseline_results=None,
        title_suffix="(Daily Baseline)",
        save_path=str(figures_dir / "baseline.png"),
        uncertainty_data=uncertainty,
    )
```

### Consumption-shock scenario

```python
from pyMacroIO import run_consumption_shock_scenario

scenario_run, baseline_run = run_consumption_shock_scenario(
    intensity=0.2,
    duration=3,
    start=2,
)

# Percentage deviation from baseline
from pyMacroIO import ScenarioManager
comparison = ScenarioManager.compare_to_baseline(scenario_run, baseline_run)
# comparison["gdp_pct"], comparison["consumption_pct"]
```

### Input-availability shock scenario

```python
from pyMacroIO import run_input_availability_shock_scenario

# Key supplier sector is used when input_sector_label is None.
# This helper is the moderate example case.
scenario_run, baseline_run = run_input_availability_shock_scenario(
    input_sector_label=None,
    reduction_pct=0.3,
    duration=3,
    start=2,
    inventory_days=5.0,
)
# run_input_availability_shock_all_prod_functions(...) uses a tighter stress case by default
# so the production rules separate for structural reasons rather than because of plotting noise.
```

## Documentation

- **Mathematical summary**: `docs/Mathematical_summary.pdf` describes the model equations, data calibration, adapted-Leontief essential-input identification, production blocks, inventories, labour, household closure, and shock semantics.

## Key Features

### Core model

- **SingleRegionInputOutputModel**: single-region Dynamic Disequilibrium IO model with Leontief / adapted Leontief / linear / CES production
- **ModelConfig**: Scenario-overridable parameters with validation (`n_periods`, `time_frequency`, `prod_function`, etc.)
- **Inventories**: Target levels and adjustment speed `tau`; beginning-of-period stocks constrain production and end-of-period stocks rebuild from realised deliveries net of input use with damped positive restocking
- **Labour**: Hiring and firing with sector-specific speeds and capacity bounds; labour moves toward the output level implied by yesterday's non-labour constraints
- **Households**: Two closure modes are available. `return_to_base` is benchmark-anchored and removes permanent demand scarring from temporary shocks. `scarred` uses lagged realised income and allows persistent demand scars. Ex post savings are reported in both cases.

### Scenarios and shocks

- **ScenarioManager**: Baseline run (cached), scenario run with shock callables, comparison to baseline
- **Consumption shock (example)**: Scenarios apply it by setting `model.epsilon_[t]` over a start period and duration (e.g. via `run_consumption_shock_scenario`). When demand is the binding constraint, all production functions should coincide.
- **Input-availability shock**: The output capacity of a chosen supplier sector is reduced by a fraction over a duration (e.g. via `run_input_availability_shock_scenario`). The example helper uses a moderate `30%` shock with `5` days of inventory cover. The all-production-function comparison helper defaults to a tighter stress specification (`50%`, `1` day) so `leontief`, `leontief.adapted`, `ces`, and `linear` separate because of their production logic. When `input_sector_label` is not specified, the key supplier sector (largest forward supply) is used. Downstream shortages arise through reduced deliveries and inventory drawdown rather than by directly reducing downstream stocks.
- **Household-closure sensitivity**: `run_consumption_shock_household_closure_sensitivity(...)` and `run_input_availability_shock_household_closure_sensitivity(...)` overlay `return_to_base` and `scarred` closures with separate Monte Carlo bands, so structural uncertainty is shown explicitly rather than folded into one envelope.
- **Overrides**: Expectations (\(\xi\)) and labour disruption (\(\delta\)) may be set via overrides on `model.xi_` and `model.delta_`; for a baseline or unshocked run they remain at defaults.

### Uncertainty

- **MonteCarloUncertaintyAnalysis**: Parameter distributions, sampling, run ensemble, and metrics (mean, std, quantiles) for GDP, consumption, and gross output; integration with `plot_results` for uncertainty bands. Uncertainty bands reflect parameter uncertainty only.

## Data

The default data file is a Python pickle (`data/example_data.pkl`) containing base-year IO and final-demand data. The following entries must be included: `sector_labels` (list of sector names), `Z0` (inter-industry flows, square matrix \(N\times N\)), `cons_vec`, `gov_vec`, `inv_vec`, `invnt_vec`, `exp_vec` (final-demand vectors of length \(N\)), `l0`, `cap0`, `tax0`, `imp0` (value-added components, length \(N\)), and `consumer_taxes_total`, `fd_imports_totals`. ROW and value-added identities are assumed and checked at load. Gross output is derived from the row identity. Definitions are given in `docs/Mathematical_summary.pdf` (Data and Parameter Calibration). The example data are derived from a subset of EXIOBASE data (see References, item 4).

## Outputs

- **figures/baseline.png**: Total output (or absolute values) for the baseline run; optional uncertainty bands (parameter uncertainty) when `uncertainty_data` is passed to `plot_results`.
- **figures/consumption_shock_all_prod_functions.png**: Percentage change from baseline for the consumption-shock scenario, all production functions in one plot; optional uncertainty bands (parameter uncertainty).
- **figures/consumption_shock_household_closure_sensitivity.png**: Consumption-shock comparison of `return_to_base` and `scarred` household closures for a chosen production function, with separate uncertainty bands for each closure.
- **figures/input_availability_shock_all_prod_functions.png**: Percentage change from baseline for the stress-tier input-availability comparison (key supplier reduced), all production functions in one plot; optional uncertainty bands (parameter uncertainty). Produced when `ENABLE_INPUT_AVAILABILITY_SHOCK_PLOT` is `True`.
- **figures/input_availability_shock_household_closure_sensitivity.png**: Input-shock comparison of `return_to_base` and `scarred` household closures for a chosen production function, with separate uncertainty bands for each closure.
- **figures/input_availability_sensitivity_panel.png**: Leontief sensitivity panel comparing the stress case against milder-shock and higher-inventory variants, so the moderate example is not treated as the whole story.

These figures are produced when the main script is run with `ENABLE_PLOTTING` set to `True`:

```bash
python3 pyMacroIO.py
```

## Licence

The licence for this software is described in the LICENSE file. The licence for the example data is different; the full terms are specified by the data source. See [4] and the [EXIOBASE licence file](https://zenodo.org/records/15689391/preview/LICENSE.txt) for the authoritative conditions.

### How to cite

This implementation should be cited as: [0], building on [1–3]; [4] should also be cited when the included example data (a small subset of EXIOBASE) are used. The licence thereof must also be adhered to (see Licence section above).

## References

0. Ross, A. G. (2025). A Python implementation of a single-region Dynamic Disequilibrium Input–Output model. [10.5281/zenodo.18419984](https://doi.org/10.5281/zenodo.18419984)  

1. Pichler, A., Pangallo, M., del Rio-Chanona, R. M., Lafond, F., & Farmer, J. D. (2022). Forecasting the propagation of pandemic shocks with a dynamic input–output model. *Journal of Economic Dynamics and Control*, 144, 104527. <https://doi.org/10.1016/j.jedc.2022.104527>

2. Ross, A. G., McGregor, P. G., & Swales, J. K. (2024). Labour market dynamics in the era of technological advancements: The system-wide impacts of labour augmenting technological change. *Technology in Society*, 77, 102539. <https://doi.org/10.1016/j.techsoc.2024.102539>

3. Raseta, M., Ross, A. G., & Voegele, S. (2025). Macro-level implications of the energy system transition to net-zero carbon emissions: Identifying quick wins amid short-term constraints. *Economic Analysis and Policy*, 85, 1065–1078. <https://doi.org/10.1016/j.eap.2025.01.011>

4. Stadler, K., Wood, R., Bulavskaya, T., Södersten, C.-J., Simas, M., Schmidt, S., Usubiaga, A., Acosta-Fernández, J., Kuenen, J., Bruckner, M., Giljum, S., Lutter, S., Merciai, S., Schmidt, J. H., Theurl, M. C., Plutzar, C., Kastner, T., Eisenmenger, N., Erb, K.-H., … Tukker, A. (2025). EXIOBASE 3 (3.9.6) [Data set]. Zenodo. <https://doi.org/10.5281/zenodo.15689391>

5. Miller, R. E., & Blair, P. D. (2009). Input-output analysis: foundations and extensions. Cambridge university press.

