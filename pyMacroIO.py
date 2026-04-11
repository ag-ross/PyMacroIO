#!/usr/bin/env python3

"""
Dynamic Disequilibrium Input-Output Model.

This module implements a simple plain-vanilla single-region Dynamic Disequilibrium
Input-Output (IO) model with Leontief- and CES-style production, inventory dynamics,
labour hiring and firing, and hooks for output and input-availability shocks.
"""


import copy
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Callable, Any

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


# -----------------------------------------------------------------------------
# Model configuration and default calibration
# -----------------------------------------------------------------------------
# Default calibration constants are used by the model when not overridden. They are
# documented here so all tunable defaults live in one place. ModelConfig (below)
# holds the parameters that are intended to be overridden per scenario or run.
#
# Default calibration (used in model logic)
# -----------------------------------------
# Fallback household savings rate used only when a baseline rate cannot be inferred from data.
DEFAULT_SAVINGS_RATE = 0.05

# Hire/fire capacity bounds. Base capacity is (1 - delta) clipped to [DELTA_FLOOR, DELTA_CAP].
# Labour capacity is restricted to [CAPACITY_MIN_SCALE, CAPACITY_MAX_SCALE] times initial labour.
DELTA_FLOOR = 0.2
DELTA_CAP = 1.0
CAPACITY_MIN_SCALE = 0.3
CAPACITY_MAX_SCALE = 1.5

# Firing-speed damping factor applied to gamma_fire when reducing labour.
FIRING_SPEED_DAMPING = 0.5

# Consumption demand floors in findemand_cd, expressed as a ratio of baseline consumption.
CONSUMPTION_FLOOR_RATIO = 0.5
CONSUMPTION_FLOOR_LABOUR_RATIO = 0.2
HOUSEHOLD_CLOSURE_MODES = ("return_to_base", "scarred")
PRODUCTION_FUNCTIONS = ("leontief", "leontief.adapted", "linear", "ces")
FIRM_PRIORITY_MODES = ("no", "yes")

# Threshold used in producing_x: inputs with A_essential above this value
# are classified as essential in adapted Leontief.
ESSENTIAL_INPUT_THRESHOLD = 0.5

# Tolerances used when checking row and value-added identity in the IO balance.
ROW_IDENTITY_ATOL = 1e-10
VA_IDENTITY_TOLERANCE = 1.0

# Default tau and labour-adjustment speeds when sector count is expanded from config.
DEFAULT_TAU = 2.9
DEFAULT_GAMMA_HIRE = 0.375
DEFAULT_GAMMA_FIRE = 0.5

# Parameter bounds enforced in _validate_parameters (gamma_hire, gamma_fire, tau).
GAMMA_HIRE_MIN = 0.1
GAMMA_HIRE_MAX = 0.8
GAMMA_FIRE_MIN = 0.1
GAMMA_FIRE_MAX = 0.8
TAU_MIN = 0.5
TAU_MAX = 5.0
CES_ELASTICITY_DEFAULT = 1.5

# Large finite value used when replacing inf/nan in producing_x for numerical stability.
NUMERIC_LARGE = 1e6


# -----------------------------------------------------------------------------
# Scenario-overridable parameters (ModelConfig)
# -----------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """Parameters that may be overridden per scenario or run. A deep copy is created for each scenario."""

    n_periods: int = 60
    time_frequency: str = "daily"
    tau: Optional[np.ndarray] = None
    gamma_hire: Optional[np.ndarray] = None
    gamma_fire: Optional[np.ndarray] = None
    benefits: float = 0.1
    c_other_coef: float = 0.1
    prod_function: str = "leontief.adapted"
    hiringfiring: bool = True
    firm_priority: str = "no"
    inventory_days: Optional[np.ndarray] = None
    inventory_days_daily: float = 2.0
    inventory_days_other: float = 2.0
    data_path: str = "data/example_data.pkl"
    savings_rate: Optional[float] = None
    ces_elasticity: float = CES_ELASTICITY_DEFAULT
    household_closure_mode: str = "return_to_base"

    def __post_init__(self) -> None:
        """n_periods and time_frequency are validated; ValueError is raised if invalid."""
        if self.n_periods <= 0:
            raise ValueError(f"n_periods must be positive; got {self.n_periods}")
        if self.time_frequency not in ("daily", "quarterly"):
            raise ValueError(
                f"time_frequency must be 'daily' or 'quarterly'; got {self.time_frequency!r}"
            )
        if self.savings_rate is not None and not (0 <= self.savings_rate < 1):
            raise ValueError(f"savings_rate must be in [0, 1); got {self.savings_rate}")
        if self.ces_elasticity <= 0:
            raise ValueError(f"ces_elasticity must be positive; got {self.ces_elasticity}")
        if self.household_closure_mode not in HOUSEHOLD_CLOSURE_MODES:
            raise ValueError(
                f"household_closure_mode must be one of {HOUSEHOLD_CLOSURE_MODES}; "
                f"got {self.household_closure_mode!r}"
            )
        if self.prod_function not in PRODUCTION_FUNCTIONS:
            raise ValueError(
                f"prod_function must be one of {PRODUCTION_FUNCTIONS}; got {self.prod_function!r}"
            )
        if self.firm_priority not in FIRM_PRIORITY_MODES:
            raise ValueError(
                f"firm_priority must be one of {FIRM_PRIORITY_MODES}; got {self.firm_priority!r}"
            )

    def clone(self) -> "ModelConfig":
        """A deep copy of this config is returned."""
        return copy.deepcopy(self)


# -----------------------------------------------------------------------------
# Script defaults (used when the file is run as __main__)
# -----------------------------------------------------------------------------
# Number of time periods used for baseline and example scenarios when the script is executed.
SIMULATION_PERIODS = 60
# When True, total-output figures are generated and saved; when False, plotting is skipped.
ENABLE_PLOTTING = True
# When True, the input-availability shock comparison (all production functions) is run and plotted; 
# may be set to False to remove this scenario.
ENABLE_INPUT_AVAILABILITY_SHOCK_PLOT = True
# Number of Monte Carlo runs used for uncertainty bands in demo plots.
MC_PLOT_SIMULATIONS = 50
# Default headline calibration for the example input-availability scenario.
INPUT_SHOCK_DEFAULT_REDUCTION_PCT = 0.3
INPUT_SHOCK_DEFAULT_DURATION = 3
INPUT_SHOCK_DEFAULT_START = 2
INPUT_SHOCK_DEFAULT_INVENTORY_DAYS = 5.0
INPUT_SHOCK_STRESS_REDUCTION_PCT = 0.5
INPUT_SHOCK_STRESS_DURATION = 3
INPUT_SHOCK_STRESS_START = 2
INPUT_SHOCK_STRESS_INVENTORY_DAYS = 1.0


# -----------------------------------------------------------------------------
# Scenario and run result structures
# -----------------------------------------------------------------------------
@dataclass
class Scenario:
    """A named scenario, defined by a config and an ordered list of shock callables that are applied to the model before each run."""

    name: str
    description: str
    config: ModelConfig
    shocks: List[Callable[["SingleRegionInputOutputModel"], None]] = field(default_factory=list)


@dataclass
class ScenarioRunResult:
    """The result of a single scenario run: the scenario definition, the model instance used, and the results dictionary returned by run_model."""

    scenario: Scenario
    model: "SingleRegionInputOutputModel"
    results: Dict[str, Any]


@dataclass(frozen=True)
class ConsumptionShockSpec:
    """A consumption-shock specification used for examples, sensitivity runs, or stress tests."""

    intensity: float = 0.2
    duration: int = 3
    start: int = 2
    tier: str = "example"


@dataclass(frozen=True)
class InputAvailabilityShockSpec:
    """An input-availability shock specification used for examples, sensitivity runs, or stress tests."""

    reduction_pct: float = INPUT_SHOCK_DEFAULT_REDUCTION_PCT
    duration: int = INPUT_SHOCK_DEFAULT_DURATION
    start: int = INPUT_SHOCK_DEFAULT_START
    inventory_days: Optional[np.ndarray] = INPUT_SHOCK_DEFAULT_INVENTORY_DAYS
    input_sector_label: Optional[str] = None
    tier: str = "example"


CONSUMPTION_EXAMPLE_SHOCK_SPEC = ConsumptionShockSpec()
INPUT_AVAILABILITY_EXAMPLE_SHOCK_SPEC = InputAvailabilityShockSpec()
INPUT_AVAILABILITY_STRESS_SHOCK_SPEC = InputAvailabilityShockSpec(
    reduction_pct=INPUT_SHOCK_STRESS_REDUCTION_PCT,
    duration=INPUT_SHOCK_STRESS_DURATION,
    start=INPUT_SHOCK_STRESS_START,
    inventory_days=INPUT_SHOCK_STRESS_INVENTORY_DAYS,
    tier="stress",
)


# -----------------------------------------------------------------------------
# Scenario manager
# -----------------------------------------------------------------------------
class ScenarioManager:
    """Baseline and scenario runs are coordinated here. The baseline result is cached and may be reused for comparisons."""

    def __init__(self, base_config: ModelConfig):
        self.base_config = base_config.clone()
        self._baseline_cache: Optional[ScenarioRunResult] = None

    def _instantiate_model(self, config: ModelConfig) -> "SingleRegionInputOutputModel":
        """A model instance is created from the given config."""
        return SingleRegionInputOutputModel(
            n_periods=config.n_periods,
            time_frequency=config.time_frequency,
            config=config
        )

    def run_baseline(self, force: bool = False) -> ScenarioRunResult:
        """The baseline scenario is run (or the cached result returned). If force is True, the cache is bypassed."""
        if force or self._baseline_cache is None:
            baseline_config = self.base_config.clone()
            model = self._instantiate_model(baseline_config)
            results = model.run_model()
            baseline_scenario = Scenario(
                name="baseline",
                description="Baseline scenario",
                config=baseline_config,
                shocks=[]
            )
            self._baseline_cache = ScenarioRunResult(baseline_scenario, model, results)
        return self._baseline_cache

    def run_scenario(self, scenario: Scenario, use_cached_baseline: bool = True) -> Tuple[ScenarioRunResult, Optional[ScenarioRunResult]]:
        """The given scenario is run; shocks are applied to a fresh model instance. The scenario run and the baseline run (if requested) are returned."""
        baseline = self.run_baseline(force=False) if use_cached_baseline else None
        scenario_model = self._instantiate_model(scenario.config.clone())
        for shock in scenario.shocks:
            shock(scenario_model)
        scenario_results = scenario_model.run_model()
        scenario_run = ScenarioRunResult(scenario, scenario_model, scenario_results)
        return scenario_run, baseline

    @staticmethod
    def compare_to_baseline(run: ScenarioRunResult, baseline: ScenarioRunResult) -> Dict[str, np.ndarray]:
        """Percentage deviations of GDP and realised consumption from the baseline are returned as arrays (keys 'gdp_pct' and 'consumption_pct')."""
        def safe_pct_change(current: np.ndarray, reference: np.ndarray) -> np.ndarray:
            pct = np.full_like(current, np.nan, dtype=np.float64)
            nonzero_reference = reference != 0
            pct[nonzero_reference] = (current[nonzero_reference] / reference[nonzero_reference] - 1) * 100
            zero_and_equal = (~nonzero_reference) & np.isclose(current, reference)
            pct[zero_and_equal] = 0.0
            return pct

        comparison: Dict[str, np.ndarray] = {}
        baseline_gdp = baseline.results['gdp']
        scenario_gdp = run.results['gdp']
        comparison['gdp_pct'] = safe_pct_change(scenario_gdp, baseline_gdp)
        baseline_realised_cons = np.sum(baseline.results['realised_consumption'], axis=0)
        scenario_realised_cons = np.sum(run.results['realised_consumption'], axis=0)
        comparison['consumption_pct'] = safe_pct_change(scenario_realised_cons, baseline_realised_cons)
        return comparison


# -----------------------------------------------------------------------------
# Single-region input-output model (core logic, shock hooks, simulation, plotting)
# -----------------------------------------------------------------------------
class SingleRegionInputOutputModel:
    """Simple plain-vanilla single-region Dynamic Disequilibrium Input-Output (IO) model with Leontief- or CES-style production, inventory dynamics, labour adjustment, and hooks for output and input-availability shocks."""

    def __init__(self, n_periods: int = 40, time_frequency: str = "daily",
                 config: Optional[ModelConfig] = None):
        """The model is initialised from the given config (or defaults). IO data are loaded from the Python-friendly file given by config.data_path (default: data/example_data.pkl)."""
        # --- Initialisation and data loading ---
        if config is None:
            config = ModelConfig(n_periods=n_periods, time_frequency=time_frequency)
        else:
            config = config.clone()
            if config.n_periods != n_periods:
                logger.debug("Overriding config n_periods=%s with explicit argument=%s", config.n_periods, n_periods)
                config.n_periods = n_periods
            if config.time_frequency != time_frequency:
                logger.debug("Overriding config time_frequency=%s with explicit argument=%s", config.time_frequency, time_frequency)
                config.time_frequency = time_frequency

        self.config = config
        self.TT = config.n_periods
        self.N = 3
        self.time_frequency = config.time_frequency
        
        self._calculate_time_step_parameters()

        def _resolve_array(value: Optional[np.ndarray], fallback: float) -> np.ndarray:
            if value is None:
                return np.array([fallback], dtype=np.float64)
            return np.atleast_1d(np.asarray(value, dtype=np.float64))

        self.tau = _resolve_array(config.tau, DEFAULT_TAU)
        self.gamma_hire = _resolve_array(config.gamma_hire, DEFAULT_GAMMA_HIRE)
        self.gamma_fire = _resolve_array(config.gamma_fire, DEFAULT_GAMMA_FIRE)
        self.benefits = config.benefits
        
        self.c_other_coef = config.c_other_coef
        self.ces_elasticity = config.ces_elasticity
        self.household_closure_mode = config.household_closure_mode
        
        self.prod_function = config.prod_function
        self.hiringfiring = config.hiringfiring
        self.firm_priority = config.firm_priority
        self.inventory_days_config = config.inventory_days
        self.inventory_days_daily = config.inventory_days_daily
        self.inventory_days_other = config.inventory_days_other
        
        self._initialize_data()
        self._validate_parameters()
    
    def _load_data_dict(self, path: Path) -> dict:
        """The data dictionary is loaded from the given path (Python pickle format)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Data file not found: {path}. The data file may be created by running the export script (e.g. from Archive/data/) from the CSV files."
            )
        with open(path, "rb") as f:
            return pickle.load(f)

    def _initialize_essential_inputs(self) -> None:
        """Essential-input indicators are initialised from the IO matrix when production is leontief.adapted; otherwise A_essential is set to None."""
        if self.prod_function == "leontief.adapted":
            temp_mc = MonteCarloUncertaintyAnalysis(self, n_simulations=1)
            self.A_essential = temp_mc.estimate_essential_inputs_from_io_data(
                self.A, method='combined_linkage'
            )
        else:
            self.A_essential = None
        
    def _resolve_inventory_days_vector(self) -> np.ndarray:
        """A vector of inventory coverage (in days) per sector is resolved from config or from time-frequency defaults."""
        if self.inventory_days_config is not None:
            days = np.asarray(self.inventory_days_config, dtype=np.float64)
            if days.size == 1:
                days = np.full(self.N, float(days.squeeze()), dtype=np.float64)
            elif days.size != self.N:
                raise ValueError(f"inventory_days length ({days.size}) does not match N ({self.N})")
        else:
            default_days = self.inventory_days_daily if self.time_frequency == "daily" else self.inventory_days_other
            days = np.full(self.N, default_days, dtype=np.float64)
        
        return days

    def _expand_sector_parameter(self, value: np.ndarray, default_value: float, name: str) -> np.ndarray:
        """A scalar or sector-length array is expanded to length N; invalid lengths raise ValueError."""
        arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
        if arr.size == 0:
            return np.full(self.N, default_value, dtype=np.float64)
        if arr.size == 1:
            return np.full(self.N, float(arr.squeeze()), dtype=np.float64)
        if arr.size == self.N:
            return arr.astype(np.float64)
        raise ValueError(f"{name} length ({arr.size}) does not match N ({self.N})")

    def _validate_savings_rate_value(self, savings_rate: float) -> float:
        """A validated savings rate in [0, 1) is returned."""
        rate = float(savings_rate)
        if not 0 <= rate < 1:
            raise ValueError(f"savings_rate must be in [0, 1); got {savings_rate}")
        return rate

    def _extra_household_expenditure(self, consumption_total: float) -> float:
        """Other household outlays implied by c_other_coef are returned."""
        return self.c_other_coef / (1 - self.c_other_coef) * consumption_total

    def _infer_base_savings_rate(self, consumption_total: float, household_income: float) -> float:
        """The savings rate implied by base-year household income and observed spending is returned."""
        total_spending = float(consumption_total) + self._extra_household_expenditure(consumption_total)
        disposable_income = max(float(household_income), 1e-9)
        implied_rate = 1 - total_spending / disposable_income
        return self._validate_savings_rate_value(np.clip(implied_rate, 0.0, 1.0 - 1e-9))

    def _household_income(self, labour_income_total: float, profit_income_total: float) -> float:
        """Disposable household income is returned from adjusted labour income and profits."""
        baseline_labour_income = float(np.sum(self.l0))
        adjusted_labour_income = self.benefits * baseline_labour_income + (1 - self.benefits) * float(labour_income_total)
        return max(adjusted_labour_income + float(profit_income_total), 1e-9)

    def _household_consumption_capacity(
        self,
        household_income: float,
        household_wealth: float = 0.0,
        savings_rate: Optional[float] = None
    ) -> float:
        """Maximum household consumption consistent with available resources and the savings rule is returned."""
        rate = self.savings_rate if savings_rate is None else self._validate_savings_rate_value(savings_rate)
        available_resources = max(float(household_income) + float(household_wealth), 1e-9)
        total_household_spending = (1 - rate) * available_resources
        return max((1 - self.c_other_coef) * total_household_spending, 1e-9)

    def _household_income_signal_for_period(self, household_income_prev: float, xit: float) -> float:
        """The beginning-of-period household income signal is returned for the selected closure mode."""
        if self.household_closure_mode == "scarred":
            return max(float(household_income_prev), 1e-9)
        return max(self.base_household_income * float(xit), 1e-9)

    def _set_household_baseline(self, cons_vec_template: np.ndarray) -> None:
        """Household shares, baseline income, and baseline consumption are initialised consistently."""
        household_consumption_template = np.asarray(cons_vec_template, dtype=np.float64)
        template_total = float(np.sum(household_consumption_template))
        if template_total > 0:
            self.household_consumption_shares = household_consumption_template / template_total
        else:
            self.household_consumption_shares = np.full(self.N, 1 / self.N, dtype=np.float64)

        self.c0 = household_consumption_template.copy()
        self.base_consumption_total = float(np.sum(self.c0))
        self.base_household_income = self._household_income(np.sum(self.l0), np.sum(self.profits0))
        if self.config.savings_rate is None:
            self.savings_rate = self._infer_base_savings_rate(self.base_consumption_total, self.base_household_income)
        else:
            self.savings_rate = self._validate_savings_rate_value(self.config.savings_rate)
        self.base_consumption_total = float(np.sum(self.c0))
        self.theta_ = np.repeat(self.household_consumption_shares[:, np.newaxis], self.TT, axis=1)
        self.mpc = self.base_consumption_total / max(self.base_household_income, 1e-9)
        self.base_household_savings = (
            self.base_household_income
            - self.base_consumption_total
            - self._extra_household_expenditure(self.base_consumption_total)
        )

    def _ces_output_constraint(self, input_capacity: np.ndarray, weights: np.ndarray) -> float:
        """A CES-style output bound from per-input capacities and technical-coefficient weights is returned."""
        valid = np.isfinite(input_capacity) & (weights > 0)
        if not np.any(valid):
            return np.inf
        q = np.maximum(input_capacity[valid], 0.0)
        w = weights[valid]
        w = w / np.sum(w)
        sigma = self.ces_elasticity
        if np.isclose(sigma, 1.0):
            safe_q = np.maximum(q, 1e-12)
            return float(np.exp(np.sum(w * np.log(safe_q))))
        rho = (sigma - 1.0) / sigma
        aggregate = np.sum(w * np.power(q, rho))
        return float(np.power(max(aggregate, 0.0), 1.0 / rho))

    def _period_output_constraints(self, t: int) -> np.ndarray:
        """The effective output constraints for period t, including supplier-side shocks, are returned."""
        output_constraint = np.array(self.output_constraint_[:, t], copy=True, dtype=np.float64)
        if t in self.input_availability_shocks_:
            for supplier_idx, reduction_pct in self.input_availability_shocks_[t].items():
                shocked_capacity = (1 - reduction_pct) * self.x0[supplier_idx]
                output_constraint[supplier_idx] = min(output_constraint[supplier_idx], shocked_capacity)
        return output_constraint

    def _calculate_time_step_parameters(self) -> None:
        """Time-step length (dt) and consumption persistence (rho0, rho1) are set from time_frequency."""
        if self.time_frequency == "quarterly":
            self.dt = 0.25
            self.rho1 = 0.6
            self.rho0 = 0.4
        elif self.time_frequency == "daily":
            self.dt = 1/90
            rho_bar = 0.6
            self.rho1 = 1 - (1 - rho_bar) * self.dt
            self.rho0 = 1 - self.rho1
        else:
            self.dt = 1.0
            self.rho1 = 0.6
            self.rho0 = 0.4
    
    def _validate_parameters(self) -> None:
        """gamma_hire, gamma_fire, and tau are validated against module bounds; sector count must match N."""
        if not np.all((self.gamma_hire >= GAMMA_HIRE_MIN) & (self.gamma_hire <= GAMMA_HIRE_MAX)):
            raise ValueError(f"gamma_hire must be between {GAMMA_HIRE_MIN} and {GAMMA_HIRE_MAX} for all sectors")
        if not np.all((self.gamma_fire >= GAMMA_FIRE_MIN) & (self.gamma_fire <= GAMMA_FIRE_MAX)):
            raise ValueError(f"gamma_fire must be between {GAMMA_FIRE_MIN} and {GAMMA_FIRE_MAX} for all sectors")
        if not np.all((self.tau >= TAU_MIN) & (self.tau <= TAU_MAX)):
            raise ValueError(f"tau must be between {TAU_MIN} and {TAU_MAX} for all sectors")
        if len(self.gamma_hire) != self.N or len(self.gamma_fire) != self.N or len(self.tau) != self.N:
            raise ValueError("All sector-specific parameter arrays must have length equal to N (number of sectors)")
        
    
    def _initialize_data(self) -> None:
        """IO data are loaded from config.data_path; matrices, shares, and time-series buffers are initialised and identity checks are performed."""
        data = self._load_data_dict(Path(self.config.data_path))
        self.sector_labels = data["sector_labels"]
        self.label_to_index = {label: idx for idx, label in enumerate(self.sector_labels)}
        self.Z0 = np.asarray(data["Z0"], dtype=np.float64)
        if self.Z0.ndim != 2 or self.Z0.shape[0] != self.Z0.shape[1]:
            raise ValueError(f"Z matrix must be square. Got shape {self.Z0.shape}")
        self.N = self.Z0.shape[0]
        self.tau = self._expand_sector_parameter(self.tau, DEFAULT_TAU, "tau")
        self.gamma_hire = self._expand_sector_parameter(self.gamma_hire, DEFAULT_GAMMA_HIRE, "gamma_hire")
        self.gamma_fire = self._expand_sector_parameter(self.gamma_fire, DEFAULT_GAMMA_FIRE, "gamma_fire")
        cons_vec = np.asarray(data["cons_vec"], dtype=np.float64)
        gov_vec = np.asarray(data["gov_vec"], dtype=np.float64)
        inv_vec = np.asarray(data["inv_vec"], dtype=np.float64)
        invnt_vec = np.asarray(data["invnt_vec"], dtype=np.float64)
        exp_vec = np.asarray(data["exp_vec"], dtype=np.float64)
        f_total = cons_vec + gov_vec + inv_vec + invnt_vec + exp_vec
        intermediate_outputs = np.sum(self.Z0, axis=1)
        self.x0 = intermediate_outputs + f_total
        with np.errstate(divide="ignore", invalid="ignore"):
            self.A = np.divide(
                self.Z0,
                self.x0[np.newaxis, :],
                out=np.zeros_like(self.Z0),
                where=self.x0[np.newaxis, :] != 0,
            )
        self._initialize_essential_inputs()
        row_identity_check = np.sum(self.Z0, axis=1) + f_total
        row_diff = self.x0 - row_identity_check
        if not np.allclose(self.x0, row_identity_check, atol=ROW_IDENTITY_ATOL):
            max_row_diff = np.max(np.abs(row_diff))
            raise ValueError(
                f"Row IO identity violation after calculation. Max diff: {max_row_diff:.10f}"
            )
        self.l0 = np.asarray(data["l0"], dtype=np.float64)
        self.cap0 = np.asarray(data["cap0"], dtype=np.float64)
        self.tax0 = np.asarray(data["tax0"], dtype=np.float64)
        self.imp0 = np.asarray(data["imp0"], dtype=np.float64)
        if len(self.l0) != self.N or len(self.cap0) != self.N or len(self.tax0) != self.N or len(self.imp0) != self.N:
            raise ValueError(
                f"Data sector count mismatch: l0/cap0/tax0/imp0 length must match N ({self.N})"
            )
        self.other0 = np.zeros(self.N)
        intermediate_inputs_domestic = np.sum(self.Z0, axis=0)
        intermediate_inputs_total = intermediate_inputs_domestic + self.imp0
        value_added_components = self.l0 + self.cap0 + self.tax0 + self.other0
        value_added_from_identity = self.x0 - intermediate_inputs_total
        if not np.allclose(value_added_from_identity, value_added_components, atol=VA_IDENTITY_TOLERANCE):
            max_diff = np.max(np.abs(value_added_from_identity - value_added_components))
            raise ValueError(
                f"Value added inconsistency: max difference = {max_diff}. "
                f"VA from column identity: {value_added_from_identity[:5]}, "
                f"VA from components: {value_added_components[:5]}"
            )
        column_identity_check = intermediate_inputs_total + value_added_components
        column_diff = self.x0 - column_identity_check
        max_column_diff = np.max(np.abs(column_diff))
        if max_column_diff > VA_IDENTITY_TOLERANCE:
            raise ValueError(
                f"Column IO identity violation. Max diff: {max_column_diff:.10f}. "
                "x0 should equal colsum(Z0) + imports + value_added"
            )
        value_added = value_added_from_identity
        self.profits0 = value_added - self.l0 - self.cap0 - self.tax0 - self.other0
        with np.errstate(divide="ignore", invalid="ignore"):
            self.cap_share = np.divide(
                self.cap0, self.x0, out=np.zeros_like(self.cap0), where=self.x0 != 0
            )
            self.tax_share = np.divide(
                self.tax0, self.x0, out=np.zeros_like(self.tax0), where=self.x0 != 0
            )
            self.imp_share = np.divide(
                self.imp0, self.x0, out=np.zeros_like(self.imp0), where=self.x0 != 0
            )
        self.n = self._resolve_inventory_days_vector()
        self.delta_ = np.zeros((self.N, self.TT))
        self.epsilon_ = np.zeros(self.TT)
        self.xi_ = np.ones(self.TT)
        self.output_constraint_ = np.full((self.N, self.TT), np.inf)
        self.input_availability_shocks_ = {}
        self.rationing_shocks_ = {}

        self._set_household_baseline(cons_vec)
        self.consumer_taxes_total = float(data["consumer_taxes_total"])
        self.fd_imports_totals = dict(data["fd_imports_totals"])
        self.fd_government_ = np.zeros((self.N, self.TT))
        self.fd_investment_ = np.zeros((self.N, self.TT))
        self.fd_inventories_ = np.zeros((self.N, self.TT))
        self.fd_exports_ = np.zeros((self.N, self.TT))
        for t in range(self.TT):
            self.fd_government_[:, t] = gov_vec
            self.fd_investment_[:, t] = inv_vec
            self.fd_inventories_[:, t] = invnt_vec
            self.fd_exports_[:, t] = exp_vec
        self.fd_other_ = np.zeros((self.N, self.TT))

    # --- Labour, demand, orders, production (single-period building blocks) ---
    def hire_fire(self, t: int, l_: np.ndarray, x_: np.ndarray,
                  delta_: np.ndarray, prod_constraints: np.ndarray) -> np.ndarray:
        """Labour by sector for period t is updated from period t-1 using hiring/firing speeds and capacity slack; capacity bounds are enforced. If hiringfiring is disabled, previous-period labour is returned."""
        if not self.hiringfiring:
            return l_[:, t-1]
        
        base_capacity = l_[:, 0] * np.clip(1 - delta_[:, t], DELTA_FLOOR, DELTA_CAP)
        max_capacity = l_[:, 0] * CAPACITY_MAX_SCALE
        min_capacity = l_[:, 0] * CAPACITY_MIN_SCALE
        disruption_active = delta_[:, t] > 0
        upper_capacity = np.where(
            disruption_active,
            np.minimum(max_capacity, np.maximum(base_capacity, l_[:, t-1])),
            max_capacity
        )
        labor_share = np.divide(l_[:, 0], x_[:, 0], out=np.zeros_like(l_[:, 0]), where=x_[:, 0] != 0)
        desired_output = np.min(prod_constraints[:, 1:4], axis=1)
        desired_l = np.clip(labor_share * desired_output, min_capacity, upper_capacity)
        gap = desired_l - l_[:, t-1]
        hire_ix = gap > 0
        gam = np.where(hire_ix, self.gamma_hire, self.gamma_fire * FIRING_SPEED_DAMPING)
        adjustment = gam * gap
        new_l = l_[:, t-1] + adjustment
        new_l = np.clip(new_l, min_capacity, upper_capacity)
        
        
        return new_l
    
    def findemand_cd(
        self,
        theta: np.ndarray,
        Cdt: float,
        xit: float,
        household_income_signal: float,
        eps: float
    ) -> Tuple[float, np.ndarray]:
        """Consumption demand is updated from the configured household closure, persistence, and expectations."""
        current_capacity = self._household_consumption_capacity(max(float(household_income_signal), 1e-9))
        expected_capacity = self._household_consumption_capacity(max(self.base_household_income * xit, 1e-9))
        baseline_ct = max(self.base_consumption_total, 1.0)
        if self.household_closure_mode == "scarred":
            log_prev_ratio = np.log(max(Cdt / baseline_ct, CONSUMPTION_FLOOR_RATIO))
        else:
            log_prev_ratio = 0.0
        log_current_ratio = np.log(max(current_capacity / baseline_ct, CONSUMPTION_FLOOR_LABOUR_RATIO, 1e-9))
        log_expected_ratio = np.log(max(expected_capacity / baseline_ct, CONSUMPTION_FLOOR_RATIO, 1e-9))
        Cdt_new = baseline_ct * np.exp(self.rho1 * log_prev_ratio + self.rho0 / 2 * log_current_ratio + self.rho0 / 2 * log_expected_ratio)
        floor_consumption = min(baseline_ct * CONSUMPTION_FLOOR_RATIO, current_capacity)
        Cdt_new = min(max(Cdt_new, floor_consumption), current_capacity)
        cd = theta * Cdt_new * (1 - eps)
        
        return Cdt_new, cd
    
    def orders_O(self, A: np.ndarray, d: np.ndarray, tau: np.ndarray,
                 S_tar: np.ndarray, S: np.ndarray) -> np.ndarray:
        """Intermediate orders (O) combine current-use demand with damped replenishment toward the inventory target."""
        d = np.nan_to_num(d, nan=0.0, posinf=NUMERIC_LARGE, neginf=0.0)
        S_tar = np.nan_to_num(S_tar, nan=0.0, posinf=NUMERIC_LARGE, neginf=0.0)
        S = np.nan_to_num(S, nan=0.0, posinf=NUMERIC_LARGE, neginf=0.0)
        use_orders = A * d[np.newaxis, :]
        restock_gap = np.maximum(S_tar - S, 0.0)
        O = use_orders + restock_gap / tau[:, np.newaxis]
        return np.maximum(O, 0)

    def producing_x(self, prod_f: str, A_essential: Optional[np.ndarray], xcap0: np.ndarray,
                    l_: np.ndarray, S: np.ndarray, A: np.ndarray, d: np.ndarray, t: int) -> Dict:
        """Feasible output per sector is computed from labour capacity, beginning-of-period inventories, demand, and supplier-side output constraints."""
        with np.errstate(divide='ignore', invalid='ignore'):
            xcap = np.divide(l_[:, t], l_[:, 0], out=np.full_like(l_[:, t], np.inf), where=l_[:, 0] != 0) * xcap0
            xcap[l_[:, 0] == 0] = np.inf
        
        if prod_f in ["leontief", "leontief.adapted"]:
            xinp = np.zeros(self.N)
            for k in range(self.N):
                if prod_f == "leontief.adapted" and A_essential is not None:
                    input_capacity = np.divide(S[:, k], A[:, k], out=np.full(self.N, np.inf), where=A[:, k] > 0)
                    essential = (A_essential[:, k] > ESSENTIAL_INPUT_THRESHOLD) & (A[:, k] > 0)
                    nonessential = (A_essential[:, k] <= ESSENTIAL_INPUT_THRESHOLD) & (A[:, k] > 0)
                    essential_constraint = np.min(input_capacity[essential]) if np.any(essential) else np.inf
                    if np.any(nonessential):
                        weights = A[nonessential, k]
                        adaptable_constraint = float(np.average(input_capacity[nonessential], weights=weights))
                    else:
                        adaptable_constraint = np.inf
                    xinp[k] = min(essential_constraint, adaptable_constraint)
                else:
                    essential = A[:, k] > 0
                    if np.any(essential):
                        xinp[k] = np.min(S[essential, k] / A[essential, k])
                    else:
                        xinp[k] = np.inf
        
        elif prod_f == "linear":
            inpshare = np.sum(A, axis=0)
            totinp = np.sum(S, axis=0)
            xinp = np.where(inpshare > 0, totinp / inpshare, np.inf)
        elif prod_f == "ces":
            xinp = np.full(self.N, np.inf, dtype=np.float64)
            for k in range(self.N):
                nonzero_inputs = A[:, k] > 0
                if np.any(nonzero_inputs):
                    input_capacity = np.divide(S[nonzero_inputs, k], A[nonzero_inputs, k], out=np.full(np.count_nonzero(nonzero_inputs), np.inf), where=A[nonzero_inputs, k] > 0)
                    weights = A[nonzero_inputs, k]
                    xinp[k] = self._ces_output_constraint(input_capacity, weights)
        else:
            raise ValueError(f"Unknown production function: {prod_f}. Supported: 'leontief', 'leontief.adapted', 'linear', 'ces'")
        
        output_constraint = self._period_output_constraints(t)
        
        xcap = np.nan_to_num(xcap, nan=0.0, posinf=NUMERIC_LARGE, neginf=0.0)
        xinp = np.nan_to_num(xinp, nan=NUMERIC_LARGE, posinf=NUMERIC_LARGE, neginf=0.0)
        d = np.nan_to_num(d, nan=0.0, posinf=NUMERIC_LARGE, neginf=0.0)
        output_constraint = np.nan_to_num(output_constraint, nan=np.inf, posinf=np.inf, neginf=0.0)
        
        capacity_constraint = np.minimum(xcap, output_constraint)
        
        x = np.minimum(np.minimum(capacity_constraint, xinp), d)
        
        x_constraints = np.column_stack([xcap, xinp, d, output_constraint])
        
        return {'output': x, 'output.constraints': x_constraints}

    # --- Shock hooks (output cap, input-availability) ---
    def apply_output_constraint_shock(self, sector_label: str, time_period: int,
                                     reduction_pct: float, baseline_output: float = None) -> Tuple[int, float]:
        """An output cap is applied for the given sector and period. reduction_pct in [0, 1) scales the cap relative to baseline_output (default: x0 for that sector). The sector index and the constrained level are returned."""
        if sector_label not in self.label_to_index:
            raise ValueError(f"Sector label '{sector_label}' not found. Available labels: {list(self.label_to_index.keys())[:10]}...")
        
        sector_idx = self.label_to_index[sector_label]
        
        if time_period < 0 or time_period >= self.TT:
            raise ValueError(f"Time period {time_period} out of range [0, {self.TT-1}]")
        
        if reduction_pct < 0 or reduction_pct >= 1:
            raise ValueError(f"Reduction percentage must be in [0, 1). Got {reduction_pct}")
        
        if baseline_output is None:
            baseline_output = self.x0[sector_idx]
        
        constraint_level = (1 - reduction_pct) * baseline_output
        self.output_constraint_[sector_idx, time_period] = constraint_level
        
        return sector_idx, constraint_level
    
    def apply_input_availability_shock(self, input_sector_label: str, time_period: int,
                                       reduction_pct: float) -> Tuple[int, float]:
        """A supplier-side input-availability shock is recorded for the given sector and period.
        The shock constrains the supplier's output capacity for that period so downstream shortages arise via
        reduced deliveries and inventory drawdown rather than through deletion of downstream stocks."""
        if input_sector_label not in self.label_to_index:
            raise ValueError(f"Sector label '{input_sector_label}' not found. Available labels: {list(self.label_to_index.keys())[:10]}...")
        
        input_sector_idx = self.label_to_index[input_sector_label]
        
        if time_period < 0 or time_period >= self.TT:
            raise ValueError(f"Time period {time_period} out of range [0, {self.TT-1}]")
        
        if reduction_pct < 0 or reduction_pct >= 1:
            raise ValueError(f"Reduction percentage must be in [0, 1). Got {reduction_pct}")
        
        if time_period not in self.input_availability_shocks_:
            self.input_availability_shocks_[time_period] = {}
        
        self.input_availability_shocks_[time_period][input_sector_idx] = reduction_pct
        
        return input_sector_idx, reduction_pct
    
    def apply_rationing_shock(self, supplier_sector_label: str, time_period: int,
                              capacity_pct: float, include_households: bool = True) -> Tuple[int, float]:
        """A proportional rationing shock is applied to the given supplier sector for the specified period.
        The supplier's output is constrained to capacity_pct of baseline, and deliveries are scaled proportionally.
        The supplier sector index and capacity percentage are returned."""
        if supplier_sector_label not in self.label_to_index:
            raise ValueError(f"Sector label '{supplier_sector_label}' not found. Available labels: {list(self.label_to_index.keys())[:10]}...")
        
        supplier_idx = self.label_to_index[supplier_sector_label]
        
        if time_period < 0 or time_period >= self.TT:
            raise ValueError(f"Time period {time_period} out of range [0, {self.TT-1}]")
        
        if capacity_pct <= 0 or capacity_pct > 1:
            raise ValueError(f"Capacity percentage must be in (0, 1]. Got {capacity_pct}")
        
        if time_period not in self.rationing_shocks_:
            self.rationing_shocks_[time_period] = {}
        
        self.rationing_shocks_[time_period][supplier_idx] = {
            'capacity_pct': capacity_pct,
            'include_households': include_households
        }
        
        # Apply output constraint to enforce the capacity reduction
        self.apply_output_constraint_shock(supplier_sector_label, time_period, 1 - capacity_pct)
        
        return supplier_idx, capacity_pct

    # --- Intermediate consumption, final consumption, inventory, accounting ---
    def intercons_Z(self, O: np.ndarray, d: np.ndarray, x: np.ndarray,
                    firm_priority: str, S: np.ndarray, t: int = None) -> np.ndarray:
        """Actual intermediate deliveries Z are computed from orders O, demand d, and output x according to supplier-side rationing.
        Beginning-of-period inventories do not cap new receipts. When rationing shocks are active, deliveries from rationed suppliers are scaled proportionally."""
        if firm_priority == "no":
            s = np.divide(x, d, out=np.zeros_like(x), where=d!=0)
        else:
            denom = np.sum(O, axis=1)
            s = np.divide(x, denom, out=np.zeros_like(x), where=denom!=0)
            s = np.minimum(1, s)
        s = np.clip(s, 0, 1)
        desired_Z = O * s[:, np.newaxis]
        Z = desired_Z
        
        # Apply proportional rationing if active for this period
        if t is not None and t in self.rationing_shocks_:
            for supplier_idx, rationing_info in self.rationing_shocks_[t].items():
                capacity_pct = rationing_info['capacity_pct']
                # Proportional rationing: scale all deliveries from this supplier by capacity_pct
                Z[supplier_idx, :] = desired_Z[supplier_idx, :] * capacity_pct
        
        return Z
    
    def finalcons_c(self, cd: np.ndarray, d: np.ndarray, x: np.ndarray,
                    Z: np.ndarray, firm_priority: str, t: int = None) -> np.ndarray:
        """Realised consumption by sector is computed from desired consumption cd, demand d, output x, and intermediate Z according to firm_priority.
        When rationing shocks are active (t in rationing_shocks_), consumption from rationed suppliers is scaled proportionally."""
        domestic_final = cd
        if firm_priority == "no":
            s = np.divide(x, d, out=np.zeros_like(x), where=d!=0)
        else:
            numerator = x - np.sum(Z, axis=1)
            denominator = d - np.sum(Z, axis=1)
            s = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
            s = np.minimum(1, s)
        s = np.clip(s, 0, 1)
        c = domestic_final * s
        
        # Apply proportional rationing if active for this period
        if t is not None and t in self.rationing_shocks_:
            for supplier_idx, rationing_info in self.rationing_shocks_[t].items():
                if rationing_info['include_households']:
                    capacity_pct = rationing_info['capacity_pct']
                    if s[supplier_idx] > 0:
                        c[supplier_idx] = domestic_final[supplier_idx] * capacity_pct
                    else:
                        c[supplier_idx] = 0
        
        return c
    
    def inventory_S(self, x: np.ndarray, S: np.ndarray, Z: np.ndarray, A: np.ndarray) -> np.ndarray:
        """End-of-period inventories are updated from previous stock S, deliveries Z, and use A @ diag(x); non-negative values are returned."""
        x = np.nan_to_num(x, nan=0.0, posinf=NUMERIC_LARGE, neginf=0.0)
        S = np.nan_to_num(S, nan=0.0, posinf=NUMERIC_LARGE, neginf=0.0)
        Z = np.nan_to_num(Z, nan=0.0, posinf=NUMERIC_LARGE, neginf=0.0)
        used_inputs = A * x[np.newaxis, :]
        res = S + Z - used_inputs
        return np.maximum(0, res)

    def profit_pi(self, x: np.ndarray, Z: np.ndarray, l: np.ndarray) -> np.ndarray:
        """Profits by sector are computed as output minus intermediate inputs, labour, and cap/tax/import shares of output."""
        intermediate_inputs = np.sum(Z, axis=0)
        return x - intermediate_inputs - l - self.cap_share * x - self.tax_share * x - self.imp_share * x

    def savings_s(self, pi: np.ndarray, l: np.ndarray, c: np.ndarray, c_other_coef: float) -> float:
        """Aggregate savings are computed from household income net of consumption and other outlays."""
        extraexpenditure = self._extra_household_expenditure(np.sum(c))
        household_income = self._household_income(np.sum(l), np.sum(pi))
        return household_income - np.sum(c) - extraexpenditure

    # --- Simulation (time loop) ---
    def run_model(self) -> Dict:
        """The model is simulated over all periods. A dictionary is returned with gross_output, gdp, and realised_consumption (time series)."""
        s = np.zeros(self.TT)
        s[0] = self.savings_s(self.profits0, self.l0, self.c0, self.c_other_coef)
        pi_ = np.zeros((self.N, self.TT))
        pi_[:, 0] = self.profits0
        d_ = np.zeros((self.N, self.TT))
        d_[:, 0] = self.x0
        x_ = np.zeros((self.N, self.TT))
        x_[:, 0] = self.x0
        cd_ = np.zeros((self.N, self.TT))
        cd_[:, 0] = self.c0
        c_ = np.zeros((self.N, self.TT))
        c_[:, 0] = self.c0
        l_ = np.zeros((self.N, self.TT))
        l_[:, 0] = self.l0
        O = [self.Z0]
        Z = [self.Z0]
        S_tar = self.Z0 * self.n[np.newaxis, :]
        initial_inventory = np.array(S_tar, copy=True)
        S = [initial_inventory]
        initial_constraints = self.producing_x(self.prod_function, self.A_essential, self.x0, 
                                               l_, S[0], self.A, d_[:, 0], 0)['output.constraints']
        x_constraints = [initial_constraints]
        Cdt = np.zeros(self.TT)
        Cdt[0] = np.sum(self.c0)
        household_income_signal_ = np.zeros(self.TT)
        household_income_ = np.zeros(self.TT)
        household_income_signal_[0] = self._household_income(np.sum(self.l0), np.sum(self.profits0))
        household_income_[0] = self._household_income(np.sum(self.l0), np.sum(self.profits0))
        
        gdp_series = np.zeros(self.TT)
        gdp_series[0] = float(np.sum(x_[:, 0] - np.sum(Z[0], axis=0) - self.imp_share * x_[:, 0]))
        
        for t in range(1, self.TT):
            new_l = self.hire_fire(t, l_, x_, self.delta_, x_constraints[t-1])
            l_[:, t] = new_l
            household_income_signal_[t] = self._household_income_signal_for_period(household_income_[t-1], self.xi_[t])
            Cdt_new, cd_new = self.findemand_cd(
                self.theta_[:, t],
                Cdt[t-1],
                self.xi_[t],
                household_income_signal_[t],
                self.epsilon_[t]
            )
            Cdt[t] = Cdt_new
            cd_[:, t] = cd_new
            O.append(self.orders_O(self.A, d_[:, t-1], self.tau, S_tar, S[t-1]))
            d_[:, t] = (
                cd_[:, t]
                + np.sum(O[t], axis=1)
                + self.fd_government_[:, t]
                + self.fd_investment_[:, t]
                + self.fd_inventories_[:, t]
                + self.fd_exports_[:, t]
                + self.fd_other_[:, t]
            )
            
            prod = self.producing_x(self.prod_function, self.A_essential, self.x0, 
                                  l_, S[t-1], self.A, d_[:, t].copy(), t)
            x_[:, t] = prod['output']
            x_constraints.append(prod['output.constraints'])
            
            Z.append(self.intercons_Z(O[t], d_[:, t], x_[:, t], self.firm_priority, S[t-1], t))
            c_[:, t] = self.finalcons_c(cd_[:, t], d_[:, t], x_[:, t], Z[t], self.firm_priority, t)
            new_S = self.inventory_S(x_[:, t], S[t-1], Z[t], self.A)
            S.append(new_S)
            pi_[:, t] = self.profit_pi(x_[:, t], Z[t], l_[:, t])
            household_income_[t] = self._household_income(np.sum(l_[:, t]), np.sum(pi_[:, t]))
            s[t] = self.savings_s(pi_[:, t], l_[:, t], c_[:, t], self.c_other_coef)
            gdp_series[t] = float(np.sum(x_[:, t] - np.sum(Z[t], axis=0) - self.imp_share * x_[:, t]))
        
        return {
            'gross_output': x_,
            'gdp': gdp_series,
            'realised_consumption': c_,
            'savings': s,
            'household_income_signal': household_income_signal_,
            'household_income': household_income_,
            'household_closure_mode': self.household_closure_mode,
            'inventories': np.stack(S, axis=2),
            'orders': np.stack(O, axis=2),
            'intermediate_deliveries': np.stack(Z, axis=2),
        }

    # --- Plotting ---
    def plot_results(self, current_results: Dict, baseline_results: Dict = None,
                    title_suffix: str = "", save_path: str = None,
                    uncertainty_data: Dict = None) -> None:
        """Total output is plotted; if baseline_results is given, percentage change from baseline is shown. Uncertainty bands may be drawn from uncertainty_data. The figure is saved to save_path when provided."""
        fig, axes = plt.subplots(1, 1, figsize=(7.2, 8))
        if title_suffix:
            fig.suptitle(title_suffix, fontsize=14)
        
        start_period = 0
        time_slice = slice(start_period, None)
        time = np.arange(start_period, self.TT) - start_period
        current_output = np.sum(current_results['gross_output'], axis=0)[time_slice]
        if baseline_results is not None:
            baseline_output = np.sum(baseline_results['gross_output'], axis=0)[time_slice]
            output_change = ((current_output / baseline_output) - 1) * 100
        else:
            output_change = current_output
        
        if uncertainty_data is not None:
            self._plot_uncertainty_bands(axes, time, uncertainty_data, baseline_results, current_results, start_period=start_period)
        
        output_colour = '#2ca02c'
        axes.plot(time, output_change, color=output_colour, linewidth=2, label='Output')
        
        # axes.set_title(...) may be set here if required.
        
        if self.time_frequency == "daily":
            axes.set_xlabel('Time Period (Days)')
        elif self.time_frequency == "quarterly":
            axes.set_xlabel('Time Period (Quarters)')
        else:
            axes.set_xlabel('Time Period')
        axes.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        if time.size > 0:
            axes.set_xlim(time[0], time[-1])
        if baseline_results is not None:
            axes.set_ylabel('Percentage Change from Baseline (%)')
        else:
            axes.set_ylabel('Absolute Value')
        axes.grid(True, alpha=0.3)
        axes.legend(loc='best')
        if baseline_results is not None:
            axes.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _plot_uncertainty_bands(self, axes, time, uncertainty_data, baseline_results, current_results, start_period: int = 0, colour: str = None) -> None:
        """Uncertainty bands (e.g. 5–95 % and 25–75 %) are drawn from uncertainty_data. When baseline_results
        is provided, bands are percentage change from baseline under parameter uncertainty, shifted so the band
        is centred on the scenario path (current_results), i.e. 'parameter uncertainty around this scenario'.
        The bands reflect parameter uncertainty only; they may remain wide or widen after the shock has ended,
        since the sampled parameters govern dynamics in every period."""
        output_data = {}
        if 'gross_output' in uncertainty_data:
            gross_output_data = uncertainty_data['gross_output']
            if 'error' not in gross_output_data and isinstance(gross_output_data, dict) and 'mean' in gross_output_data:
                output_data = {
                    'mean': np.sum(gross_output_data['mean'], axis=0),
                    'q05': np.sum(gross_output_data['q05'], axis=0),
                    'q25': np.sum(gross_output_data['q25'], axis=0),
                    'q75': np.sum(gross_output_data['q75'], axis=0),
                    'q95': np.sum(gross_output_data['q95'], axis=0)
                }
        elif 'gdp' in uncertainty_data:
            output_data = uncertainty_data['gdp']
        
        band_colour = colour if colour is not None else '#2ca02c'
        variables = {
            'output': {'data': output_data, 'colour': band_colour, 'alpha': 0.12},
        }
        time_slice = slice(start_period, None) if start_period > 0 else slice(None)
        for var_name, var_info in variables.items():
            data = var_info['data']
            if 'error' in data or not data or len(data) == 0:
                continue
            
            if baseline_results is not None:
                baseline_values = np.sum(baseline_results['gross_output'], axis=0)[time_slice]
                current_values = np.sum(current_results['gross_output'], axis=0)[time_slice]
                mc_mean_values = data['mean'][time_slice] if 'mean' in data else np.zeros_like(baseline_values)
                
                current_pct = ((current_values / baseline_values) - 1) * 100
                mc_mean_pct = ((mc_mean_values / baseline_values) - 1) * 100
                mc_q05_pct = ((data['q05'][time_slice] / baseline_values) - 1) * 100
                mc_q25_pct = ((data['q25'][time_slice] / baseline_values) - 1) * 100
                mc_q75_pct = ((data['q75'][time_slice] / baseline_values) - 1) * 100
                mc_q95_pct = ((data['q95'][time_slice] / baseline_values) - 1) * 100
                
                offset = current_pct - mc_mean_pct
                
                q05_pct = mc_q05_pct + offset
                q25_pct = mc_q25_pct + offset
                q75_pct = mc_q75_pct + offset
                q95_pct = mc_q95_pct + offset
            else:
                current_values = np.sum(current_results['gross_output'], axis=0)[time_slice]
                mc_mean_values = data['mean'][time_slice]
                
                offset = current_values - mc_mean_values
                q05_pct = data['q05'][time_slice] + offset
                q25_pct = data['q25'][time_slice] + offset
                q75_pct = data['q75'][time_slice] + offset
                q95_pct = data['q95'][time_slice] + offset
            
            axes.fill_between(time, q05_pct, q95_pct, 
                              alpha=0.12, color=var_info['colour'], label='_nolegend_')
            axes.fill_between(time, q25_pct, q75_pct,
                              alpha=0.20, color=var_info['colour'], label='_nolegend_')


# -----------------------------------------------------------------------------
# Monte Carlo uncertainty analysis
# -----------------------------------------------------------------------------
class MonteCarloUncertaintyAnalysis:
    """Monte Carlo uncertainty analysis over parameter distributions. Parameter draws are taken from
    define_parameter_distributions; results and derived metrics are stored on the instance. Bounds
    for gamma_hire, gamma_fire, and tau match the model's validated ranges. Uncertainty bands
    derived from these results reflect parameter uncertainty only and may persist or widen after
    the shock has ended, since the same parameters govern dynamics in every period."""

    def __init__(self, base_model, n_simulations: int = 1000):
        """The base model and the number of simulations are stored; results are initialised as an empty dict."""
        self.base_model = base_model
        self.n_simulations = n_simulations
        self.results = {}

    def estimate_essential_inputs_from_io_data(self, A_matrix: np.ndarray, method: str = 'combined_linkage',
                                               value_threshold: float = 0.05, top_n: int = 3) -> np.ndarray:
        """An essential-input matrix is estimated from the IO matrix A_matrix using the given method (value, top_n, linkage, forward_linkage, combined_linkage, elasticity, or combined). Used when prod_function is leontief.adapted or ces."""
        N = A_matrix.shape[0]
        A_essential = np.zeros_like(A_matrix)
        
        if method == 'value':
            input_shares = A_matrix / (A_matrix.sum(axis=0) + 1e-10)
            A_essential = (input_shares >= value_threshold).astype(int)
        elif method == 'top_n':
            for j in range(N):
                if A_matrix[:, j].sum() > 0:
                    top_inputs = np.argsort(A_matrix[:, j])[-top_n:]
                    A_essential[top_inputs, j] = 1
                    
        elif method == 'linkage':
            try:
                I = np.eye(N)
                L = np.linalg.inv(I - A_matrix)
                backward_linkage = L.sum(axis=0)
                normalised_linkage = backward_linkage / (backward_linkage.mean() + 1e-10)
                for j in range(N):
                    A_essential[:, j] = (normalised_linkage > 1.5).astype(int)
            except np.linalg.LinAlgError:
                return self.estimate_essential_inputs_from_io_data(A_matrix, 'value')
                
        elif method == 'forward_linkage':
            try:
                I = np.eye(N)
                G = np.linalg.inv(I - A_matrix.T)
                forward_linkage = G.sum(axis=1)
                normalised_linkage = forward_linkage / (forward_linkage.mean() + 1e-10)
                
                for j in range(N):
                    A_essential[:, j] = (normalised_linkage > 1.5).astype(int)
                    
            except np.linalg.LinAlgError:
                return self.estimate_essential_inputs_from_io_data(A_matrix, 'value')
                
        elif method == 'combined_linkage':
            try:
                I = np.eye(N)
                L = np.linalg.inv(I - A_matrix)
                G = np.linalg.inv(I - A_matrix.T)
                
                backward_linkage = L.sum(axis=0)
                forward_linkage = G.sum(axis=1)
                
                norm_backward = backward_linkage / (backward_linkage.mean() + 1e-10)
                norm_forward = forward_linkage / (forward_linkage.mean() + 1e-10)
                
                combined_linkage = np.outer(norm_forward, norm_backward)
                threshold = 1.0
                A_essential = (combined_linkage > threshold).astype(int)
                
            except np.linalg.LinAlgError:
                return self.estimate_essential_inputs_from_io_data(A_matrix, 'value')
                
        elif method == 'elasticity':
            try:
                I = np.eye(N)
                L = np.linalg.inv(I - A_matrix)
                
                for i in range(N):
                    for j in range(N):
                        if A_matrix[i, j] > 0:
                            elasticity = A_matrix[i, j] * L[j, j] / (L[j, j] + 1e-10)
                            if elasticity > 0.1:
                                A_essential[i, j] = 1
            except np.linalg.LinAlgError:
                return self.estimate_essential_inputs_from_io_data(A_matrix, 'value')
                
        elif method == 'combined':
            try:
                value_importance = A_matrix / (A_matrix.sum(axis=0) + 1e-10)
                
                I = np.eye(N)
                L = np.linalg.inv(I - A_matrix)
                G = np.linalg.inv(I - A_matrix.T)
                
                backward_linkage = L.sum(axis=0)
                forward_linkage = G.sum(axis=1)
                
                norm_backward = backward_linkage / (backward_linkage.mean() + 1e-10)
                norm_forward = forward_linkage / (forward_linkage.mean() + 1e-10)
                
                linkage_importance = np.outer(norm_forward, norm_backward)
                
                elasticity_importance = A_matrix * L.T
                
                combined_score = (0.3 * value_importance + 
                                0.25 * linkage_importance +
                                0.25 * elasticity_importance +
                                0.2 * A_matrix)
                
                max_score = combined_score.max()
                if max_score > 0:
                    normalised_score = combined_score / max_score
                    A_essential = (normalised_score > 0.3).astype(int)
                else:
                    return self.estimate_essential_inputs_from_io_data(A_matrix, 'value')
                    
            except np.linalg.LinAlgError:
                return self.estimate_essential_inputs_from_io_data(A_matrix, 'value')
        
        else:
            raise ValueError(f"Unknown method: {method}")

        return A_essential

    def define_parameter_distributions(self) -> Dict[str, Any]:
        """Parameter names, distribution types, bounds, and short descriptions are returned for use in sampling.
        Bounds for gamma_hire, gamma_fire, and tau match the model's validated ranges so every draw is admissible."""
        return {
            'rho1': {
                'distribution': 'uniform',
                'bounds': (0.2, 0.9),
                'description': 'Consumption persistence (baseline: 0.6)'
            },
            'gamma_hire': {
                'distribution': 'uniform',
                'bounds': (GAMMA_HIRE_MIN, GAMMA_HIRE_MAX),
                'description': 'Labour hiring speed (baseline: ~0.30)'
            },
            'gamma_fire': {
                'distribution': 'uniform',
                'bounds': (GAMMA_FIRE_MIN, GAMMA_FIRE_MAX),
                'description': 'Labour firing speed (baseline: ~0.40)'
            },
            'tau': {
                'distribution': 'uniform',
                'bounds': (TAU_MIN, TAU_MAX),
                'description': 'Inventory adjustment speed (baseline: ~2.17)'
            },
            'savings_rate': {
                'distribution': 'uniform',
                'bounds': (0.01, 0.20),
                'description': 'Household savings rate (baseline: 0.05)'
            }
        }

    def sample_parameters(self, seed: int = None):
        """Parameters are sampled from the defined distributions; a dict of arrays (per simulation) is returned.
        The draw sequence is fixed by the random seed for reproducibility when this analysis is run in isolation."""
        if seed is not None:
            np.random.seed(seed)
            
        distributions = self.define_parameter_distributions()
        sampled_params = {}
        
        for param_name, param_info in distributions.items():
            if param_info['distribution'] == 'uniform':
                bounds = param_info['bounds']
                
                if param_name == 'rho1':
                    rho1_samples = np.random.uniform(bounds[0], bounds[1], size=self.n_simulations)
                    sampled_params['rho1'] = rho1_samples
                    sampled_params['rho0'] = 1 - rho1_samples
                elif param_name in ['gamma_fire', 'gamma_hire', 'tau']:
                    sampled_params[param_name] = np.random.uniform(
                        bounds[0], bounds[1], size=(self.base_model.N, self.n_simulations)
                    )
                elif param_name == 'savings_rate':
                    sampled_params[param_name] = np.random.uniform(
                        bounds[0], bounds[1], size=self.n_simulations
                    )

        return sampled_params

    def run_uncertainty_analysis(self, shock_scenario: str = "baseline",
                                 shock_params: dict = None, seed: int = 42):
        """Parameter distributions are sampled and the model is run for each draw; shock_scenario and shock_params
        are applied when not baseline. Results are stored on the instance and returned. Failed runs are filled
        with NaN and excluded from metrics; reproducibility for this analysis in isolation is ensured by the seed."""
        sampled_params = self.sample_parameters(seed)
        self.results = {
            'gdp': np.zeros((self.base_model.TT, self.n_simulations)),
            'consumption': np.zeros((self.base_model.TT, self.n_simulations)),
            'gross_output': np.zeros((self.base_model.TT, self.base_model.N, self.n_simulations)),
            'parameters': sampled_params
        }
        for i in range(self.n_simulations):
            model = self._create_model_with_parameters(sampled_params, i)
            if shock_scenario != "baseline":
                model = self._apply_shock(model, shock_scenario, shock_params)
            try:
                sim_results = model.run_model()
                self.results['gdp'][:, i] = sim_results['gdp']
                self.results['consumption'][:, i] = np.sum(sim_results['realised_consumption'], axis=0)
                self.results['gross_output'][:, :, i] = sim_results['gross_output'].T
            except Exception as e:
                logger.debug("Simulation %s failed: %s", i, e)
                self.results['gdp'][:, i] = np.nan
                self.results['consumption'][:, i] = np.nan
                self.results['gross_output'][:, :, i] = np.nan

        return self.results

    def _create_model_with_parameters(self, sampled_params: dict, simulation_idx: int) -> "SingleRegionInputOutputModel":
        """A model instance is created from the base model's config and parameterised from the sampled values at the given simulation index."""
        model = SingleRegionInputOutputModel(
            n_periods=self.base_model.TT,
            time_frequency=self.base_model.time_frequency,
            config=self.base_model.config.clone(),
        )
        if 'rho0' in sampled_params and 'rho1' in sampled_params:
            model.rho0 = sampled_params['rho0'][simulation_idx]
            model.rho1 = sampled_params['rho1'][simulation_idx]
        if 'gamma_hire' in sampled_params:
            model.gamma_hire = sampled_params['gamma_hire'][:, simulation_idx]
        if 'gamma_fire' in sampled_params:
            model.gamma_fire = sampled_params['gamma_fire'][:, simulation_idx]
        if 'tau' in sampled_params:
            model.tau = sampled_params['tau'][:, simulation_idx]
        if 'savings_rate' in sampled_params:
            model.savings_rate = model._validate_savings_rate_value(sampled_params['savings_rate'][simulation_idx])

        return model

    def _apply_shock(self, model: "SingleRegionInputOutputModel", shock_scenario: str, shock_params: dict) -> "SingleRegionInputOutputModel":
        """The consumption or input-availability shock is applied to the model in place when shock_scenario is \"consumption\" or \"input_availability\"; shock_params are used. The model is returned."""
        if shock_scenario == "consumption" and shock_params:
            intensity = shock_params.get('intensity', 0.2)
            duration = shock_params.get('duration', 3)
            start = shock_params.get('start', 2)
            for t in range(start, min(start + duration, model.TT)):
                model.epsilon_[t] = intensity
        elif shock_scenario == "input_availability" and shock_params:
            input_sector_label = shock_params.get('input_sector_label')
            reduction_pct = shock_params.get('reduction_pct', 0.3)
            duration = shock_params.get('duration', 3)
            start = shock_params.get('start', 2)
            if input_sector_label is not None:
                for t in range(start, min(start + duration, model.TT)):
                    model.apply_input_availability_shock(input_sector_label, t, reduction_pct)

        return model

    def calculate_uncertainty_metrics(self) -> Dict[str, Any]:
        """Mean, standard deviation, and selected quantiles are computed over valid simulations for gdp,
        consumption, and gross_output. A run is valid only if it has no NaN in any period."""
        metrics = {}
        for variable in ['gdp', 'consumption']:
            if variable not in self.results:
                continue
            data = self.results[variable]
            valid_mask = ~np.any(np.isnan(data), axis=0)
            valid_data = data[:, valid_mask]
            if valid_data.shape[1] == 0:
                metrics[variable] = {'error': 'No valid simulations'}
                continue
            metrics[variable] = {
                'mean': np.mean(valid_data, axis=1),
                'std': np.std(valid_data, axis=1),
                'q05': np.percentile(valid_data, 5, axis=1),
                'q25': np.percentile(valid_data, 25, axis=1),
                'q75': np.percentile(valid_data, 75, axis=1),
                'q95': np.percentile(valid_data, 95, axis=1),
                'n_valid': valid_data.shape[1]
            }
        
        if 'gross_output' in self.results:
            data = self.results['gross_output']
            valid_mask = ~np.any(np.isnan(data), axis=(0, 1))
            if np.sum(valid_mask) > 0:
                valid_data = data[:, :, valid_mask]
                if valid_data.shape[2] > 0:
                    # Transpose from (TT, N) to (N, TT) to match plotting expectations
                    metrics['gross_output'] = {
                        'mean': np.mean(valid_data, axis=2).T,
                        'std': np.std(valid_data, axis=2).T,
                        'q05': np.percentile(valid_data, 5, axis=2).T,
                        'q25': np.percentile(valid_data, 25, axis=2).T,
                        'q75': np.percentile(valid_data, 75, axis=2).T,
                        'q95': np.percentile(valid_data, 95, axis=2).T,
                        'n_valid': valid_data.shape[2]
                    }

        return metrics

    def get_uncertainty_data_for_plotting(self) -> Dict[str, Any]:
        """Uncertainty metrics (mean and quantiles) for GDP, consumption, and gross output are returned in the form expected by plot_results."""
        metrics = self.calculate_uncertainty_metrics()
        return metrics


# -----------------------------------------------------------------------------
# Example scenario runner
# -----------------------------------------------------------------------------
def run_consumption_shock_scenario(intensity: float = 0.2, duration: int = 3, start: int = 2,
                                   prod_function: str = "leontief",
                                   shock_spec: Optional[ConsumptionShockSpec] = None,
                                   household_closure_mode: str = "return_to_base"):
    """A consumption-shock scenario is run from an explicit shock spec or the provided arguments. By default this is the moderate example case."""
    spec = _resolve_consumption_shock_spec(intensity, duration, start, shock_spec)
    manager = ScenarioManager(ModelConfig(
        n_periods=SIMULATION_PERIODS,
        time_frequency="daily",
        prod_function=prod_function,
        household_closure_mode=household_closure_mode,
    ))
    manager.run_baseline(force=True)

    def consumption_shock(model: SingleRegionInputOutputModel) -> None:
        for t in range(spec.start, min(spec.start + spec.duration, model.TT)):
            model.epsilon_[t] = spec.intensity

    scenario = Scenario(
        name="consumption_shock",
        description=f"{spec.intensity * 100:.0f}% consumption shock for {spec.duration} periods ({spec.tier})",
        config=manager.base_config.clone(),
        shocks=[consumption_shock]
    )
    scenario_run, baseline_run = manager.run_scenario(scenario)
    return scenario_run, baseline_run


def _key_supplier_sector_label(model: "SingleRegionInputOutputModel") -> str:
    """The sector label with the largest forward supply (row sum of A) is returned, so that an input-availability shock on it is likely to bind for downstream sectors."""
    row_sums = np.sum(model.A, axis=1)
    idx = int(np.argmax(row_sums))
    return model.sector_labels[idx]


def _resolve_consumption_shock_spec(
    intensity: float,
    duration: int,
    start: int,
    shock_spec: Optional[ConsumptionShockSpec],
) -> ConsumptionShockSpec:
    """A consumption-shock spec is resolved from either an explicit spec or the function arguments."""
    if shock_spec is not None:
        return shock_spec
    return ConsumptionShockSpec(intensity=intensity, duration=duration, start=start, tier="example")


def _resolve_input_availability_shock_spec(
    reduction_pct: float,
    duration: int,
    start: int,
    inventory_days: Optional[np.ndarray],
    input_sector_label: Optional[str],
    shock_spec: Optional[InputAvailabilityShockSpec],
) -> InputAvailabilityShockSpec:
    """An input-availability shock spec is resolved from either an explicit spec or the function arguments."""
    if shock_spec is not None:
        return shock_spec
    return InputAvailabilityShockSpec(
        reduction_pct=reduction_pct,
        duration=duration,
        start=start,
        inventory_days=inventory_days,
        input_sector_label=input_sector_label,
        tier="example",
    )


def run_input_availability_shock_scenario(input_sector_label: Optional[str] = None,
                                          reduction_pct: float = INPUT_SHOCK_DEFAULT_REDUCTION_PCT,
                                          duration: int = INPUT_SHOCK_DEFAULT_DURATION,
                                          start: int = INPUT_SHOCK_DEFAULT_START,
                                          prod_function: str = "leontief",
                                          inventory_days: Optional[np.ndarray] = INPUT_SHOCK_DEFAULT_INVENTORY_DAYS,
                                          shock_spec: Optional[InputAvailabilityShockSpec] = None,
                                          household_closure_mode: str = "return_to_base"):
    """An input-availability shock scenario is run from an explicit shock spec or the provided arguments. By default this is the moderate example case, while the comparison helper uses a tighter stress spec."""
    spec = _resolve_input_availability_shock_spec(
        reduction_pct,
        duration,
        start,
        inventory_days,
        input_sector_label,
        shock_spec,
    )
    manager = ScenarioManager(ModelConfig(
        n_periods=SIMULATION_PERIODS,
        time_frequency="daily",
        prod_function=prod_function,
        inventory_days=spec.inventory_days,
        household_closure_mode=household_closure_mode,
    ))
    manager.run_baseline(force=True)
    baseline_run = manager.run_baseline(force=False)
    sector_label = spec.input_sector_label if spec.input_sector_label is not None else _key_supplier_sector_label(baseline_run.model)

    def input_availability_shock(model: SingleRegionInputOutputModel) -> None:
        for t in range(spec.start, min(spec.start + spec.duration, model.TT)):
            model.apply_input_availability_shock(sector_label, t, spec.reduction_pct)

    scenario = Scenario(
        name="input_availability_shock",
        description=f"{spec.reduction_pct * 100:.0f}% input-availability shock ({sector_label}) for {spec.duration} periods ({spec.tier})",
        config=manager.base_config.clone(),
        shocks=[input_availability_shock]
    )
    scenario_run, baseline_run = manager.run_scenario(scenario)
    return scenario_run, baseline_run


def _resolve_household_closure_modes(closure_modes: Optional[List[str]]) -> List[str]:
    """A validated list of household-closure modes is returned."""
    modes = list(HOUSEHOLD_CLOSURE_MODES) if closure_modes is None else list(closure_modes)
    invalid_modes = [mode for mode in modes if mode not in HOUSEHOLD_CLOSURE_MODES]
    if invalid_modes:
        raise ValueError(f"Unknown household closure modes: {invalid_modes}. Supported: {HOUSEHOLD_CLOSURE_MODES}")
    return modes


def _closure_mode_label(closure_mode: str) -> str:
    """A presentation label for the given household closure mode is returned."""
    return {
        "return_to_base": "Return To Base",
        "scarred": "Permanent Scarring",
    }[closure_mode]


def _plot_household_closure_comparison(results_data: Dict[str, Dict[str, Any]], save_path: Optional[str] = None) -> None:
    """Scenario paths and Monte Carlo bands are plotted for each household closure mode."""
    if not results_data:
        logger.error("No closure-mode results to plot")
        return

    fig, axes = plt.subplots(1, 1, figsize=(7.2, 8))
    colours = {
        "return_to_base": "#1f77b4",
        "scarred": "#d62728",
    }
    linestyles = {
        "return_to_base": "-",
        "scarred": "--",
    }

    first_mode = next(iter(results_data))
    first_data = results_data[first_mode]
    time = np.arange(first_data["model"].TT)

    for closure_mode, data in results_data.items():
        colour = colours.get(closure_mode, "#000000")
        linestyle = linestyles.get(closure_mode, "-")
        if data["uncertainty_data"] is not None:
            data["model"]._plot_uncertainty_bands(
                axes,
                time,
                data["uncertainty_data"],
                data["baseline_results"],
                data["scenario_results"],
                colour=colour,
            )
        current_output = np.sum(data["scenario_results"]["gross_output"], axis=0)
        baseline_output = np.sum(data["baseline_results"]["gross_output"], axis=0)
        output_change = ((current_output / baseline_output) - 1) * 100
        axes.plot(
            time,
            output_change,
            color=colour,
            linewidth=2.5,
            linestyle=linestyle,
            label=_closure_mode_label(closure_mode),
        )

    if first_data["model"].time_frequency == "daily":
        axes.set_xlabel("Time Period (Days)")
    elif first_data["model"].time_frequency == "quarterly":
        axes.set_xlabel("Time Period (Quarters)")
    else:
        axes.set_xlabel("Time Period")
    axes.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    if time.size > 0:
        axes.set_xlim(time[0], time[-1])
    axes.set_ylabel("Percentage Change from Baseline (%)")
    axes.grid(True, alpha=0.3)
    axes.legend(loc="best")
    axes.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Figure saved: %s", save_path)
    plt.show()


def run_consumption_shock_household_closure_sensitivity(
    intensity: float = 0.2,
    duration: int = 3,
    start: int = 2,
    prod_function: str = "leontief",
    closure_modes: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    n_simulations: int = MC_PLOT_SIMULATIONS,
) -> Dict[str, Dict[str, Any]]:
    """The consumption shock is run for each household closure mode, with separate Monte Carlo bands for structural sensitivity."""
    modes = _resolve_household_closure_modes(closure_modes)
    results_data: Dict[str, Dict[str, Any]] = {}
    for closure_mode in modes:
        scenario_run, baseline_run = run_consumption_shock_scenario(
            intensity=intensity,
            duration=duration,
            start=start,
            prod_function=prod_function,
            household_closure_mode=closure_mode,
        )
        mc_shock = MonteCarloUncertaintyAnalysis(baseline_run.model, n_simulations=n_simulations)
        mc_shock.run_uncertainty_analysis(
            shock_scenario="consumption",
            shock_params={"intensity": intensity, "duration": duration, "start": start},
            seed=42,
        )
        results_data[closure_mode] = {
            "scenario_results": scenario_run.results,
            "baseline_results": baseline_run.results,
            "uncertainty_data": mc_shock.get_uncertainty_data_for_plotting(),
            "model": scenario_run.model,
        }

    _plot_household_closure_comparison(results_data, save_path=save_path)
    return results_data


def run_input_availability_shock_household_closure_sensitivity(
    input_sector_label: Optional[str] = None,
    reduction_pct: float = INPUT_SHOCK_DEFAULT_REDUCTION_PCT,
    duration: int = INPUT_SHOCK_DEFAULT_DURATION,
    start: int = INPUT_SHOCK_DEFAULT_START,
    prod_function: str = "leontief",
    inventory_days: Optional[np.ndarray] = INPUT_SHOCK_DEFAULT_INVENTORY_DAYS,
    shock_spec: Optional[InputAvailabilityShockSpec] = None,
    closure_modes: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    n_simulations: int = MC_PLOT_SIMULATIONS,
) -> Dict[str, Dict[str, Any]]:
    """The input-availability shock is run for each household closure mode, with separate Monte Carlo bands for structural sensitivity."""
    spec = _resolve_input_availability_shock_spec(
        reduction_pct,
        duration,
        start,
        inventory_days,
        input_sector_label,
        shock_spec,
    )
    modes = _resolve_household_closure_modes(closure_modes)
    results_data: Dict[str, Dict[str, Any]] = {}
    sector_label = spec.input_sector_label

    for closure_mode in modes:
        scenario_run, baseline_run = run_input_availability_shock_scenario(
            prod_function=prod_function,
            shock_spec=InputAvailabilityShockSpec(
                reduction_pct=spec.reduction_pct,
                duration=spec.duration,
                start=spec.start,
                inventory_days=spec.inventory_days,
                input_sector_label=sector_label,
                tier=spec.tier,
            ),
            household_closure_mode=closure_mode,
        )
        if sector_label is None:
            sector_label = _key_supplier_sector_label(baseline_run.model)
        mc_shock = MonteCarloUncertaintyAnalysis(baseline_run.model, n_simulations=n_simulations)
        mc_shock.run_uncertainty_analysis(
            shock_scenario="input_availability",
            shock_params={
                "input_sector_label": sector_label,
                "reduction_pct": spec.reduction_pct,
                "duration": spec.duration,
                "start": spec.start,
            },
            seed=42,
        )
        results_data[closure_mode] = {
            "scenario_results": scenario_run.results,
            "baseline_results": baseline_run.results,
            "uncertainty_data": mc_shock.get_uncertainty_data_for_plotting(),
            "model": scenario_run.model,
        }

    _plot_household_closure_comparison(results_data, save_path=save_path)
    return results_data


def run_consumption_shock_all_prod_functions(intensity: float = 0.2, duration: int = 3, start: int = 2,
                                             save_path: str = None, n_simulations: int = MC_PLOT_SIMULATIONS,
                                             household_closure_mode: str = "return_to_base"):
    """The consumption shock scenario is run for all production function settings with Monte Carlo uncertainty analysis, and results are plotted together in one figure using existing plotting functionality."""
    production_functions = ["leontief", "leontief.adapted", "linear", "ces"]
    results_data = {}
    time_frequency = None
    
    # The scenario is run for each production function
    for prod_func in production_functions:
        logger.debug("Running consumption shock scenario with production function: %s", prod_func)
        try:
            scenario_run, baseline_run = run_consumption_shock_scenario(
                intensity=intensity, duration=duration, start=start, prod_function=prod_func,
                household_closure_mode=household_closure_mode,
            )
            
            # Production functions differ only when input availability becomes binding through the inventory channel.
            # With a consumption shock (reduced demand), demand becomes binding, so all production functions
            # produce identical results. This is expected behaviour; they should overlap in this scenario.
            
            # Time frequency and model are stored from the first successful run
            if time_frequency is None:
                time_frequency = scenario_run.model.time_frequency
            
            # Monte Carlo uncertainty analysis is run
            logger.debug("Running Monte Carlo analysis for %s (%s simulations)", prod_func, n_simulations)
            mc_shock = MonteCarloUncertaintyAnalysis(baseline_run.model, n_simulations=n_simulations)
            mc_shock.run_uncertainty_analysis(
                shock_scenario="consumption", 
                shock_params={'intensity': intensity, 'duration': duration, 'start': start}, 
                seed=42
            )
            shock_uncertainty = mc_shock.get_uncertainty_data_for_plotting()
            
            results_data[prod_func] = {
                'scenario_results': scenario_run.results,
                'baseline_results': baseline_run.results,
                'uncertainty_data': shock_uncertainty,
                'model': scenario_run.model
            }
        except Exception as e:
            logger.debug("Failed to run scenario for production function %s: %s", prod_func, e)
            import traceback
            logger.debug(traceback.format_exc())
            continue
    
    # All production functions are plotted in one figure using existing plotting functionality
    if not results_data:
        logger.error("No results to plot")
        return
    
    # Figure and axes are created
    fig, axes = plt.subplots(1, 1, figsize=(7.2, 8))
    
    # Colour palette and line styles are defined for different production functions
    colours = {
        'leontief': '#2ca02c',
        'leontief.adapted': '#1f77b4',
        'linear': '#ff7f0e',
        'ces': '#d62728'
    }
    
    linestyles = {
        'leontief': '-',
        'leontief.adapted': '--',
        'linear': '-.',
        'ces': ':'
    }
    
    # Time axis is extracted from the first result
    start_period = 0
    time_slice = slice(start_period, None)
    first_prod_func = list(results_data.keys())[0]
    first_data = results_data[first_prod_func]
    time = np.arange(start_period, first_data['model'].TT) - start_period
    
    # Uncertainty bands and lines are plotted for each production function
    for prod_func, data in results_data.items():
        colour = colours.get(prod_func, '#000000')
        linestyle = linestyles.get(prod_func, '-')
        
        # Existing _plot_uncertainty_bands method is used with production function-specific colour
        if data['uncertainty_data'] is not None:
            data['model']._plot_uncertainty_bands(
                axes, time, data['uncertainty_data'], 
                data['baseline_results'], data['scenario_results'],
                start_period=start_period, colour=colour
            )
        
        # Main line is plotted
        current_output = np.sum(data['scenario_results']['gross_output'], axis=0)[time_slice]
        baseline_output = np.sum(data['baseline_results']['gross_output'], axis=0)[time_slice]
        output_change = ((current_output / baseline_output) - 1) * 100
        axes.plot(time, output_change, color=colour, linewidth=2.5, linestyle=linestyle,
                 label=prod_func.replace('.', ' ').title())
    
    # Labels and formatting are set
    if time_frequency == "daily":
        axes.set_xlabel('Time Period (Days)')
    elif time_frequency == "quarterly":
        axes.set_xlabel('Time Period (Quarters)')
    else:
        axes.set_xlabel('Time Period')
    axes.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    if time.size > 0:
        axes.set_xlim(time[0], time[-1])
    axes.set_ylabel('Percentage Change from Baseline (%)')
    axes.grid(True, alpha=0.3)
    axes.legend(loc='best')
    axes.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info("Figure saved: %s", save_path)
    plt.show()


def run_input_availability_shock_all_prod_functions(input_sector_label: Optional[str] = None,
                                                     reduction_pct: float = INPUT_SHOCK_STRESS_REDUCTION_PCT,
                                                     duration: int = INPUT_SHOCK_STRESS_DURATION,
                                                     start: int = INPUT_SHOCK_STRESS_START,
                                                     inventory_days: Optional[np.ndarray] = INPUT_SHOCK_STRESS_INVENTORY_DAYS,
                                                     save_path: str = None,
                                                     n_simulations: int = MC_PLOT_SIMULATIONS,
                                                     shock_spec: Optional[InputAvailabilityShockSpec] = None,
                                                     household_closure_mode: str = "return_to_base"):
    """The production-function comparison is run on a tighter stress-style input shock by default so substitution differences become visible rather than being hidden by generous inventories."""
    spec = _resolve_input_availability_shock_spec(
        reduction_pct,
        duration,
        start,
        inventory_days,
        input_sector_label,
        shock_spec if shock_spec is not None else INPUT_AVAILABILITY_STRESS_SHOCK_SPEC,
    )
    production_functions = ["leontief", "leontief.adapted", "linear", "ces"]
    results_data = {}
    time_frequency = None
    sector_label = spec.input_sector_label

    # The scenario is run for each production function
    for prod_func in production_functions:
        logger.debug("Running input-availability shock scenario with production function: %s", prod_func)
        try:
            scenario_run, baseline_run = run_input_availability_shock_scenario(
                prod_function=prod_func,
                shock_spec=InputAvailabilityShockSpec(
                    reduction_pct=spec.reduction_pct,
                    duration=spec.duration,
                    start=spec.start,
                    inventory_days=spec.inventory_days,
                    input_sector_label=sector_label,
                    tier=spec.tier,
                ),
                household_closure_mode=household_closure_mode,
            )
            if sector_label is None:
                sector_label = _key_supplier_sector_label(baseline_run.model)

            # Time frequency and model are stored from the first successful run
            if time_frequency is None:
                time_frequency = scenario_run.model.time_frequency

            # Monte Carlo uncertainty analysis is run
            logger.debug("Running Monte Carlo analysis for %s (%s simulations)", prod_func, n_simulations)
            mc_shock = MonteCarloUncertaintyAnalysis(baseline_run.model, n_simulations=n_simulations)
            mc_shock.run_uncertainty_analysis(
                shock_scenario="input_availability",
                shock_params={'input_sector_label': sector_label, 'reduction_pct': spec.reduction_pct,
                              'duration': spec.duration, 'start': spec.start},
                seed=42
            )
            shock_uncertainty = mc_shock.get_uncertainty_data_for_plotting()

            results_data[prod_func] = {
                'scenario_results': scenario_run.results,
                'baseline_results': baseline_run.results,
                'uncertainty_data': shock_uncertainty,
                'model': scenario_run.model
            }
        except Exception as e:
            logger.debug("Failed to run scenario for production function %s: %s", prod_func, e)
            import traceback
            logger.debug(traceback.format_exc())
            continue

    # All production functions are plotted in one figure using existing plotting functionality
    if not results_data:
        logger.error("No results to plot")
        return

    # Figure and axes are created
    fig, axes = plt.subplots(1, 1, figsize=(7.2, 8))

    # Colour palette and line styles are defined for different production functions
    colours = {
        'leontief': '#2ca02c',
        'leontief.adapted': '#1f77b4',
        'linear': '#ff7f0e',
        'ces': '#d62728'
    }
    linestyles = {
        'leontief': '-',
        'leontief.adapted': '--',
        'linear': '-.',
        'ces': ':'
    }

    # Time axis is extracted from the first result
    start_period = 0
    time_slice = slice(start_period, None)
    first_prod_func = list(results_data.keys())[0]
    first_data = results_data[first_prod_func]
    time = np.arange(start_period, first_data['model'].TT) - start_period

    # Uncertainty bands and lines are plotted for each production function
    for prod_func, data in results_data.items():
        colour = colours.get(prod_func, '#000000')
        linestyle = linestyles.get(prod_func, '-')
        if data['uncertainty_data'] is not None:
            data['model']._plot_uncertainty_bands(
                axes, time, data['uncertainty_data'],
                data['baseline_results'], data['scenario_results'],
                start_period=start_period, colour=colour
            )
        current_output = np.sum(data['scenario_results']['gross_output'], axis=0)[time_slice]
        baseline_output = np.sum(data['baseline_results']['gross_output'], axis=0)[time_slice]
        output_change = ((current_output / baseline_output) - 1) * 100
        axes.plot(time, output_change, color=colour, linewidth=2.5, linestyle=linestyle,
                 label=prod_func.replace('.', ' ').title())

    # Labels and formatting are set
    if time_frequency == "daily":
        axes.set_xlabel('Time Period (Days)')
    elif time_frequency == "quarterly":
        axes.set_xlabel('Time Period (Quarters)')
    else:
        axes.set_xlabel('Time Period')
    axes.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    if time.size > 0:
        axes.set_xlim(time[0], time[-1])
    axes.set_ylabel('Percentage Change from Baseline (%)')
    axes.grid(True, alpha=0.3)
    axes.legend(loc='best')
    axes.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info("Figure saved: %s", save_path)
    plt.show()


def run_input_availability_sensitivity_panel(input_sector_label: Optional[str] = None,
                                             prod_function: str = "leontief",
                                             save_path: Optional[str] = None) -> None:
    """A small sensitivity panel is run for the input-availability shock so the moderate example is contextualised against a tighter stress case and higher-inventory variants."""
    cases = [
        {
            'label': 'Stress Test (50%, 1 Day)',
            'colour': '#2ca02c',
            'linestyle': '-',
            'reduction_pct': INPUT_SHOCK_STRESS_REDUCTION_PCT,
            'inventory_days': INPUT_SHOCK_STRESS_INVENTORY_DAYS,
        },
        {
            'label': 'Moderate Shock (30%, 1 Day)',
            'colour': '#1f77b4',
            'linestyle': '--',
            'reduction_pct': INPUT_SHOCK_DEFAULT_REDUCTION_PCT,
            'inventory_days': INPUT_SHOCK_STRESS_INVENTORY_DAYS,
        },
        {
            'label': 'Stress + Buffer (50%, 5 Days)',
            'colour': '#ff7f0e',
            'linestyle': '-.',
            'reduction_pct': INPUT_SHOCK_STRESS_REDUCTION_PCT,
            'inventory_days': INPUT_SHOCK_DEFAULT_INVENTORY_DAYS,
        },
        {
            'label': 'Moderate + Buffer (30%, 5 Days)',
            'colour': '#d62728',
            'linestyle': ':',
            'reduction_pct': INPUT_SHOCK_DEFAULT_REDUCTION_PCT,
            'inventory_days': INPUT_SHOCK_DEFAULT_INVENTORY_DAYS,
        },
    ]

    fig, axes = plt.subplots(1, 1, figsize=(7.2, 8))
    time = None
    time_frequency = None
    for case in cases:
        scenario_run, baseline_run = run_input_availability_shock_scenario(
            input_sector_label=input_sector_label,
            reduction_pct=case['reduction_pct'],
            duration=INPUT_SHOCK_DEFAULT_DURATION,
            start=INPUT_SHOCK_DEFAULT_START,
            prod_function=prod_function,
            inventory_days=case['inventory_days'],
        )
        if time is None:
            time = np.arange(scenario_run.model.TT)
            time_frequency = scenario_run.model.time_frequency
        current_output = np.sum(scenario_run.results['gross_output'], axis=0)
        baseline_output = np.sum(baseline_run.results['gross_output'], axis=0)
        output_change = ((current_output / baseline_output) - 1) * 100
        axes.plot(
            time,
            output_change,
            color=case['colour'],
            linewidth=2.5,
            linestyle=case['linestyle'],
            label=case['label'],
        )

    if time_frequency == "daily":
        axes.set_xlabel('Time Period (Days)')
    elif time_frequency == "quarterly":
        axes.set_xlabel('Time Period (Quarters)')
    else:
        axes.set_xlabel('Time Period')
    axes.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    if time is not None and time.size > 0:
        axes.set_xlim(time[0], time[-1])
    axes.set_ylabel('Percentage Change from Baseline (%)')
    axes.grid(True, alpha=0.3)
    axes.legend(loc='best')
    axes.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info("Figure saved: %s", save_path)
    plt.show()


# -----------------------------------------------------------------------------
# Entry point (when the file is run as the main script)
# -----------------------------------------------------------------------------
# A baseline scenario is executed with the default config (daily frequency,
# Leontief production). A small Monte Carlo uncertainty
# run is performed; when ENABLE_PLOTTING is True, total-output results are
# plotted to figures/baseline.png and a single baseline GDP value is logged.
# When run as main, only warnings and errors are logged to the terminal.
if __name__ == "__main__":
    logging.getLogger().setLevel(logging.WARNING)
    manager = ScenarioManager(ModelConfig(n_periods=SIMULATION_PERIODS, time_frequency="daily", prod_function="leontief"))
    baseline_run = manager.run_baseline(force=True)

    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Baseline plot: single path, no uncertainty bands. Baseline replication = one calibration, no shock.
    if ENABLE_PLOTTING:
        baseline_run.model.plot_results(
            baseline_run.results,
            baseline_results=None,
            title_suffix="(Daily Baseline)",
            save_path=str(figures_dir / "baseline.png"),
            uncertainty_data=None
        )

    # Consumption-shock example: run for all production functions and plot together
    if ENABLE_PLOTTING:
        run_consumption_shock_all_prod_functions(
            intensity=0.2, duration=3, start=2,
            save_path=str(figures_dir / "consumption_shock_all_prod_functions.png"),
            n_simulations=MC_PLOT_SIMULATIONS
        )
        run_consumption_shock_household_closure_sensitivity(
            intensity=0.2,
            duration=3,
            start=2,
            prod_function="leontief",
            save_path=str(figures_dir / "consumption_shock_household_closure_sensitivity.png"),
            n_simulations=MC_PLOT_SIMULATIONS,
        )

    # Input-availability comparison: use the stress-tier helper so production-function differences are visible
    if ENABLE_PLOTTING and ENABLE_INPUT_AVAILABILITY_SHOCK_PLOT:
        run_input_availability_shock_all_prod_functions(
            input_sector_label=None,
            save_path=str(figures_dir / "input_availability_shock_all_prod_functions.png"),
            n_simulations=MC_PLOT_SIMULATIONS
        )
        run_input_availability_shock_household_closure_sensitivity(
            input_sector_label=None,
            prod_function="leontief",
            save_path=str(figures_dir / "input_availability_shock_household_closure_sensitivity.png"),
            n_simulations=MC_PLOT_SIMULATIONS,
        )
        run_input_availability_sensitivity_panel(
            input_sector_label=None,
            prod_function="leontief",
            save_path=str(figures_dir / "input_availability_sensitivity_panel.png"),
        )
