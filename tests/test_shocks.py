"""Tests for shock application methods of InputOutputModel.

Covers consumption, input availability, technical change, factor productivity, and rationing shocks.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyMacroIO.model import InputOutputModel
from pyMacroIO.config import ModelConfig


# Fresh-model helper
def _fresh_model(data_dict: dict) -> InputOutputModel:
    """Return a freshly constructed InputOutputModel from the given data dictionary."""
    config = ModelConfig(n_periods=10, time_frequency="quarterly")
    return InputOutputModel(
        n_periods=10,
        time_frequency="quarterly",
        config=config,
        _data_dict=data_dict,
    )


# Rationing branch
class TestRationingBranch:
    """Verifies the rationing branch of intercons_Z and finalcons_c, active when t is in rationing_shocks_."""

    def test_rationing_overrides_intermediate_deliveries(
        self, minimal_data_dict: dict
    ) -> None:
        """Verifies that Z[0,:] equals Z0[0,:]*capacity_pct when a rationing shock is registered for t."""
        m = _fresh_model(minimal_data_dict)
        m.apply_rationing_shock("A", time_period=5, capacity_pct=0.5)
        Z = m.intercons_Z(
            O=m.Z0, d=m.x0, x=m.x0, firm_priority="no", S=m.Z0, t=5
        )
        np.testing.assert_allclose(Z[0, :], m.Z0[0, :] * 0.5, rtol=1e-10)

    def test_rationing_leaves_other_sectors_unaffected_in_Z(
        self, minimal_data_dict: dict
    ) -> None:
        """Verifies that sectors not covered by the rationing shock are unaffected in Z."""
        m = _fresh_model(minimal_data_dict)
        m.apply_rationing_shock("A", time_period=5, capacity_pct=0.5)
        Z = m.intercons_Z(
            O=m.Z0, d=m.x0, x=m.x0, firm_priority="no", S=m.Z0, t=5
        )
        # With x=x0 and d=x0, s=1 and the normal formula gives Z=Z0 for unshocked rows.
        np.testing.assert_allclose(Z[1, :], m.Z0[1, :], rtol=1e-10)
        np.testing.assert_allclose(Z[2, :], m.Z0[2, :], rtol=1e-10)

    def test_no_rationing_at_unregistered_period_leaves_Z_unchanged(
        self, minimal_data_dict: dict
    ) -> None:
        """Checks that passing an unregistered period t leaves Z equal to the no-rationing result."""
        m = _fresh_model(minimal_data_dict)
        m.apply_rationing_shock("A", time_period=5, capacity_pct=0.5)
        # t=3 has no registered shock, so the branch is not entered.
        Z = m.intercons_Z(
            O=m.Z0, d=m.x0, x=m.x0, firm_priority="no", S=m.Z0, t=3
        )
        np.testing.assert_allclose(Z, m.Z0, rtol=1e-10)

    def test_rationing_overrides_household_consumption(
        self, minimal_data_dict: dict
    ) -> None:
        """Verifies that c[0] equals c0[0]*capacity_pct when include_households=True."""
        m = _fresh_model(minimal_data_dict)
        m.apply_rationing_shock("A", time_period=5, capacity_pct=0.5, include_households=True)
        Z = m.intercons_Z(
            O=m.Z0, d=m.x0, x=m.x0, firm_priority="no", S=m.Z0, t=5
        )
        c = m.finalcons_c(
            cd=m.c0, d=m.x0, x=m.x0, Z=Z, firm_priority="no", t=5
        )
        assert np.isclose(c[0], m.c0[0] * 0.5, rtol=1e-10)
        np.testing.assert_allclose(c[1:], m.c0[1:], rtol=1e-10)

    def test_rationing_without_households_does_not_override_c(
        self, minimal_data_dict: dict
    ) -> None:
        """Checks that finalcons_c leaves household consumption at its normal level when include_households=False."""
        m = _fresh_model(minimal_data_dict)
        m.apply_rationing_shock("A", time_period=5, capacity_pct=0.5, include_households=False)
        Z = m.intercons_Z(
            O=m.Z0, d=m.x0, x=m.x0, firm_priority="no", S=m.Z0, t=5
        )
        np.testing.assert_allclose(Z[0, :], m.Z0[0, :] * 0.5, rtol=1e-10)
        c = m.finalcons_c(
            cd=m.c0, d=m.x0, x=m.x0, Z=Z, firm_priority="no", t=5
        )
        assert np.isclose(c[0], m.c0[0], rtol=1e-5)


# Consumption shock
class TestConsumptionShock:
    """Verifies that apply_consumption_shock reduces demand and GDP during the shock window."""

    def test_consumption_shock_reduces_gdp_in_shocked_window(
        self, minimal_data_dict: dict
    ) -> None:
        """Checks that GDP falls in the shock window relative to the pre-shock period."""
        m = _fresh_model(minimal_data_dict)
        m.apply_consumption_shock(start=3, duration=3, intensity=0.5)
        result = m.run_model()
        assert result["gdp"][3] < result["gdp"][0]

    def test_consumption_shock_gdp_recovers_after_window(
        self, minimal_data_dict: dict
    ) -> None:
        """Checks that GDP rises after the shock window ends relative to the shocked period."""
        m = _fresh_model(minimal_data_dict)
        m.apply_consumption_shock(start=3, duration=3, intensity=0.5)
        result = m.run_model()
        assert result["gdp"][9] > result["gdp"][3]

    def test_consumption_shock_reduces_realised_consumption(
        self, minimal_data_dict: dict
    ) -> None:
        """Checks that total realised consumption is lower in the shocked period than at t=0."""
        m = _fresh_model(minimal_data_dict)
        m.apply_consumption_shock(start=3, duration=3, intensity=0.5)
        result = m.run_model()
        assert (
            np.sum(result["realised_consumption"][:, 3])
            < np.sum(result["realised_consumption"][:, 0])
        )


# Input availability shock
class TestInputAvailabilityShock:
    """Verifies that apply_input_availability_shock constrains the affected sector's output."""

    def test_input_availability_shock_caps_output_in_shocked_period(
        self, minimal_data_dict: dict
    ) -> None:
        """Checks that gross output for the affected sector does not exceed (1-reduction_pct)*x0 at the shocked period."""
        m = _fresh_model(minimal_data_dict)
        m.apply_input_availability_shock("A", time_period=4, reduction_pct=0.5)
        result = m.run_model()
        assert result["gross_output"][0, 4] <= m.x0[0] * 0.5 + 1e-6

    def test_input_availability_shock_unaffected_period_has_normal_output(
        self, minimal_data_dict: dict
    ) -> None:
        """Checks that gross output for the affected sector is near x0 in an unshocked period."""
        m = _fresh_model(minimal_data_dict)
        m.apply_input_availability_shock("A", time_period=4, reduction_pct=0.5)
        result = m.run_model()
        assert result["gross_output"][0, 3] > m.x0[0] * 0.95


# Technical change
class TestTechnicalChange:
    """Verifies that apply_technical_change alters the steady state from the changed period."""

    def test_halving_A_reduces_intermediate_costs_from_changed_period(
        self, minimal_data_dict: dict
    ) -> None:
        """Verifies that halving the technical coefficient matrix from t=5 reduces intermediate deliveries."""
        m = _fresh_model(minimal_data_dict)
        new_A = m.A * 0.5
        m.apply_technical_change(t=5, new_A=new_A)
        result = m.run_model()
        # Total intermediate inputs (column sums of Z) should fall after the A change.
        # Compare the last period (t=9) against t=4 (just before the change).
        assert result["Z_colsums"][:, 9].sum() < result["Z_colsums"][:, 4].sum() * 0.9, (
            "Total intermediate inputs should be substantially lower after halving A."
        )

    def test_invalid_A_column_sum_raises_value_error(
        self, minimal_data_dict: dict
    ) -> None:
        """Verifies that a new_A with column sums >= 1 raises ValueError."""
        m = _fresh_model(minimal_data_dict)
        with pytest.raises(ValueError):
            m.apply_technical_change(t=3, new_A=np.ones((3, 3)))


# Factor productivity shock
class TestFactorProductivityShock:
    """Verifies that apply_factor_productivity_shock alters the labour capacity constraint."""

    def test_labour_productivity_reduction_lowers_output(
        self, minimal_data_dict: dict
    ) -> None:
        """Verifies that halving labour productivity for sector A reduces its output at the shocked period."""
        m = _fresh_model(minimal_data_dict)
        m.apply_factor_productivity_shock("A", t=4, prod_L=0.5)
        result = m.run_model()
        # With labour productivity halved, sector A can only produce at most 0.5 * x0[0]
        # from labour alone; output should fall below the base-year value.
        assert result["gross_output"][0, 4] < m.x0[0] - 1e-3, (
            "Sector A output should fall when labour productivity is halved."
        )
        # Finite hiring speed means output at t=3 may differ from x0;
        # compare against an unshocked reference run to isolate the shock at t=4.
        m_ref = _fresh_model(minimal_data_dict)
        ref = m_ref.run_model()
        np.testing.assert_allclose(
            result["gross_output"][0, 3],
            ref["gross_output"][0, 3],
            rtol=1e-10,
            err_msg="Sector A output at t=3 should be unaffected by a shock at t=4.",
        )

    def test_invalid_sector_label_raises_value_error(
        self, minimal_data_dict: dict
    ) -> None:
        """Verifies that an unrecognised sector label raises ValueError."""
        m = _fresh_model(minimal_data_dict)
        with pytest.raises(ValueError):
            m.apply_factor_productivity_shock("Z", t=3, prod_L=1.5)
