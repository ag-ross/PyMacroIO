"""Tests for the rationing, inventory-update, profit, and savings methods of InputOutputModel."""

from __future__ import annotations

import numpy as np
import pytest

from pyMacroIO.model import InputOutputModel
from pyMacroIO.config import ModelConfig


# Fixtures
@pytest.fixture(scope="module")
def model(minimal_data_dict) -> InputOutputModel:
    """Return a 3-sector InputOutputModel built from the minimal conftest data dictionary."""
    config = ModelConfig(n_periods=10, time_frequency="quarterly")
    return InputOutputModel(
        n_periods=10,
        time_frequency="quarterly",
        config=config,
        _data_dict=minimal_data_dict,
    )


@pytest.fixture(scope="module")
def bare_model(minimal_data_dict) -> InputOutputModel:
    """Return a model instance used by TestInventoryS for self-contained array tests."""
    config = ModelConfig(n_periods=10, time_frequency="quarterly")
    return InputOutputModel(
        n_periods=10,
        time_frequency="quarterly",
        config=config,
        _data_dict=minimal_data_dict,
    )


# Tests for intercons_Z
class TestIntermedConsZ:
    """Tests for the intercons_Z intermediate-delivery rationing method."""

    def test_full_supply_no_priority_returns_orders(self, model: InputOutputModel) -> None:
        """Verifies that intercons_Z returns Z0 exactly when supply equals demand under 'no' priority."""
        m = model
        Z = m.intercons_Z(O=m.Z0, d=m.x0, x=m.x0, firm_priority="no", S=m.Z0)
        np.testing.assert_allclose(Z, m.Z0, rtol=1e-10)

    def test_half_supply_no_priority_scales_rows(self, model: InputOutputModel) -> None:
        """Verifies that constraining two sectors independently scales their respective delivery rows."""
        m = model
        x = np.array([30.0, 20.0, 40.0])
        Z = m.intercons_Z(O=m.Z0, d=m.x0, x=x, firm_priority="no", S=m.Z0)
        np.testing.assert_allclose(Z[0, :], np.array([0.0, 10.0, 2.5]),  rtol=1e-10)
        np.testing.assert_allclose(Z[1, :], np.array([1.5,  0.0, 7.5]),  rtol=1e-10)
        np.testing.assert_allclose(Z[2, :], m.Z0[2, :],                  rtol=1e-10)

    def test_supplier_priority_tight_constraint(self, model: InputOutputModel) -> None:
        """Verifies that 'supplier' priority applies the fill ratio against total orders."""
        m = model
        x = np.array([15.0, 40.0, 40.0])
        Z = m.intercons_Z(O=m.Z0, d=m.x0, x=x, firm_priority="supplier", S=m.Z0)
        expected_row0 = np.array([0.0, 12.0, 3.0])
        np.testing.assert_allclose(Z[0, :], expected_row0, rtol=1e-10)
        np.testing.assert_allclose(Z[1, :], m.Z0[1, :], rtol=1e-10)
        np.testing.assert_allclose(Z[2, :], m.Z0[2, :], rtol=1e-10)

    def test_supplier_priority_more_generous_to_intermediate_users(
        self, model: InputOutputModel
    ) -> None:
        """Verifies that 'supplier' priority delivers more to intermediate users than 'no' priority when output is constrained."""
        m = model
        x = np.array([15.0, 40.0, 40.0])
        Z_supplier = m.intercons_Z(O=m.Z0, d=m.x0, x=x, firm_priority="supplier", S=m.Z0)
        Z_no = m.intercons_Z(O=m.Z0, d=m.x0, x=x, firm_priority="no", S=m.Z0)
        assert Z_supplier[0, 1] > Z_no[0, 1]

    def test_zero_demand_sector_yields_no_nan(self, model: InputOutputModel) -> None:
        """Verifies that a sector with zero demand and zero output does not propagate NaN into deliveries from that sector."""
        m = model
        # Sector 1: x=0, d=0; safe-divide guard prevents NaN.
        d = m.x0.copy()
        d[1] = 0.0
        x = m.x0.copy()
        x[1] = 0.0
        Z = m.intercons_Z(O=m.Z0, d=d, x=x, firm_priority="no", S=m.Z0)
        assert not np.any(np.isnan(Z))
        # Sector 1 cannot deliver anything when its output is zero.
        np.testing.assert_allclose(Z[1, :], 0.0, atol=1e-10)


# Tests for finalcons_c
class TestFinalconsC:
    """Tests for the finalcons_c realised-consumption method."""

    def test_full_supply_no_priority_returns_desired(self, model: InputOutputModel) -> None:
        """Verifies that realised consumption equals desired consumption when supply is full under 'no' priority."""
        m = model
        c = m.finalcons_c(cd=m.c0, d=m.x0, x=m.x0, Z=m.Z0, firm_priority="no")
        np.testing.assert_allclose(c, m.c0, rtol=1e-10)

    def test_half_supply_no_priority_scales_constrained_sector(
        self, model: InputOutputModel
    ) -> None:
        """Verifies that realised consumption is halved for the sector with half output under 'no' priority."""
        m = model
        x = np.array([30.0, 40.0, 40.0])
        # Use Z from the half-supply intercons_Z result
        Z_half = m.intercons_Z(O=m.Z0, d=m.x0, x=x, firm_priority="no", S=m.Z0)
        c = m.finalcons_c(cd=m.c0, d=m.x0, x=x, Z=Z_half, firm_priority="no")
        expected = np.array([m.c0[0] * 0.5, m.c0[1], m.c0[2]])
        np.testing.assert_allclose(c, expected, rtol=1e-10)

    def test_supplier_priority_fully_constrained_sector_zero_consumption(
        self, model: InputOutputModel
    ) -> None:
        """Verifies that consumption is zero for a sector whose entire output is absorbed by intermediate users under 'supplier' priority."""
        m = model
        x = np.array([15.0, 40.0, 40.0])
        # Z[0,:] = [0,12,3]; sum = 15 = x[0], so nothing remains for households
        Z_sup = m.intercons_Z(O=m.Z0, d=m.x0, x=x, firm_priority="supplier", S=m.Z0)
        c = m.finalcons_c(cd=m.c0, d=m.x0, x=x, Z=Z_sup, firm_priority="supplier")
        assert np.isclose(c[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(c[1:], m.c0[1:], rtol=1e-10)

    def test_supplier_priority_partial_remainder_matches_hand_computed(
        self, model: InputOutputModel
    ) -> None:
        """Verifies the supplier-priority finalcons_c formula when output exceeds intermediate orders but falls short of total demand."""
        m = model
        x = np.array([40.0, 40.0, 40.0])
        Z_sup = m.intercons_Z(O=m.Z0, d=m.x0, x=x, firm_priority="supplier", S=m.Z0)
        c = m.finalcons_c(cd=m.c0, d=m.x0, x=x, Z=Z_sup, firm_priority="supplier")
        expected_c0 = m.c0[0] * 15.0 / 35.0   # = 225/35
        np.testing.assert_allclose(c[0], expected_c0, rtol=1e-10)
        # Sectors 1 and 2 are at full supply; their consumption is unaffected.
        np.testing.assert_allclose(c[1:], m.c0[1:], rtol=1e-10)

    def test_supplier_priority_full_supply_returns_desired(
        self, model: InputOutputModel
    ) -> None:
        """Verifies that 'supplier' priority at full supply returns the same result as 'no' priority."""
        m = model
        c = m.finalcons_c(cd=m.c0, d=m.x0, x=m.x0, Z=m.Z0, firm_priority="supplier")
        np.testing.assert_allclose(c, m.c0, rtol=1e-10)

    def test_consumption_non_negative_at_half_capacity(
        self, model: InputOutputModel
    ) -> None:
        """Verifies that realised consumption is non-negative under half-capacity output."""
        m = model
        x = m.x0 * 0.5
        Z_half = m.intercons_Z(O=m.Z0 * 0.5, d=m.x0, x=x, firm_priority="no", S=m.Z0)
        c = m.finalcons_c(cd=m.c0, d=m.x0, x=x, Z=Z_half, firm_priority="no")
        assert np.all(c >= 0.0)


# Tests for inventory_S
class TestInventoryS:
    """Tests for the inventory_S end-of-period inventory-update method.

    All tests use self-contained arrays rather than the conftest model fixture
    to make the hand-computed expectations transparent.
    """

    def test_delivery_covers_use_exactly_leaves_S_unchanged(
        self, bare_model: InputOutputModel
    ) -> None:
        """Verifies that inventories are unchanged when deliveries exactly cover production requirements."""
        m = bare_model
        x = np.array([10.0, 10.0])
        S = np.array([[5.0, 4.0], [3.0, 2.0]])
        Z = np.array([[1.0, 0.0], [0.0, 1.0]])
        A = np.array([[0.1, 0.0], [0.0, 0.1]])
        S_new = m.inventory_S(x=x, S=S, Z=Z, A=A)
        np.testing.assert_allclose(S_new, S, rtol=1e-10)

    def test_no_deliveries_depletes_stocks(
        self, bare_model: InputOutputModel
    ) -> None:
        """Verifies that inventories are reduced by production requirements when no deliveries arrive."""
        m = bare_model
        x = np.array([10.0, 10.0])
        S = np.array([[1.0, 2.0], [3.0, 1.0]])
        Z = np.zeros((2, 2))
        A = np.array([[0.2, 0.1], [0.1, 0.2]])
        S_new = m.inventory_S(x=x, S=S, Z=Z, A=A)
        expected = np.array([[0.0, 1.0], [2.0, 0.0]])
        np.testing.assert_allclose(S_new, expected, rtol=1e-10)

    def test_excess_deliveries_accumulate_stocks(
        self, bare_model: InputOutputModel
    ) -> None:
        """Verifies that inventories accumulate when deliveries exceed production requirements."""
        m = bare_model
        x = np.array([5.0, 5.0])
        S = np.array([[1.0, 0.0], [0.0, 1.0]])
        Z = np.array([[2.0, 1.0], [1.0, 2.0]])
        A = np.array([[0.1, 0.1], [0.1, 0.1]])
        S_new = m.inventory_S(x=x, S=S, Z=Z, A=A)
        expected = np.array([[2.5, 0.5], [0.5, 2.5]])
        np.testing.assert_allclose(S_new, expected, rtol=1e-10)

    def test_floor_binds_giving_non_negative_inventories(
        self, bare_model: InputOutputModel
    ) -> None:
        """Verifies that the zero floor is binding and no inventory entry turns negative."""
        m = bare_model
        x = np.array([10.0, 10.0])
        S = np.array([[1.0, 2.0], [3.0, 1.0]])
        Z = np.zeros((2, 2))
        A = np.array([[0.2, 0.1], [0.1, 0.2]])
        S_new = m.inventory_S(x=x, S=S, Z=Z, A=A)
        assert np.all(S_new >= 0.0)


# Tests for profit_pi
class TestProfitPi:
    """Tests for the profit_pi accounting identity method."""

    def test_doubling_inputs_doubles_profits(self, model: InputOutputModel) -> None:
        """Verifies that proportional doubling of output and all inputs doubles profits."""
        m = model
        x2 = 2.0 * m.x0
        Z2 = 2.0 * m.Z0
        l2 = 2.0 * m.l0
        pi2 = m.profit_pi(x=x2, Z=Z2, l=l2)
        np.testing.assert_allclose(pi2, 2.0 * m.profits0, rtol=1e-10)

    def test_halved_capital_productivity_reduces_profits(
        self, model: InputOutputModel
    ) -> None:
        """Verifies that halving capital productivity for sector 0 reduces its profit below the base-year level."""
        m = model
        fprod_k = np.array([0.5, 1.0, 1.0])
        pi_base = m.profit_pi(x=m.x0, Z=m.Z0, l=m.l0)
        pi_reduced = m.profit_pi(x=m.x0, Z=m.Z0, l=m.l0, fprod_k=fprod_k)
        assert pi_reduced[0] < pi_base[0]
        assert np.isclose(pi_reduced[0], -10.0, rtol=1e-10)
        np.testing.assert_allclose(pi_reduced[1:], pi_base[1:], rtol=1e-10)

    def test_zero_output_gives_zero_profit(self, model: InputOutputModel) -> None:
        """Verifies that profit is zero when output, intermediate inputs, and labour are all zero."""
        m = model
        x = np.zeros(m.N)
        Z = np.zeros((m.N, m.N))
        l = np.zeros(m.N)
        pi = m.profit_pi(x=x, Z=Z, l=l)
        np.testing.assert_allclose(pi, np.zeros(m.N), atol=1e-10)


# Tests for savings_s_regional
class TestSavingsRegional:
    """Tests for the savings_s_regional per-region household savings method."""

    def test_known_income_and_consumption(self, model: InputOutputModel) -> None:
        """Verifies the savings formula for a known income and consumption pair in a single-region model."""
        m = model
        income_r = np.array([100.0])
        c_r = np.array([45.0])
        result = m.savings_s_regional(household_income_r_t=income_r, c_r_t=c_r)
        assert np.isclose(result[0], 50.0, rtol=1e-10)

    def test_higher_consumption_reduces_savings(self, model: InputOutputModel) -> None:
        """Verifies that savings fall when consumption rises, holding income fixed."""
        m = model
        income = np.array([100.0])
        s_low = m.savings_s_regional(
            household_income_r_t=income, c_r_t=np.array([45.0])
        )
        s_high = m.savings_s_regional(
            household_income_r_t=income, c_r_t=np.array([81.0])
        )
        assert s_high[0] < s_low[0]

    def test_zero_consumption_savings_equals_income(self, model: InputOutputModel) -> None:
        """Verifies that savings equal income when consumption is zero."""
        m = model
        income = np.array([100.0])
        c = np.array([0.0])
        result = m.savings_s_regional(household_income_r_t=income, c_r_t=c)
        assert np.isclose(result[0], 100.0, rtol=1e-10)

    def test_output_shape_is_n_regions(self, model: InputOutputModel) -> None:
        """Verifies that the return array has shape (n_regions,), which is (1,) for the single-region test model."""
        m = model
        income = np.array([100.0])
        c = np.array([45.0])
        result = m.savings_s_regional(household_income_r_t=income, c_r_t=c)
        assert result.shape == (m.n_regions,)
        assert result.shape == (1,)
