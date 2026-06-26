"""Shared fixtures for the pyMacroIO test suite."""

from __future__ import annotations

import numpy as np
import pytest


def _make_data_dict() -> dict:
    """Return a minimal self-consistent 3-sector IO data dictionary."""
    Z0 = np.array(
        [
            [ 0., 20.,  5.],
            [ 3.,  0., 15.],
            [ 8.,  2.,  0.],
        ],
        dtype=np.float64,
    )

    cons  = np.array([[15.], [10.], [14.]], dtype=np.float64)
    gov   = np.array([[ 8.], [ 5.], [ 7.]], dtype=np.float64)
    inv   = np.array([[ 5.], [ 3.], [ 5.]], dtype=np.float64)
    invnt = np.array([[ 4.], [ 2.], [ 2.]], dtype=np.float64)
    exp   = np.array([[ 3.], [ 2.], [ 2.]], dtype=np.float64)
    l0       = np.array([20.,  7.,  6.], dtype=np.float64)
    cap0     = np.array([15.,  4.,  5.], dtype=np.float64)
    tax0     = np.array([ 5.,  2.,  2.], dtype=np.float64)
    imp0     = np.array([ 4.,  3.,  5.], dtype=np.float64)
    profits0 = np.array([5., 2., 2.], dtype=np.float64)

    return {
        "sector_labels":        ["A", "B", "C"],
        "Z0":                   Z0,
        "l0":                   l0,
        "cap0":                 cap0,
        "tax0":                 tax0,
        "imp0":                 imp0,
        "profits0":             profits0,
        "cons_vec":             cons,
        "gov_vec":              gov,
        "inv_vec":              inv,
        "invnt_vec":            invnt,
        "exp_vec":              exp,
        "consumer_taxes_total": 0.0,
        "fd_imports_totals":    {
            "cons": 0.0, "gov": 0.0, "inv": 0.0, "invnt": 0.0, "exp": 0.0
        },
    }


@pytest.fixture(scope="module")
def minimal_data_dict():
    """Return the minimal 3-sector IO data dictionary."""
    return _make_data_dict()


@pytest.fixture(scope="module")
def minimal_model(minimal_data_dict):
    """Return an InputOutputModel built from the minimal 3-sector data dictionary."""
    from pyMacroIO.model import InputOutputModel
    from pyMacroIO.config import ModelConfig

    config = ModelConfig(n_periods=10, time_frequency="quarterly")
    return InputOutputModel(
        n_periods=10,
        time_frequency="quarterly",
        config=config,
        _data_dict=minimal_data_dict,
    )


def _make_klems_data_dict() -> dict:
    """Return a self-consistent 3-sector IO data dictionary for KLEMS tests."""
    Z0 = np.array(
        [
            [0., 8., 4.],
            [3., 0., 6.],
            [2., 1., 0.],
        ],
        dtype=np.float64,
    )

    cons  = np.array([[10.], [ 5.], [10.]], dtype=np.float64)
    gov   = np.array([[ 4.], [ 3.], [ 4.]], dtype=np.float64)
    inv   = np.array([[ 2.], [ 2.], [ 2.]], dtype=np.float64)
    invnt = np.array([[ 1.], [ 1.], [ 1.]], dtype=np.float64)
    exp   = np.array([[ 1.], [ 0.], [ 0.]], dtype=np.float64)

    l0       = np.array([5., 3., 6.], dtype=np.float64)
    cap0     = np.array([3., 2., 1.], dtype=np.float64)
    tax0     = np.array([1., 1., 1.], dtype=np.float64)
    imp0     = np.array([2., 1., 1.], dtype=np.float64)
    profits0 = np.array([14., 4., 1.], dtype=np.float64)

    # l0_by_skill: (3 skill tiers, 3 sectors); column sums must equal l0.
    l0_by_skill = np.array(
        [
            [2., 1., 2.],
            [2., 1., 3.],
            [1., 1., 1.],
        ],
        dtype=np.float64,
    )

    return {
        "sector_labels":        ["electricity", "steel", "services"],
        "Z0":                   Z0,
        "l0":                   l0,
        "cap0":                 cap0,
        "tax0":                 tax0,
        "imp0":                 imp0,
        "profits0":             profits0,
        "cons_vec":             cons,
        "gov_vec":              gov,
        "inv_vec":              inv,
        "invnt_vec":            invnt,
        "exp_vec":              exp,
        "consumer_taxes_total": 0.0,
        "fd_imports_totals":    {
            "cons": 0.0, "gov": 0.0, "inv": 0.0, "invnt": 0.0, "exp": 0.0,
        },
        "l0_by_skill":          l0_by_skill,
    }


@pytest.fixture(scope="module")
def klems_data_dict():
    """Return the KLEMS 3-sector IO data dictionary."""
    return _make_klems_data_dict()


def _make_two_region_data_dict() -> dict:
    """Return a 3-sector IO data dictionary with two regions for multi-region tests."""
    Z0 = np.array(
        [[0., 20.,  5.],
         [3.,  0., 15.],
         [8.,  2.,  0.]],
        dtype=np.float64,
    )

    # row sums = [15, 10, 14]; region 0 consumes A and B, region 1 consumes C.
    cons  = np.array([[15.,  0.], [10.,  0.], [ 0., 14.]], dtype=np.float64)
    gov   = np.array([[ 8.,  0.], [ 5.,  0.], [ 0.,  7.]], dtype=np.float64)
    inv   = np.array([[ 5.,  0.], [ 3.,  0.], [ 0.,  5.]], dtype=np.float64)
    invnt = np.array([[ 4.,  0.], [ 2.,  0.], [ 0.,  2.]], dtype=np.float64)
    exp   = np.array([[ 3.,  0.], [ 2.,  0.], [ 0.,  2.]], dtype=np.float64)

    l0       = np.array([20.,  7.,  6.], dtype=np.float64)
    cap0     = np.array([15.,  4.,  5.], dtype=np.float64)
    tax0     = np.array([ 5.,  2.,  2.], dtype=np.float64)
    imp0     = np.array([ 4.,  3.,  5.], dtype=np.float64)
    profits0 = np.array([ 5.,  2.,  2.], dtype=np.float64)

    return {
        "sector_labels":        ["A", "B", "C"],
        "Z0":                   Z0,
        "l0":                   l0,
        "cap0":                 cap0,
        "tax0":                 tax0,
        "imp0":                 imp0,
        "profits0":             profits0,
        "cons_vec":             cons,
        "gov_vec":              gov,
        "inv_vec":              inv,
        "invnt_vec":            invnt,
        "exp_vec":              exp,
        "consumer_taxes_total": 0.0,
        "fd_imports_totals":    {
            "cons": 0.0, "gov": 0.0, "inv": 0.0, "invnt": 0.0, "exp": 0.0
        },
        "region_map":           np.array([0, 0, 1], dtype=np.int32),
    }


@pytest.fixture(scope="module")
def two_region_data_dict():
    """Return the two-region 3-sector IO data dictionary."""
    return _make_two_region_data_dict()


def _make_skill_data_dict() -> dict:
    """Return the standard 3-sector IO data dictionary augmented with l0_by_skill."""
    Z0 = np.array(
        [
            [ 0., 20.,  5.],
            [ 3.,  0., 15.],
            [ 8.,  2.,  0.],
        ],
        dtype=np.float64,
    )

    cons  = np.array([[15.], [10.], [14.]], dtype=np.float64)
    gov   = np.array([[ 8.], [ 5.], [ 7.]], dtype=np.float64)
    inv   = np.array([[ 5.], [ 3.], [ 5.]], dtype=np.float64)
    invnt = np.array([[ 4.], [ 2.], [ 2.]], dtype=np.float64)
    exp   = np.array([[ 3.], [ 2.], [ 2.]], dtype=np.float64)

    l0       = np.array([20.,  7.,  6.], dtype=np.float64)
    cap0     = np.array([15.,  4.,  5.], dtype=np.float64)
    tax0     = np.array([ 5.,  2.,  2.], dtype=np.float64)
    imp0     = np.array([ 4.,  3.,  5.], dtype=np.float64)
    profits0 = np.array([ 5.,  2.,  2.], dtype=np.float64)

    l0_by_skill = np.array(
        [
            [10., 3., 3.],
            [ 7., 2., 2.],
            [ 3., 2., 1.],
        ],
        dtype=np.float64,
    )

    return {
        "sector_labels":        ["A", "B", "C"],
        "Z0":                   Z0,
        "l0":                   l0,
        "cap0":                 cap0,
        "tax0":                 tax0,
        "imp0":                 imp0,
        "profits0":             profits0,
        "cons_vec":             cons,
        "gov_vec":              gov,
        "inv_vec":              inv,
        "invnt_vec":            invnt,
        "exp_vec":              exp,
        "consumer_taxes_total": 0.0,
        "fd_imports_totals":    {
            "cons": 0.0, "gov": 0.0, "inv": 0.0, "invnt": 0.0, "exp": 0.0
        },
        "l0_by_skill":          l0_by_skill,
    }


@pytest.fixture(scope="module")
def skill_data_dict():
    """Return the 3-sector IO data dictionary augmented with l0_by_skill."""
    return _make_skill_data_dict()


def _make_row_data_dict() -> dict:
    """Return a self-consistent 4-sector, 2-region IO data dict with a RoW region matching domestic sectors."""
    Z0 = np.array(
        [
            [0., 5., 0., 0.],
            [3., 0., 0., 0.],
            [0., 0., 0., 3.],
            [0., 0., 2., 0.],
        ],
        dtype=np.float64,
    )

    l0       = np.array([10., 5., 8., 4.], dtype=np.float64)
    cap0     = np.array([ 4., 2., 4., 2.], dtype=np.float64)
    tax0     = np.array([ 2., 1., 1., 1.], dtype=np.float64)
    imp0     = np.array([ 2., 1., 1., 1.], dtype=np.float64)
    profits0 = np.array([ 2., 1., 1., 0.], dtype=np.float64)

    cons  = np.array([[10., 0.], [7., 0.], [0., 9.], [0., 5.]], dtype=np.float64)
    gov   = np.array([[ 4., 0.], [2., 0.], [0., 2.], [0., 2.]], dtype=np.float64)
    inv   = np.array([[ 2., 0.], [1., 0.], [0., 1.], [0., 1.]], dtype=np.float64)
    invnt = np.array([[ 1., 0.], [1., 0.], [0., 1.], [0., 1.]], dtype=np.float64)
    exp   = np.array([[ 1., 0.], [1., 0.], [0., 1.], [0., 0.]], dtype=np.float64)

    return {
        "sector_labels":        ["A", "B", "RoW:A", "RoW:B"],
        "Z0":                   Z0,
        "l0":                   l0,
        "cap0":                 cap0,
        "tax0":                 tax0,
        "imp0":                 imp0,
        "profits0":             profits0,
        "cons_vec":             cons,
        "gov_vec":              gov,
        "inv_vec":              inv,
        "invnt_vec":            invnt,
        "exp_vec":              exp,
        "consumer_taxes_total": 0.0,
        "fd_imports_totals":    {
            "cons": 0.0, "gov": 0.0, "inv": 0.0, "invnt": 0.0, "exp": 0.0,
        },
        "region_map":           np.array([0, 0, 1, 1], dtype=np.int32),
        "region_labels":        ["domestic", "RoW"],
    }


@pytest.fixture(scope="module")
def row_data_dict():
    """Return the 4-sector, 2-region IO data dictionary with an explicit RoW region."""
    return _make_row_data_dict()
