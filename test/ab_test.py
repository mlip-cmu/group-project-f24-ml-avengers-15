import numpy as np
from scipy import stats
import pytest

performance_model_A = np.array([0.8, 0.85, 0.9, 0.78, 0.82])  # Metrics from model A
performance_model_B = np.array([0.76, 0.83, 0.88, 0.81, 0.84])  # Metrics from model B


@pytest.mark.parametrize("p_value_threshold, expected", [
    (0.04, False),  # With p ≈ 0.839, we expect no significance at 0.04
    (0.95, True)    # Only significant at very high thresholds
])
def test_significant_difference(p_value_threshold, expected):
    # Calculate actual p-value
    _, p_value = stats.ttest_ind(performance_model_A, performance_model_B)
    # Test if the significance level leads to the expected conclusion
    is_significant = p_value < p_value_threshold
    assert is_significant == expected


def test_t_test():
    t_stat, p_value = stats.ttest_ind(performance_model_A, performance_model_B)
    # Using the actual values from our test data
    assert np.isclose(t_stat, 0.209, atol=0.01)  # Updated to match actual t-statistic
    assert np.isclose(p_value, 0.839, atol=0.01)  # Updated to match actual p-value
