import numpy as np
from scipy import stats

# Example collected data (replace with actual data)
performance_model_A = np.array([0.8, 0.85, 0.9, 0.78, 0.82])  # Metrics from model A
performance_model_B = np.array([0.76, 0.83, 0.88, 0.81, 0.84])  # Metrics from model B

# Perform a two-sample t-test
t_stat, p_value = stats.ttest_ind(performance_model_A, performance_model_B)

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

if p_value < 0.05:
    print("There is a statistically significant difference between model A and model B.")
else:
    print("No statistically significant difference was found between model A and model B.")
