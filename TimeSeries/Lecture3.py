import numpy as np
from scipy import stats

# Time (t)	1	2	3	4	5	6	7	8	9	10
Y = np.array([0.82, 1.9, 1.8, 2.4, 2, 2.4, 1.45, 2.1, 0.7, 1.34]).reshape(-1, 1)
U_time = np.array([0.8, 1, 1.2, 0.5, 0.6, 0.2, 0.8, 0.5, 0.3, 0.1])
U = np.zeros([10, 4])
for i, u in enumerate(U):
    u[0] = U_time[i]
    if i != 0:
        u[1] = U_time[i - 1]
        if i != 1:
            u[2] = U_time[i - 2]
            if i != 2:
                u[3] = U_time[i - 3]
UT_U_inv = np.linalg.inv(U.T @ U)
UT_Y = U.T @ Y
H = UT_U_inv @ UT_Y
H_var = 1 * np.diagonal(UT_U_inv)

target_probability = 0.6827

z_score_upper_bound = (1 + target_probability) / 2
z_value = stats.norm.ppf(z_score_upper_bound)
print(H_var)
standard_deviations = np.sqrt(H_var)
b_values = z_value * standard_deviations
print(H, b_values)
