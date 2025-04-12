from scipy.stats import chi2_contingency
import numpy as np 

data = [[207, 282, 241],
        [234, 242, 232]]

print("Observed Data (Contingency Table):")
print(np.array(data))
try:
    stat, p, dof, expected = chi2_contingency(data)

    
    alpha = 0.05

    # Print the results
    print(f"\nChi-Square Statistic (χ²): {stat:.4f}")
    print(f"Degrees of Freedom (dof): {dof}")
    print(f"P-value: {p:.4f}")
    print(f"Significance Level (alpha): {alpha}")

    print("\nExpected Frequencies (if variables were independent):")
    print(expected.round(2)) 

    
    print("\n--- Conclusion ---")
    if p <= alpha:
        print(f"Since p-value ({p:.4f}) <= alpha ({alpha}), we reject the null hypothesis (H0).")
        print("Conclusion: There is a statistically significant association between the variables (Dependent).")
    else:
        print(f"Since p-value ({p:.4f}) > alpha ({alpha}), we fail to reject the null hypothesis (H0).")
        print("Conclusion: There is not enough statistical evidence to say the variables are associated (Independent).")

except ValueError as e:
    print(f"Error performing Chi-Square test: {e}")
    print("Please ensure the input data is a valid contingency table with non-negative values.")
