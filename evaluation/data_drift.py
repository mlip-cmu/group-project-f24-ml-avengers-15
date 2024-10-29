# data_drift.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from config import DATA_DIR

# Function to plot distributions and check data drift
def show_data_drift(file1, file2, column_name='rating', output_file='data_drift_report.txt'):
    # Load data from both files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Drop NA values and extract the rating column
    ratings1 = df1[column_name].dropna()
    ratings2 = df2[column_name].dropna()
    
    # Plot the distributions for visual comparison
    plt.figure(figsize=(10, 6))
    sns.histplot(ratings1, kde=True, color='blue', label='File 1', bins=20)
    sns.histplot(ratings2, kde=True, color='orange', label='File 2', bins=20)
    plt.title("Distribution of Ratings Over Time")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.legend()

    # Save the plot to a file
    plt.savefig('data_drift_plot.png')  # Save plot as PNG
    plt.close()  # Close the plot to free memory
    
    # Kolmogorov-Smirnov Test for data drift
    ks_stat, ks_p_value = ks_2samp(ratings1, ratings2)
    
    # Prepare output content
    results = [
        f"Kolmogorov-Smirnov test statistic: {ks_stat}",
        f"p-value: {ks_p_value}",
        "Interpretation: "
    ]
    
    if ks_p_value < 0.05:
        results.append("The distributions of ratings between the two files are significantly different (data drift detected).")
    else:
        results.append("No significant data drift detected between the two files' rating distributions.")

    # Print the results to the console
    for line in results:
        print(line)

    # Write the results to the output file
    with open(output_file, 'w') as f:
        for line in results:
            f.write(line + '\n')

# File paths
file1 = DATA_DIR + "/extracted_ratings_m1.csv"
file2 = DATA_DIR + "/extracted_ratings_m2.csv"

# Show data drift in ratings
show_data_drift(file1, file2)
