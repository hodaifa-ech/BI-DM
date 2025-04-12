import numpy as np
import pandas as pd
from apyori import apriori


csv_file_path = 'Market_Basket_Optimisation.csv'

try:
    Data = pd.read_csv(csv_file_path, header=None)
    print(f"Dataset loaded successfully. Shape: {Data.shape}")

    
    transacts = []
    num_rows = Data.shape[0]
    num_cols = Data.shape[1]

    print(f"Processing {num_rows} transactions...")
    for i in range(0, num_rows):
        transaction = [str(Data.values[i, j]) for j in range(0, num_cols) if pd.notna(Data.values[i, j])] # Only add non-NaN items
        
        transacts.append(transaction)

    print(f"Preprocessing complete. {len(transacts)} transactions prepared.")
  
except FileNotFoundError:
    print(f"Error: The file '{csv_file_path}' was not found.")
    print("Please ensure the file exists and the path is correct.")
    
    exit()
except Exception as e:
    print(f"An error occurred during data loading or preprocessing: {e}")
    exit()

    

print("\nTraining Apriori model...")


rules_generator = apriori(transactions=transacts,
                          min_support=0.003,
                          min_confidence=0.2,
                          min_lift=3,
                          min_length=2,
                          max_length=2)

output_rules_list = list(rules_generator)

print(f"Apriori training complete. Found {len(output_rules_list)} rules/relations.")