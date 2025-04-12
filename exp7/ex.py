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
def inspect(output_list):
    """
    Parses the list of RelationRecord objects from apriori for rules
    generated with max_length=2.
    Extracts LHS, RHS, Support, Confidence, and Lift.
    """
    parsed_results = []
    for record in output_list:
       
        if record.ordered_statistics:
            
            rule_info = record.ordered_statistics[0]
            lhs = tuple(rule_info.items_base)[0] 
            rhs = tuple(rule_info.items_add)[0] 
            support = record.support
            confidence = rule_info.confidence
            lift = rule_info.lift
            parsed_results.append((lhs, rhs, support, confidence, lift))
    return parsed_results

if output_rules_list:
    print("\nProcessing rules into a DataFrame...")
    parsed_output = inspect(output_rules_list)

    output_DataFrame = pd.DataFrame(parsed_output,
                                    columns=['Left_Hand_Side', 'Right_Hand_Side',
                                             'Support', 'Confidence', 'Lift'])

    output_DataFrame = output_DataFrame.sort_values(by='Lift', ascending=False)

    print("\n--- Discovered Association Rules ---")
  
    print(output_DataFrame)
   
    print("\n--- Top 5 Rules by Lift ---")
    print(output_DataFrame.head(5))

else:
    print("\nNo association rules found meeting the specified criteria (Support, Confidence, Lift).")
    print("Consider adjusting the parameters (e.g., lowering min_support, min_confidence, or min_lift).")