import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns         


try:
    
    file_path = 'Groceries_dataset.csv'
    basket = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
    display(basket.head())

  
    print("\nDataset Info:")
    basket.info()
    print(f"\nNumber of unique items: {basket['itemDescription'].nunique()}")
    print(f"\nDate range: {basket['Date'].min()} to {basket['Date'].max()}")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    print("Please ensure the file exists and the path is correct.")
    exit()
except KeyError as e:
    print(f"Error: Expected column {e} not found in the CSV. Please check the file format.")
    exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()



basket['Date'] = pd.to_datetime(basket['Date'], format='%d-%m-%Y') # Adjust format if necessary


basket['Transaction_ID'] = basket['Member_number'].astype(str) + '_' + basket['Date'].astype(str)
print("\nPreprocessing: Grouping items by transaction...")


transactions_grouped = basket.groupby('Transaction_ID')['itemDescription'].apply(list)
transactions_list = transactions_grouped.values.tolist()

print(f"Number of transactions: {len(transactions_list)}")




print("\nEncoding transactions...")
te = TransactionEncoder()
te_ary = te.fit(transactions_list).transform(transactions_list)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)




print("\nRunning Apriori to find frequent itemsets...")

min_sup = 0.005
frequent_itemsets = apriori(df_encoded, min_support=min_sup, use_colnames=True)


frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)

print(f"\nFound {len(frequent_itemsets)} frequent itemsets with min_support={min_sup}")
print("\n--- Top 10 Frequent Itemsets ---")
display(frequent_itemsets.head(10))



print("\nGenerating association rules...")

metric_choice = 'lift'
min_threshold = 1.2 

rules = association_rules(frequent_itemsets, metric=metric_choice, min_threshold=min_threshold)


rules = rules.sort_values(by='lift', ascending=False)

print(f"\nFound {len(rules)} association rules with {metric_choice} >= {min_threshold}")
print("\n--- Top 10 Association Rules by Lift ---")

rules_display = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
display(rules_display.head(10))