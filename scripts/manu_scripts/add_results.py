import pandas as pd
import json

# Data paths
excel_file_path = './data/patient_data_with_results_and_annotators.xlsx'
model_dir = './llm-project/results/vllm-serve-async_guided/unsloth'
quant_schemes = ['Q2_K_XL', 'Q3_K_XL', 'Q4_K_XL', 'Q5_K_XL', 'Q6_K_XL', 'Q8_0']
model_classes = [
    'Qwen3-0.6B-GGUF',
    'Qwen3-1.7B-GGUF',
    'Qwen3-4B-GGUF',
    'Qwen3-8B-GGUF',
    'Qwen3-32B-GGUF',
    'Qwen3-30B-A3B-Thinking-2507-GGUF',
]
model_strs = [f"{m}-{q}" for m in model_classes for q in quant_schemes]
output_file_path = './data/patient_data_with_results.csv'
pred_strs = ['single', 'maj_3', 'maj_5', 'maj_10']

# Load the original data file into a DataFrame
df = pd.read_excel(excel_file_path)

# Add model predictions
for model_str in model_strs:

    # Extract model results
    json_file_path = f'{model_dir}/{model_str}.json'
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: results file {json_file_path} not found, skipping.")
        continue
    
    for pred_str in pred_strs:
        new_column_data = data['y_pred'][pred_str]

        # Verify lengths match
        if len(df) != len(new_column_data):
            print(f"Warning: The Excel file has {len(df)} rows, but the JSON list has {len(new_column_data)} items.")
            print("Pandas will truncate or fill with NaN if lengths do not match exactly.")

        # Create the new column
        df[f'{model_str}-{pred_str}'] = new_column_data

# Save the result to a CSV file
df.to_csv(output_file_path, index=False)
print(f"New columns added and saved to {output_file_path}")
output_file_path = output_file_path.replace('.csv', '.xlsx')
df.to_excel(output_file_path, index=False, engine='openpyxl')
print(f"New columns added and saved to {output_file_path}")