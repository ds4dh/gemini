import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

# Configuration
file_path = "/data/paper/patient_data_with_results.csv"
model_id = "Qwen/Qwen3-32B"

def main():
    # Load the Hugging Face Tokenizer
    print(f"Loading tokenizer: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Load the Dataset
    print(f"Loading dataset from: {file_path}...")
    df = pd.read_csv(file_path)
    
    # Drop rows where 'contentText' is missing to prevent tokenization errors
    df = df.dropna(subset=['contentText'])

    # Process Token Lengths
    print("Tokenizing and calculating lengths...")
    df['token_length'] = df['contentText'].astype(str).apply(lambda x: len(tokenizer.encode(x)))

    # Calculate Statistics
    mean_val = df['token_length'].mean()
    median_val = df['token_length'].median()
    std_val = df['token_length'].std()
    
    print("\n--- Token Length Statistics ---")
    print(f"Total Samples: {len(df)}")
    print(f"Mean:          {mean_val:.2f}")
    print(f"Median:        {median_val:.2f}")
    print(f"Std Dev:       {std_val:.2f}")
    print(f"Min:           {df['token_length'].min()}")
    print(f"Max:           {df['token_length'].max()}")

    # Plot the Distribution using Matplotlib + Pandas
    plt.figure(figsize=(10, 6))
    
    # Plot Histogram (Matplotlib native)
    plt.hist(df['token_length'], bins=50, color='skyblue', edgecolor='black', alpha=0.7, density=True, label='Token Count')
    df['token_length'].plot.kde(color='steelblue', linewidth=2, label='KDE')
    
    # Add vertical lines for Mean and Median
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.0f}')
    plt.axvline(median_val, color='green', linestyle='solid', linewidth=2, label=f'Median: {median_val:.0f}')

    # Formatting the plot
    plt.title('Distribution of Token Lengths in "contentText"', fontsize=14, pad=15)
    plt.xlabel('Number of Tokens', fontsize=12)
    plt.ylabel('Density', fontsize=12) # Changed to Density because of the KDE overlay
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # Save and display
    plt.savefig('token_length_distribution.png', dpi=300)
    print("\nPlot saved successfully as 'token_length_distribution.png'")
    plt.show()

if __name__ == "__main__":
    main()