import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer

# Configuration
file_path = r"gemini\data\paper\patient_data_with_results.csv"
model_id = "Qwen/Qwen3-32B"  # Replace with your specific Qwen3 repo if different

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
    print("Tokenizing and calculating lengths (this may take a moment)...")
    # Using lambda to encode each text row and count the tokens
    df['token_length'] = df['contentText'].astype(str).apply(lambda x: len(tokenizer.encode(x)))

    # Calculate and Print Statistics
    mean_val = df['token_length'].mean()
    median_val = df['token_length'].median()
    min_val = df['token_length'].min()
    max_val = df['token_length'].max()
    std_val = df['token_length'].std()
    
    print("\n--- Token Length Statistics ---")
    print(f"Total Samples: {len(df)}")
    print(f"Mean:          {mean_val:.2f}")
    print(f"Median:        {median_val:.2f}")
    print(f"Std Dev:       {std_val:.2f}")
    print(f"Min:           {min_val}")
    print(f"Max:           {max_val}")

    # Plot the Distribution
    plt.figure(figsize=(10, 6))
    
    # Plot histogram with a Kernel Density Estimate (KDE) line
    sns.histplot(df['token_length'], bins=50, kde=True, color='skyblue', edgecolor='black')
    
    # Add vertical lines for Mean and Median
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.0f}')
    plt.axvline(median_val, color='green', linestyle='solid', linewidth=2, label=f'Median: {median_val:.0f}')

    # Formatting the plot
    plt.title('Distribution of Token Lengths in "contentText"', fontsize=14, pad=15)
    plt.xlabel('Number of Tokens', fontsize=12)
    plt.ylabel('Frequency (Number of Samples)', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # Save to file and display
    plt.savefig('token_length_distribution.png', dpi=300)
    print("\nPlot saved successfully as 'token_length_distribution.png'")
    plt.show()

if __name__ == "__main__":
    main()