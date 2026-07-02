import pandas as pd
import os

def process_consensus_data():
    # Load the data
    input_file = './data/paper/patient_data_with_consensus_details.xlsx'
    output_file = './data/paper/consensus_details.csv'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"Loading data from {input_file}...")
    df = pd.read_excel(input_file)

    # Keep only the requested columns
    columns_to_keep = [
        "Etudiants1", "Etudiants2", "Etudiants3_consensus", 
        "Internes1", "Internes2", "Internes3_consensus", 
        "CDC1", "CDC2", "CDC3_consensus"
    ]
    
    # Filter the dataframe to just these columns
    df = df[columns_to_keep]

    # Rename columns based on your rules:
    # - Rename "CDC1" and "CDC2" to "Chefs_de_Clinique1" and "Chefs_de_Clinique2"
    # - Rename all "3_consensus" columns to just their group names
    rename_dict = {
        "CDC1": "Chefs_de_Clinique1",
        "CDC2": "Chefs_de_Clinique2",
        "CDC3_consensus": "Chefs_de_Clinique",
        "Etudiants3_consensus": "Etudiants",
        "Internes3_consensus": "Internes"
    }
    df = df.rename(columns=rename_dict)

    # Save the modified dataframe to CSV
    print(f"Saving processed data to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    process_consensus_data()