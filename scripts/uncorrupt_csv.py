import pandas as pd
import shutil
import sys
import os


def uncorrupt_csv(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Attempting to uncorrupt: {file_path}")
    
    # Work on a temporary copy to keep your original safe until we succeed
    temp_path = file_path + ".fixing"
    shutil.copy2(file_path, temp_path)
    
    lines_removed = 0
    while True:
        try:
            # Test if pandas can parse the current state of the file
            df = pd.read_csv(temp_path, lineterminator='\n')
            print(f"Success! File is structurally sound.")
            print(f"Recovered {len(df)} complete rows.")
            print(f"Total corrupted lines removed from the end: {lines_removed}")
            break
        except pd.errors.ParserError:
            # If it fails, read all lines, drop the last one, and overwrite the temp file
            with open(temp_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Stop if we delete everything down to the header
            if len(lines) <= 1:
                print("Error: Reached the header without finding a clean row. Aborting.")
                os.remove(temp_path)
                return
            
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.writelines(lines[:-1])
            
            lines_removed += 1

    # Replace the original corrupted file with the fixed one
    shutil.move(temp_path, file_path)
    print(f"Saved fixed data back to: {file_path}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python uncorrupt_csv.py <path_to_csv>")
    else:
        uncorrupt_csv(sys.argv[1])