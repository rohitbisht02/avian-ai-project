import os
import pandas as pd

# The main directory where your species folders are located
# This path is now correct for running from the project root
AUDIO_BASE_DIR = "data/birds/raw_audio/"

# Path where the final metadata CSV will be saved
OUTPUT_CSV_PATH = "data/birds/metadata.csv" # Also corrected this path

def generate_metadata():
    """
    Scans the audio directory, extracts file paths and labels,
    and saves them to a CSV file.
    """
    data = []
    print("Scanning audio files...")

    for root, _, files in os.walk(AUDIO_BASE_DIR):
        for file in files:
            if file.endswith((".ogg", ".wav")):
                species_label = os.path.basename(root)
                full_path = os.path.join(root, file)
                data.append([full_path, species_label])

    if not data:
        print("Warning: No audio files found! Check the AUDIO_BASE_DIR path.")
        return

    df = pd.DataFrame(data, columns=["file_path", "species_name"])
    df.to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"Metadata generation complete!")
    print(f"{len(df)} files listed in '{OUTPUT_CSV_PATH}'")

if __name__ == '__main__':
    generate_metadata()