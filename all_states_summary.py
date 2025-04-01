# Moves Final Scores of all geoetries (ground and bright state) to single file, then takes the average scores across all geometries

import os
import pandas as pd
import io
import re
from pathlib import Path

# Define paths
cwd = Path(os.getcwd())
bright_state_opt_dir = cwd / "bright_state_opt"

# Find all Final_Scores.txt files
score_files = [cwd / "Final_Scores.txt"] + list(bright_state_opt_dir.rglob("Final_Scores.txt"))

# Ensure at least one file exists
if not any(f.exists() for f in score_files):
    print("No Final_Scores.txt files found. Exiting.")
    exit()

geom_count = {}
df_list = []

# Read all Final_Scores.txt files
for score_file in score_files:
    try:
        print(f"Processing file: {score_file}")  # Debugging step
        with open(score_file, 'r') as file:
            lines = file.readlines()

        # Identify where tables start
        table_indices = [i for i, line in enumerate(lines) if "Method" in line and "Final Score" in line]

        if not table_indices:
            print(f"Warning: No valid tables found in {score_file}. Skipping.")
            continue

        for table_start in table_indices:
            table_data = lines[table_start:]
            table_end = next((i for i, line in enumerate(table_data) if line.strip() == ""), len(table_data))

            table_block = table_data[:table_end]
            header_line = table_block[0].strip()
            data_lines = table_block[1:]

            #Manually define headers because I made the txt file poorly, sorry 
            headers = [
                "Method",
                "Normalized Autopylot Component",
                "Raw Autopylot Score",
                "Overlap",
                "Spectra Penalty",
                "Rel. Energy Penalty",
                "Time Component",
                "Run Time",
                "Final Score"
            ]

            #Split each line into 9 columns 
            cleaned_data = [re.split(r"\s+", line.strip(), maxsplit=len(headers)-1) for line in data_lines if line.strip()]

            if not cleaned_data:
                print(f"Warning: No data found in table at {score_file}. Skipping.")
                continue

            try:
                df = pd.DataFrame(cleaned_data, columns=headers)
            except Exception as e:
                print(f"Error creating DataFrame from {score_file}: {e}")
                continue

            print(f"Columns detected in {score_file}: {df.columns.tolist()}")  # Debugging step

            if "Method" not in df.columns or "Final Score" not in df.columns:
                print(f"Warning: {score_file} does not have properly formatted columns. Skipping.")
                continue

            df["Final Score"] = pd.to_numeric(df["Final Score"], errors='coerce')
            df = df[["Method", "Final Score"]].dropna()

            # Count occurrences of each method
            for method in df["Method"]:
                geom_count[method] = geom_count.get(method, 0) + 1

            df_list.append(df)
            print(f"Successfully processed {score_file}\n")  # Debugging step

    except Exception as e:
        print(f"Error reading {score_file}: {e}")

# Check if any data was collected
if not df_list:
    print("No valid data found in Final_Scores.txt files.")
    exit()

# Combine all DataFrames
combined_df = pd.concat(df_list, ignore_index=True)

# Compute average final score for each unique method
avg_scores = combined_df.groupby("Method", as_index=False)["Final Score"].mean()

# Maximum count of occurrences across all methods
max_occurrences = max(geom_count.values(), default=0)

# Add a marker `*` if the method has fewer occurrences than the max
avg_scores["Count"] = avg_scores["Method"].map(geom_count)
avg_scores["Marker"] = avg_scores["Count"].apply(lambda x: "*" if x < max_occurrences else "")

# Sort avg Final Scores
avg_scores = avg_scores.sort_values(by="Final Score", ascending=False)

# Debugging: Print summary before writing the file
print("\nSummary of Averaged Scores:")
print(avg_scores)

# Define output file path
output_file = cwd / "All_Geoms_Final_Scores.txt"

# Writing results to the output file
with open(output_file, 'w') as f:
    f.write("\nMethod                     Final Score   Count\n")
    f.write("-------------------------------------------------\n")

    for _, row in avg_scores.iterrows():
        method_name = str(row['Method']).strip()
        final_score = f"{row['Final Score']:.3f}"
        count = f"({row['Count']})"
        marker = row["Marker"] if row["Marker"] else " "

        line = f"{method_name:<25} {final_score:<10} {marker} {count}\n"
        print("\nWriting line:", line.strip())  # Debugging step
        f.write(line)

print(f"\nCombined final scores saved to {output_file}")
