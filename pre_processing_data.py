import os
import pandas as pd
import numpy as np
from pathlib import Path

def convert_txt_to_array(file_path):
    """
    Parses a .txt file containing lines like:
        3.52 FSR2 Voltage (A3) = 3.76 V
        3.82 FSR2 Voltage (A3) = 3.06 V
        4.85 FSR3 Voltage (A5) = 2.72 V
        ...
    For multiple sensors (FSR1, FSR2, FSR3).

    Returns:
      arr: np.ndarray of shape (N, 4).
           Where columns are [FSR1, FSR2, FSR3, delta_t].
    """
    # Step 1: Read file line by line, parse
    data_rows = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines
            
            # Example line.split():
            # ["3.52", "FSR2", "Voltage", "(A3)", "=", "3.76", "V"]
            tokens = line.split()
            
            if len(tokens) < 6:
                print(f"Skipping malformed line in {file_path}: {line}")
                continue  # skip malformed lines
            
            try:
                # time is the first token (float)
                time_val = float(tokens[0])
                
                # sensor is the second token, e.g. "FSR1" / "FSR2" / "FSR3"
                sensor_name = tokens[1]  # e.g. "FSR2"
                
                # voltage is the 6th token (index 5), e.g. "3.76"
                voltage_val = float(tokens[5])
            except ValueError as e:
                print(f"Error parsing line in {file_path}: {line}")
                print(f"Error: {e}")
                continue  # skip lines with parsing errors
            
            data_rows.append((time_val, sensor_name, voltage_val))
    
    if not data_rows:
        print(f"No valid data found in {file_path}. Skipping.")
        return None
    
    # Step 2: Build a DataFrame with columns [time, sensor, voltage]
    df_long = pd.DataFrame(data_rows, columns=["time", "sensor", "voltage"])
    
    # Step 3: Pivot so each sensor is a column, index is time
    # This yields columns = ['FSR1', 'FSR2', 'FSR3'] (potentially)
    df_wide = df_long.pivot(index="time", columns="sensor", values="voltage")
    
    # Step 4: Sort by time (ascending)
    df_wide = df_wide.sort_index()
    
    # If some sensors are missing at some times, we get NaN. Let's interpolate to fill them:
    df_wide = df_wide.interpolate(method='linear')  # linear interpolation
    # Optionally fill any remaining NaNs at start/end
    df_wide = df_wide.fillna(method='bfill').fillna(method='ffill')
    
    # Ensure columns exist (even if a sensor might be missing entirely)
    for sensor_col in ['FSR1', 'FSR2', 'FSR3']:
        if sensor_col not in df_wide.columns:
            # If a sensor never appeared, create a column of zeros (or NaNs)
            df_wide[sensor_col] = 0.0
    
    # Reorder columns just to be consistent: [FSR1, FSR2, FSR3]
    df_wide = df_wide[['FSR1', 'FSR2', 'FSR3']]
    
    # Step 5: Compute delta_t = difference in consecutive time steps
    # First, let's get the sorted time index as a NumPy array
    time_index = df_wide.index.to_numpy()
    
    # delta_t[0] can be 0 or we can skip the first one
    delta_t = np.diff(time_index, prepend=time_index[0])
    
    # Add it as a new column to df_wide
    df_wide["delta_t"] = delta_t
    
    # Step 6: Convert to np.ndarray of shape (N, 4)
    # columns: [FSR1, FSR2, FSR3, delta_t]
    arr = df_wide[['FSR1', 'FSR2', 'FSR3', 'delta_t']].to_numpy()
    
    return arr


def process_folder(input_folder, output_folder, label, sensor_columns=['FSR1', 'FSR2', 'FSR3']):
    """
    Processes all .txt files in the input_folder, converts them to arrays,
    and saves them as .npy files in the output_folder.

    Args:
      input_folder (str or Path): Path to the folder containing .txt files.
      output_folder (str or Path): Path to the folder where .npy files will be saved.
      label (int): Class label for the files in this folder (e.g., 0 for square).
      sensor_columns (list): List of sensor column names.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # List all .txt files in the input directory
    txt_files = list(input_path.glob("*.txt"))
    
    if not txt_files:
        print(f"No .txt files found in {input_folder}.")
        return
    
    for file in txt_files:
        print(f"Processing file: {file.name}")
        array_out = convert_txt_to_array(file)
        
        if array_out is None:
            print(f"Skipping file due to parsing issues: {file.name}")
            continue
        
        
        # Define output file name (e.g., B5.npy for B5.txt)
        output_file = output_path / (file.stem + ".npy")
        
        # Save the array
        np.save(output_file, array_out)
        print(f"Saved processed data to: {output_file}\n")


def main():
    """
    Main function to process all files in the 'square readings' folder.
    """
    # Define input and output folders
    input_folder = "Sphere_Readings"  # Replace with actual folder path
    output_folder = "processed_ball_readings"  # Desired output folder
    
    # Define label for this class (e.g., 1 for square)
    # IMPORTANT!!
    label = 0  # Adjust as needed IMPORTANT!!
    
    # Process the folder
    process_folder(input_folder, output_folder, label)
    
    print("Processing completed.")


if __name__ == "__main__":
    main()
