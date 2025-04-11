import pandas as pd
import os

def process_excel(input_path, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read all sheets from the input Excel file
    xls = pd.ExcelFile(input_path)
    sheet_names = xls.sheet_names

    for sheet_name in sheet_names:
        # Read the current sheet
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        # Check for required columns
        required_columns = ['Row', 'Timestamp', 'Speaker', 'Text']
        if not all(col in df.columns for col in required_columns):
            print(f"Skipping sheet '{sheet_name}' - missing required columns")
            continue

        max_row = len(df)
        segment_num = 1
        start_index = 0  # 0-based index in DataFrame

        while start_index + 9 < max_row:
            end_index = start_index + 9
            segment_rows = df.iloc[start_index:end_index+1]  # Inclusive range

            # Create new DataFrame with required columns
            new_df = pd.DataFrame({
                'Speaker': segment_rows['Speaker'],
                'Timestamp': segment_rows['Timestamp'],
                'Utterance': segment_rows['Text'],
                'File Origin': sheet_name,
                'Original Row': segment_rows['Row'],
                'Segment Type': ['Reference' if i < 5 else 'Coding' 
                                for i in range(len(segment_rows))]
            })

            # Save to CSV
            output_path = os.path.join(output_dir, 
                                     f"{sheet_name}_Segment{segment_num}.csv")
            new_df.to_csv(output_path, index=False)
            print(f"Created: {output_path}")

            # Update for next segment
            segment_num += 1
            start_index += 5  # Move window by 5 rows

if __name__ == "__main__":
    input_file = "wy3.xlsx"  # Replace with your input file path
    output_directory = "output"  # Output directory
    
    process_excel(input_file, output_directory)