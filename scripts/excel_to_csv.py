"""
Script to read an Excel file and generate CSV files for each sheet.

This script reads the sample_dataset.xlsx file from datasets/usaa/data/
and creates separate CSV files for each sheet in the Excel file,
saving them to datasets/usaa/data/extracted_variables/
"""

import pandas as pd
import os
from pathlib import Path


def excel_to_csv(excel_path, output_dir):
    """
    Read an Excel file and save each sheet as a separate CSV file.

    Parameters:
    -----------
    excel_path : str
        Path to the input Excel file
    output_dir : str
        Directory where CSV files will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read all sheets from the Excel file
    excel_file = pd.ExcelFile(excel_path)

    print(f"Found {len(excel_file.sheet_names)} sheet(s) in the Excel file:")

    # Process each sheet
    for sheet_name in excel_file.sheet_names:
        print(f"  - Processing sheet: '{sheet_name}'")

        # Read the sheet into a DataFrame
        df = pd.read_excel(excel_file, sheet_name=sheet_name)

        # Create a valid filename from the sheet name
        # Replace special characters that might cause issues in filenames
        safe_sheet_name = sheet_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
        csv_filename = f"{safe_sheet_name}.csv"
        csv_path = os.path.join(output_dir, csv_filename)

        # Save the DataFrame as CSV
        df.to_csv(csv_path, index=False)

        print(f"    ✓ Saved to: {csv_path} ({len(df)} rows, {len(df.columns)} columns)")

    print(f"\nSuccessfully exported {len(excel_file.sheet_names)} sheet(s) to CSV files.")


if __name__ == "__main__":
    # Define paths
    base_dir = Path(__file__).parent.parent
    excel_path = base_dir / "datasets" / "usaa" / "data" / "sample_dataset.xlsx"
    output_dir = base_dir / "datasets" / "usaa" / "data" / "extracted_variables"

    # Check if input file exists
    if not excel_path.exists():
        print(f"Error: Excel file not found at {excel_path}")
        exit(1)

    # Convert Excel to CSV
    excel_to_csv(str(excel_path), str(output_dir))
