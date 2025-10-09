#!/usr/bin/env python3
"""
Script to order coefficients by magnitude.
Reads coefficients.csv and outputs a sorted version.
"""

import pandas as pd
import sys
from pathlib import Path


def order_by_magnitude(input_file, output_file=None, column='human'):
    """
    Order coefficients by their absolute magnitude.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the sorted CSV (optional)
        column: Which column to use for sorting ('human' or 'gpt')
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Add a column for absolute magnitude
    df['magnitude'] = df[column].abs()
    
    # Sort by magnitude in descending order
    df_sorted = df.sort_values('magnitude', ascending=False)
    
    # Remove the temporary magnitude column
    df_sorted = df_sorted.drop('magnitude', axis=1)
    
    # Print summary statistics
    print(f"\nCoefficients ordered by magnitude (using '{column}' column):")
    print(f"Total features: {len(df_sorted)}")
    print(f"\nTop 10 features by magnitude:")
    print("=" * 80)
    
    for idx, row in df_sorted.head(10).iterrows():
        feature = row['Feature']
        value = row[column]
        print(f"{feature:50s} {value:+.6f}")
    
    print("\n" + "=" * 80)
    print(f"\nBottom 10 features by magnitude:")
    print("=" * 80)
    
    for idx, row in df_sorted.tail(10).iterrows():
        feature = row['Feature']
        value = row[column]
        print(f"{feature:50s} {value:+.6f}")
    
    # Save to file if output path is provided
    if output_file:
        df_sorted.to_csv(output_file, index=False)
        print(f"\nSorted coefficients saved to: {output_file}")
    
    return df_sorted


def main():
    # Default input file
    script_dir = Path(__file__).parent
    input_file = script_dir / 'coefficients.csv'
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])
    
    column = 'human'  # Default to 'human' column
    if len(sys.argv) > 2:
        column = sys.argv[2]
        if column not in ['human', 'gpt']:
            print(f"Error: Column must be 'human' or 'gpt', got '{column}'")
            sys.exit(1)
    
    output_file = None
    if len(sys.argv) > 3:
        output_file = Path(sys.argv[3])
    
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    # Order coefficients
    df_sorted = order_by_magnitude(input_file, output_file, column)
    
    return df_sorted


if __name__ == '__main__':
    main()

