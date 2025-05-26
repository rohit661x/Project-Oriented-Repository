# plot_convergence.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os # To help construct file paths

def plot_convergence(filename_prefix, option_type):
    # Construct the full path to the CSV file.
    # Assumes the script is run from the project root, and CSVs are in the 'build' folder.
    # os.path.join handles path separators correctly across OS.
    filepath = os.path.join('build', f'{filename_prefix}_convergence.csv')

    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: CSV file not found at '{filepath}'.")
        print("Please ensure you have successfully run the C++ program first to generate the CSV data.")
        return

    # Create a figure with two subplots side-by-side
    plt.figure(figsize=(15, 7)) # Increased figure size for better readability

    # --- Subplot 1: Option Price Convergence with Confidence Interval ---
    plt.subplot(1, 2, 1) # 1 row, 2 columns, first plot
    plt.plot(df['Simulations'], df['Price'], marker='o', linestyle='-', color='blue', label='Estimated Price', markersize=4)
    plt.fill_between(df['Simulations'], df['CILower'], df['CIUpper'], color='blue', alpha=0.2, label='95% Confidence Interval')

    plt.xscale('log') # Use a log scale for the x-axis (number of simulations) for clearer trends
    plt.xlabel('Number of Simulations (Log Scale)', fontsize=12)
    plt.ylabel('Option Price', fontsize=12)
    plt.title(f'{option_type} Price Convergence', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.7) # Add grid lines for better readability
    plt.legend(fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)


    # --- Subplot 2: Standard Error Convergence ---
    plt.subplot(1, 2, 2) # 1 row, 2 columns, second plot
    plt.plot(df['Simulations'], df['StandardError'], marker='o', linestyle='-', color='red', label='Standard Error', markersize=4)

    plt.xscale('log') # Log scale for x-axis
    plt.yscale('log') # Log scale for y-axis (standard error) to better show the 1/sqrt(N) decay
    plt.xlabel('Number of Simulations (Log Scale)', fontsize=12)
    plt.ylabel('Standard Error (Log Scale)', fontsize=12)
    plt.title(f'{option_type} Standard Error Convergence', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend(fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)

    # Add a main title for the entire figure
    # Add a main title for the entire figure
    plt.suptitle(f'Monte Carlo Convergence Analysis for {option_type}', fontsize=18, y=0.98) # Changed y from 1.02 to 0.98

    # Adjust layout to prevent titles/labels from overlapping
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Adjust layout to prevent titles/labels from overlapping
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to make space for suptitle

    # Save the plot as a PNG image
    output_filename = f'{filename_prefix}_convergence.png'
    plt.savefig(output_filename, dpi=300) # Save with higher resolution
    print(f"Plot saved to '{output_filename}'")

    # plt.show() # Uncomment this line if you want the plot to pop up immediately after generation

# This ensures the functions are called only when the script is executed directly
if __name__ == "__main__":
    plot_convergence("call_option", "European Call Option")
    plot_convergence("put_option", "European Put Option")