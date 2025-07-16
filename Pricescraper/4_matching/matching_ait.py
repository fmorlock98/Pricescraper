import pandas as pd
import os
from datetime import datetime

# Define the mapping dictionary for exact matches
MAPPING = {
    "alpha innotec Hybrox 11": "alpha innotec Hybrox 11",
    "NOVELAN Helox 11": "Novelan Luft Wasser Wärmepumpe-Helox 11, 10380601",
    "alpha innotec Hybrox 5": "Alpha Innotec air-to-water heat pump Hybrox outdoor unit, 5kW",
    "NOVELAN Helox 5": "Novelan Luft Wasser Wärmepumpe-Helox 5, 10380401",
    "alpha innotec Hybrox 8": "Alpha Innotec air-to-water heat pump Hybrox outdoor unit, 8kW",
    "NOVELAN Helox 8": "Novelan Luft Wasser Wärmepumpe-Helox 8, 10380501",
    "alpha innotec LWD 50A-HMD": "alpha innotec Luft-Wasser Wärmepumpe alira LWD 50A-HMD mit Hydraulikmodul 5,6 kW",
    "alpha innotec LWD 50A-HTD": "alpha innotec Luft-Wasser Wärmepumpe alira LWD 50A-HTD mit Hydrauliktower 5,6 kW",
    "NOVELAN LAD 5 - CSD": "Novelan Luft Wasser Wärmepumpe-LAD 5-CSD, 103601CSD22",
    "NOVELAN LAD 5 - HID": "Novelan Luft Wasser Wärmepumpe-LAD 5-HID, 103601HID22",
    "alpha innotec LWD 70A-HMD": "alpha innotec Luft-Wasser Wärmepumpe alira LWD 70A-HMD mit Hydraulikmodul 7,7 kW",
    "alpha innotec LWD 70A-HTD": "alpha innotec Luft-Wasser Wärmepumpe alira LWD 70A-HTD mit Hydrauliktower 7,7 kW",
    "NOVELAN LAD 7 - CSD": "Novelan Paket Luft/Wasser-Wärmepumpe LAD-7 7,7kW mit Compact Station Dual CSD 103602CSD22",
    "NOVELAN LAD 7 - HID": "Novelan Luft Wasser Wärmepumpe-LAD 7-HID, 103602HID22",
    "alpha innotec LWD 90A-HMD": "alpha innotec Luft-Wasser Wärmepumpe alira LWD 90A-HMD mit Hydraulikmodul 9 kW",
    "alpha innotec LWD 90A-HTD": "alpha innotec Luft-Wasser Wärmepumpe alira LWD 90A-HTD mit Hydrauliktower 9 kW",
    "NOVELAN LAD 9 - CSD": "Paket Novelan Luft/Wasser-Wärmepumpe LAD-9 9,0kW mit Compact Station Dual CSD 103609CSD22",
    "NOVELAN LAD 9 - HID": "Novelan Luft/Wasser-Wärmepumpe Dual LAD 9-HID, 9 kW mit Hydraulikmodul HID 1, 103609HID22",
    "alpha innotec LWDV 91-1/3-HDV 12-3": "alira V-line - LWDV bis 12 kW duale Luft/Wasser Wärmepumpe mit Hydraulikmodul",
    "alpha innotec LWDV 91-1/3-HDV 9-1/3": "alira V-line - LWDV 91-1/3-HDV 9-1/3 bis 9 kW duale Luft/Wasser Wärmepumpe mit Hydraulikmodul",
    "alpha innotec LWDV 91-1/3-HSDV 12M": "alpha innotec Luft-Wasser Wärmepumpe alira V-line - LWDV bis 12 kW mit Hydraulikstation",
    "alpha innotec LWDV-91-1/3-HSDV9 M1/3": "Luft-Wasser Wärmepumpe alira V-line - LWDV bis 9 kW mit Hydraulikstation",
    "NOVELAN LADV 9.1-1/3": "Novelan Luft-Wasser-Wärmepumpe LADV9.1-1/3, 230/400 V, Heizen, Außeneinheit, Dual, 10369901"
}

def load_and_prepare_data():
    # Read the source files
    hplib_file = 'data/interim/filtered_hplib_GbM/ait-deutschland_GmbH.csv'
    scraped_file = 'data/interim/ait-deutschland GmbH.csv'
    output_file = 'data/processed/matched_manufacturers/ait_matched.csv'
    
    # Read the files
    hplib_df = pd.read_csv(hplib_file)
    scraped_df = pd.read_csv(scraped_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load existing matched file if it exists, otherwise use hplib data
    if os.path.exists(output_file):
        matched_df = pd.read_csv(output_file)
    else:
        matched_df = hplib_df.copy()
    
    return hplib_df, scraped_df, matched_df, output_file

def update_match(row, scraped_data):
    """Update a row with scraped data if it's more recent"""
    match = scraped_data.iloc[0]
    
    # If no existing date or new date is more recent, update the data
    if pd.isna(row['Date_scraped']) or \
       (pd.notna(match['Date_scraped']) and 
        pd.to_datetime(match['Date_scraped']) > pd.to_datetime(row['Date_scraped'])):
        
        row['Model/Type'] = match['Model/Type']
        row['Price'] = match['Price']
        row['Currency'] = match['Currency']
        row['Website'] = match['Website']
        row['Date_scraped'] = match['Date_scraped']
    
    return row

def match_products(matched_df, scraped_df):
    """Match products based on Titel and Model/Type using the mapping dictionary"""
    matches_found = 0
    unmatched_items = []
    
    # Create a copy to avoid modifying during iteration
    result_df = matched_df.copy()
    
    for idx, row in result_df.iterrows():
        titel = row['Titel']
        if titel in MAPPING:
            # Look for match in scraped data using the mapping
            matching_items = scraped_df[scraped_df['Model/Type'] == MAPPING[titel]]
            
            if not matching_items.empty:
                # Update the row with scraped data
                result_df.iloc[idx] = update_match(row, matching_items)
                matches_found += 1
            else:
                unmatched_items.append(f"{titel} -> {MAPPING[titel]}")
        else:
            unmatched_items.append(titel)
    
    return result_df, matches_found, unmatched_items

def main():
    # Load data
    hplib_file = 'data/interim/filtered_hplib_GbM/ait-deutschland_GmbH.csv'
    scraped_file = 'data/interim/all_scraped_data_standardized.csv'
    output_file = 'data/processed/matched_manufacturers/ait_matched.csv'
    
    # Read the files
    hplib_df = pd.read_csv(hplib_file)
    scraped_df = pd.read_csv(scraped_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load existing matched file if it exists, otherwise use hplib data
    if os.path.exists(output_file):
        matched_df = pd.read_csv(output_file)
    else:
        matched_df = hplib_df.copy()
    
    # Perform matching
    result_df, matches_found, unmatched_items = match_products(matched_df, scraped_df)
    
    # Save results
    result_df.to_csv(output_file, index=False)
    
    # Print statistics
    print(f"\nMatching Statistics:")
    print(f"Total items processed: {len(result_df)}")
    print(f"Matches found: {matches_found}")
    print(f"Items without matches: {len(unmatched_items)}")
    
    if unmatched_items:
        print("\nUnmatched items:")
        for item in unmatched_items:
            print(f"- {item}")

if __name__ == "__main__":
    main() 