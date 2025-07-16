import pandas as pd
import re
import os
from pathlib import Path

def clean_string(s):
    """Clean string by removing extra spaces and standardizing separators."""
    if pd.isna(s):
        return ""
    # Replace multiple spaces with single space
    s = re.sub(r'\s+', ' ', str(s))
    # Convert commas to forward slashes
    s = s.replace(',', '/')
    # Remove dots and standardize other separators
    s = s.replace('.', ' ').replace('+', ' ')
    return s.strip()

def normalize_model_code(s):
    """Normalize model code by removing spaces and standardizing format."""
    if pd.isna(s):
        return ""
    # Remove all spaces and convert to uppercase
    s = re.sub(r'\s+', '', str(s))
    # Remove dots and standardize format
    s = s.replace('.', '')
    return s.upper()

def extract_model_codes(s):
    """
    Extract model codes like RAS-XXXXX, RASM-XXXXX, RWM/RWD-XXXXX from a string.
    Returns a list of extracted full model codes.
    """
    if pd.isna(s):
        return []
    
    # For debugging, print the input string
    print(f"Extracting from: {s}")
    
    # Pattern for Hitachi model codes (typical formats like RAS-2WHVRP1, RWM-3.0R1E, RASM-3VRE, etc.)
    # This regex captures the full model code as a single group, not just the prefix
    pattern = r'((?:RAS|RASM|RWM|RWD|RAW|TAW|DHWS|DWHS)-[0-9A-Za-z\.\-]+)'
    
    # Find all matches
    matches = re.findall(pattern, str(s))
    
    # For debugging, print what we found
    print(f"Found model codes: {matches}")
    
    return matches

def strip_parentheses(s):
    """
    Remove text in parentheses from a string.
    Example: RAS-4WHVNPE (Model) becomes RAS-4WHVNPE
    """
    return re.sub(r'\(.*?\)', '', s).strip()

def find_matches(hplib_df, scraped_df):
    """Find direct matches between database and scraped data."""
    matches = []
    
    for _, hplib_row in hplib_df.iterrows():
        hplib_title = str(hplib_row['Titel']) if pd.notna(hplib_row['Titel']) else ""
        
        # Skip if no title
        if not hplib_title:
            continue
        
        # Extract model codes from the database title
        print(f"\nChecking database item: {hplib_title}")
        hplib_codes = extract_model_codes(hplib_title)
        
        if not hplib_codes:
            print(f"No model codes found in: {hplib_title}, skipping")
            continue
        
        # Search for matches in scraped data
        for _, scraped_row in scraped_df.iterrows():
            scraped_model = str(scraped_row['Model/Type']) if pd.notna(scraped_row['Model/Type']) else ""
            
            # Skip if no model
            if not scraped_model:
                continue
            
            # Extract model codes from scraped model
            scraped_codes = extract_model_codes(scraped_model)
            
            if not scraped_codes:
                continue
            
            # Check if all database codes are found exactly in the scraped codes
            # This requires a direct string comparison without normalization
            all_codes_matched = True
            
            # We need to verify that all database codes exist exactly in the scraped codes
            for db_code in hplib_codes:
                code_found = False
                for sc_code in scraped_codes:
                    if db_code == sc_code:  # Exact string match, no normalization
                        code_found = True
                        break
                
                if not code_found:
                    all_codes_matched = False
                    print(f"No match for DB code {db_code} in scraped codes {scraped_codes}")
                    break
            
            if all_codes_matched:
                # For extra verification, print what is being matched
                print(f"MATCH FOUND: {hplib_title} with {scraped_model}")
                matches.append({
                    'scraped_row': scraped_row,
                    'matching_hplib_rows': [hplib_row],
                    'is_combined': False
                })
                break
    
    return matches

def create_output_dataframe(matches):
    """Create output dataframe from matches."""
    if not matches:
        return pd.DataFrame()
    
    # Create list of rows for the output dataframe
    rows = []
    for match in matches:
        scraped_row = match['scraped_row']
        hplib_row = match['matching_hplib_rows'][0]
        
        # Create row with all database columns
        row = hplib_row.to_dict()
        
        # Add scraped data columns
        row.update({
            'Model/Type': scraped_row['Model/Type'],
            'Price': scraped_row['Price'],
            'Currency': scraped_row['Currency'],
            'Website': scraped_row['Website'],
            'Date_scraped': scraped_row['Date_scraped']
        })
        
        rows.append(row)
    
    # Create dataframe
    df = pd.DataFrame(rows)
    
    # Define the order of columns
    columns = [
        # Database columns
        'Manufacturer', 'Model', 'Titel', 'Date',
        # Scraped data columns
        'Model/Type', 'Price', 'Currency', 'Website', 'Date_scraped',
        # Remaining database columns
        'Type', 'Subtype', 'Group', 'Rated Power low T [kW]', 'Rated Power medium T [kW]',
        'Refrigerant', 'Mass of Refrigerant [kg]', 'SPL indoor low Power [dBA]',
        'SPL outdoor low Power [dBA]', 'SPL indoor high Power [dBA]', 'SPL outdoor high Power [dBA]',
        'Bivalence temperature [°C]', 'Tolerance temperature [°C]', 'Max. water heating temperature [°C]',
        'Power heating rod low T [kW]', 'Power heating rod medium T [kW]', 'Poff [W]', 'PTOS [W]',
        'PSB [W]', 'PCKS [W]', 'eta low T [%]', 'eta medium T [%]', 'SCOP', 'SEER low T',
        'SEER medium T', 'P_th_h_ref [W]', 'P_th_c_ref [W]', 'P_el_h_ref [W]', 'P_el_c_ref [W]',
        'COP_ref', 'EER_c_ref'
    ]
    
    # Reorder columns and handle missing columns
    existing_columns = [col for col in columns if col in df.columns]
    df = df[existing_columns]
    
    return df

def main():
    # Load data
    hplib_file = 'data/interim/filtered_hplib_GbM/Johnson_Hitachi.csv'
    scraped_file = 'data/interim/scraped_data_standardized_GbM/Johnson Controls Hitachi Air-Conditioning Europe SAS Johnson Controls Industries Johnson Controls-Hitachi AirConditioning Spain.csv'
    output_file = 'data/processed/matched_manufacturers/johnson_hitachi_matched.csv'
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Read the files
    hplib_df = pd.read_csv(hplib_file)
    scraped_df = pd.read_csv(scraped_file)
    
    print("\n=== Looking for exact model code matches ===")
    # Find matches
    matches = find_matches(hplib_df, scraped_df)
    
    # Create output dataframe
    result_df = create_output_dataframe(matches)
    
    # Save results
    result_df.to_csv(output_file, index=False)
    
    # Track which hplib titles have already been matched
    matched_hplib_titles = set()
    for match in matches:
        for hplib_row in match['matching_hplib_rows']:
            matched_hplib_titles.add(hplib_row['Titel'])
    
    # Find unmatched items
    unmatched_hplib = hplib_df[~hplib_df['Titel'].isin(matched_hplib_titles)]
    
    # Print statistics
    print("\n=== Matching Statistics ===")
    print(f"Total items in database: {len(hplib_df)}")
    print(f"Total matches found: {len(matches)}")
    print(f"Unmatched database items: {len(unmatched_hplib)}")
    
    # Print ALL unmatched database items
    print("\n=== ALL Unmatched Database Items ===")
    for _, row in unmatched_hplib.iterrows():
        print(f"- {row['Titel']}")

if __name__ == "__main__":
    main() 