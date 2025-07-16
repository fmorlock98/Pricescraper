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

def wildcard_to_regex(pattern):
    """
    Convert wildcard pattern to regex pattern.
    * matches exactly one character (letter or number)
    Example: E*ST20D-*M*D becomes E[A-Z0-9]ST20D-[A-Z0-9]M[A-Z0-9]D
    """
    # Escape all regex special characters first
    pattern = re.escape(pattern)
    
    # Replace \* (escaped *) with [A-Z0-9] to match exactly one character
    pattern = pattern.replace('\\*', '[A-Z0-9]')
    
    return pattern

def strip_parentheses(s):
    """
    Remove text in parentheses from a string.
    Example: PUD-SWM100YAA(-BS) becomes PUD-SWM100YAA
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
            
        # Split by '+' to get outdoor and indoor units
        parts = [part.strip() for part in hplib_title.split('+')]
        
        # Skip if not exactly two parts
        if len(parts) != 2:
            continue
            
        # Handle the outdoor unit (first part)
        outdoor_unit = strip_parentheses(parts[0])
        # Create a normalized version for matching
        outdoor_unit_normalized = normalize_model_code(outdoor_unit)
        
        # Handle the indoor unit pattern (second part)
        indoor_unit_pattern = parts[1].strip()
        # Convert wildcards to regex pattern that matches exactly one character per *
        indoor_regex = wildcard_to_regex(indoor_unit_pattern)
        
        # Search for matches in scraped data
        for _, scraped_row in scraped_df.iterrows():
            scraped_model = clean_string(scraped_row['Model/Type'])
            
            # Skip if no model
            if not scraped_model:
                continue
                
            # Skip items containing "Kaskaden"
            if "Kaskaden" in scraped_model:
                continue
                
            # Normalize scraped model for comparison
            scraped_model_normalized = normalize_model_code(scraped_model)
            
            # For outdoor unit, check for exact match (not substring)
            # Split the scraped model into words to check for exact matches
            scraped_parts = re.split(r'[ +&,]', scraped_model)
            scraped_parts_normalized = [normalize_model_code(part) for part in scraped_parts]
            
            # Check if the exact outdoor unit is in the scraped parts
            outdoor_match = False
            for part in scraped_parts_normalized:
                # Check for exact match of outdoor unit
                if part == outdoor_unit_normalized:
                    outdoor_match = True
                    break
            
            if not outdoor_match:
                continue
            
            # Check if indoor unit pattern matches in scraped model
            # Compile the regex pattern for case-insensitive matching
            indoor_regex_pattern = re.compile(indoor_regex, re.IGNORECASE)
            
            # Look for the indoor unit pattern in the scraped model parts
            indoor_match = False
            for part in scraped_parts:
                if indoor_regex_pattern.fullmatch(part):
                    indoor_match = True
                    break
            
            # If both outdoor and indoor units match, add to matches
            if outdoor_match and indoor_match:
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
    hplib_file = 'data/interim/filtered_hplib_GbM/Mitsubishi_Electric_Air_Conditioning_Systems_Europe_LTD.csv'
    scraped_file = 'data/interim/scraped_data_standardized_GbM/Mitsubishi Electric Air Conditioning Systems Europe LTD.csv'
    output_file = 'data/processed/matched_manufacturers/mitsubishi_matched.csv'
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Read the files
    hplib_df = pd.read_csv(hplib_file)
    scraped_df = pd.read_csv(scraped_file)
    
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
    
    # Print unmatched database items
    print("\n=== Unmatched Database Items ===")
    for _, row in unmatched_hplib.iterrows():
        print(f"- {row['Titel']}")

if __name__ == "__main__":
    main() 