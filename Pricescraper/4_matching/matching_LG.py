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
    # For PHCS0 control units, normalize to PHCS0
    if 'PHCS0' in s:
        # If it's just the PHCS0 part, return only PHCS0
        if len(s) < 10:  # PHCS0 + a few extra chars at most
            return 'PHCS0'
    return s.upper()

def is_set_product(s):
    """Check if a product is a set product containing multiple parts."""
    if pd.isna(s):
        return False
    s = s.lower()
    return 'set' in s

def is_multipart_product(s):
    """Check if a product contains multiple parts."""
    if pd.isna(s):
        return False
    s = s.lower()
    return '+' in s or '&' in s or 'set' in s or 'bibloc' in s

def find_combined_matches(hplib_df, scraped_df):
    """Find matches for combined products (indoor + outdoor units)."""
    matches = []
    
    # First create a lookup for the specific PHCS0 matches
    phcs0_matches = {
        "HM071HFUB40": "LG THERMA V Set Monobloc HM071HF.UB40 + PHCS0 Kontrolleinheit",
        "HM091HFUB40": "LG THERMA V Set Monobloc HM091HF.UB40 + PHCS0 Kontrolleinheit",
        "HM093HFUB40": "LG THERMA V Set Monobloc HM093HF.UB40 + PHCS0 Kontrolleinheit",
        "HM123HFUB60": "LG THERMA V Set Monobloc HM123HF.UB60 + PHCS0 Kontrolleinheit",
        "HM143HFUB60": "LG THERMA V Set Monobloc HM143HF.UB60 + PHCS0 Kontrolleinheit",
        "HM163HFUB60": "LG THERMA V Set Monobloc HM163HF.UB60 + PHCS0 Kontrolleinheit"
    }
    
    for _, hplib_row in hplib_df.iterrows():
        hplib_title = clean_string(hplib_row['Titel'])
        hplib_model = hplib_row['Model']
        
        # Skip if no title or model
        if pd.isna(hplib_title) or pd.isna(hplib_model):
            continue
            
        # Check if this hplib row contains a comma - if so, it already has multiple parts, skip combination
        if ',' in hplib_title:
            continue
            
        # Split the model codes by '/'
        model_parts = [part.strip() for part in hplib_title.split('/')]
        
        if len(model_parts) != 2:
            continue
            
        outdoor_code = normalize_model_code(model_parts[0])
        indoor_code = normalize_model_code(model_parts[1])
        
        # Special handling for PHCS0 matches
        if indoor_code == 'PHCS0':
            # Create a normalized key for lookup
            outdoor_key = outdoor_code.replace(' ', '')
            
            direct_match_found = False
            
            # Check for the specific matches first
            if outdoor_key in phcs0_matches:
                phcs0_pattern = phcs0_matches[outdoor_key]
                
                # Look for this pattern in the scraped data
                for _, scraped_row in scraped_df.iterrows():
                    scraped_model = clean_string(scraped_row['Model/Type'])
                    if pd.isna(scraped_model):
                        continue
                    
                    # Check if this scraped model matches our pattern
                    if phcs0_pattern.lower() in scraped_model.lower():
                        matches.append({
                            'scraped_row': scraped_row,
                            'matching_hplib_rows': [hplib_row],
                            'is_combined': False
                        })
                        direct_match_found = True
                        break
            
            # If no specific match found, try the general approach
            if not direct_match_found:
                for _, scraped_row in scraped_df.iterrows():
                    scraped_model = clean_string(scraped_row['Model/Type'])
                    if pd.isna(scraped_model):
                        continue
                        
                    normalized_scraped = normalize_model_code(scraped_model)
                    
                    # Check if this product contains both the outdoor unit and PHCS0
                    if 'PHCS0' in normalized_scraped and outdoor_code in normalized_scraped:
                        matches.append({
                            'scraped_row': scraped_row,
                            'matching_hplib_rows': [hplib_row],
                            'is_combined': False
                        })
                        direct_match_found = True
                        break
            
            # Skip to next hplib row - PHCS0 items are only matched directly
            continue
            
        direct_match_found = False
        
        # Look for direct matches first
        for _, scraped_row in scraped_df.iterrows():
            scraped_model = clean_string(scraped_row['Model/Type'])
            if pd.isna(scraped_model):
                continue
                
            normalized_scraped = normalize_model_code(scraped_model)
            
            # For HU12/14/16 series, need to check both outdoor and indoor codes match
            if outdoor_code.startswith(('HU12', 'HU14', 'HU16')):
                # Only match if it contains both the outdoor and indoor unit codes
                if outdoor_code in normalized_scraped and indoor_code in normalized_scraped:
                    matches.append({
                        'scraped_row': scraped_row,
                        'matching_hplib_rows': [hplib_row],
                        'is_combined': False
                    })
                    direct_match_found = True
                    break
        
        # If direct match found, continue to next item
        if direct_match_found:
            continue
            
        # Skip HU12/14/16 series for combination as they should have exact matches
        if outdoor_code.startswith(('HU12', 'HU14', 'HU16')):
            continue
            
        # If no direct match found, try the combination method for other models
        outdoor_matches = []
        indoor_matches = []
        
        for _, scraped_row in scraped_df.iterrows():
            scraped_model = clean_string(scraped_row['Model/Type'])
            if pd.isna(scraped_model):
                continue
            
            # Skip multipart products for combination (any product with +, &, set, or bibloc)
            if is_multipart_product(scraped_model):
                continue
                
            normalized_scraped = normalize_model_code(scraped_model)
            
            # Check for outdoor unit match - make sure it doesn't already contain the indoor unit
            if outdoor_code in normalized_scraped and indoor_code not in normalized_scraped:
                outdoor_matches.append(scraped_row)
            
            # Check for indoor unit match - make sure it doesn't already contain the outdoor unit
            if indoor_code in normalized_scraped and outdoor_code not in normalized_scraped:
                indoor_matches.append(scraped_row)
        
        # If we found both indoor and outdoor matches, create a combined match
        if outdoor_matches and indoor_matches:
            # Use the first matches found
            outdoor_match = outdoor_matches[0]
            indoor_match = indoor_matches[0]
            
            # Calculate combined price
            combined_price = outdoor_match['Price'] + indoor_match['Price']
            
            # Create a combined row
            combined_row = {
                'Model/Type': f"{outdoor_match['Model/Type']} & {indoor_match['Model/Type']}",
                'Price': combined_price,
                'Currency': outdoor_match['Currency'],
                'Website': f"{outdoor_match['Website']} ({outdoor_match['Price']} {outdoor_match['Currency']}) & {indoor_match['Website']} ({indoor_match['Price']} {indoor_match['Currency']})",
                'Date_scraped': max(outdoor_match['Date_scraped'], indoor_match['Date_scraped'])
            }
            
            matches.append({
                'scraped_row': combined_row,
                'matching_hplib_rows': [hplib_row],
                'is_combined': True
            })
    
    return matches

def find_single_matches(hplib_df, scraped_df):
    """Find matches for single products."""
    matches = []
    
    for _, hplib_row in hplib_df.iterrows():
        hplib_title = clean_string(hplib_row['Titel'])
        hplib_model = hplib_row['Model']
        
        # Skip if no title or model
        if pd.isna(hplib_title) or pd.isna(hplib_model):
            continue
            
        # Skip if it's a combined product (contains '/')
        if '/' in hplib_title:
            continue
            
        # Try to find exact matches
        for _, scraped_row in scraped_df.iterrows():
            scraped_model = clean_string(scraped_row['Model/Type'])
            
            # Skip if no model
            if pd.isna(scraped_model):
                continue
                
            # Skip if it's a set product
            if is_set_product(scraped_model):
                continue
                
            # Check if the model codes match
            hplib_codes = [normalize_model_code(code.strip()) for code in hplib_title.split()]
            scraped_codes = [normalize_model_code(code.strip()) for code in scraped_model.split()]
            
            # Check if all codes from hplib are in scraped data
            if all(code in scraped_codes for code in hplib_codes if code):
                matches.append({
                    'scraped_row': scraped_row,
                    'matching_hplib_rows': [hplib_row],
                    'is_combined': False
                })
                break
    
    return matches

def find_matches(hplib_df, scraped_df):
    """Find all matches between database and scraped data."""
    # Find combined matches first
    combined_matches = find_combined_matches(hplib_df, scraped_df)
    
    # Find single matches
    single_matches = find_single_matches(hplib_df, scraped_df)
    
    # Combine all matches
    return combined_matches + single_matches

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
    hplib_file = 'data/interim/filtered_hplib_GbM/LG_Electronics_Inc..csv'
    scraped_file = 'data/interim/scraped_data_standardized_GbM/LG Electronics Inc.csv'
    output_file = 'data/processed/matched_manufacturers/lg_matched.csv'
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Read the files
    hplib_df = pd.read_csv(hplib_file)
    scraped_df = pd.read_csv(scraped_file)
    
    # Clean database titles by removing ENCXLEU
    hplib_df['Titel'] = hplib_df['Titel'].apply(lambda x: str(x).replace('ENCXLEU', '') if pd.notna(x) else x)
    
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