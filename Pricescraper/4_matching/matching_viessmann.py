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
    s = s.replace('+', ' ')
    return s.strip()

def normalize_model_code(s):
    """Normalize model code by removing spaces and standardizing format."""
    if pd.isna(s):
        return ""
    # Remove all spaces and convert to uppercase
    s = re.sub(r'\s+', '', str(s))
    return s.upper()

def strip_parentheses(s):
    """
    Remove text in parentheses from a string.
    Example: Vitocal 252-A (123456) becomes Vitocal 252-A
    """
    return re.sub(r'\(.*?\)', '', s).strip()

def find_matches(hplib_df, scraped_df):
    """Find matches between database and scraped data."""
    matches = []
    
    # Define model variant indicators
    variant_indicators = ["2C", "NEV", "SP"]
    
    for _, hplib_row in hplib_df.iterrows():
        hplib_title = str(hplib_row['Titel']) if pd.notna(hplib_row['Titel']) else ""
        
        # Skip if no title
        if not hplib_title:
            continue
            
        # Process the database title
        clean_title = strip_parentheses(hplib_title)
        
        # Check if database title contains any variant indicators
        db_variants = []
        for indicator in variant_indicators:
            pattern = r'\b' + re.escape(indicator) + r'\b'
            if re.search(pattern, clean_title):
                db_variants.append(indicator)
        
        # Track all possible matches for this database item
        possible_matches = []
        
        # Search for matches in scraped data
        for _, scraped_row in scraped_df.iterrows():
            scraped_model = clean_string(scraped_row['Model/Type'])
            
            # Skip if no model
            if not scraped_model:
                continue
            
            # Check if scraped model contains all the variant indicators from the database title
            all_variants_present = True
            if db_variants:
                for indicator in db_variants:
                    pattern = r'\b' + re.escape(indicator) + r'\b'
                    if not re.search(pattern, scraped_model):
                        all_variants_present = False
                        break
                
                if not all_variants_present:
                    continue
            else:
                # If database has no variants, check if scraped model has any variants
                has_unwanted_variants = False
                for indicator in variant_indicators:
                    pattern = r'\b' + re.escape(indicator) + r'\b'
                    if re.search(pattern, scraped_model):
                        has_unwanted_variants = True
                        break
                
                if has_unwanted_variants:
                    continue
            
            # Check if all elements of the title are in the scraped model
            title_elements = clean_title.split()
            all_elements_match = True
            
            for element in title_elements:
                if element.strip() and element.strip().lower() not in scraped_model.lower():
                    all_elements_match = False
                    break
            
            if all_elements_match:
                possible_matches.append(scraped_row)
        
        # If we found multiple matches, choose the one with the lowest price
        if possible_matches:
            # Filter out rows with NaN prices
            valid_price_matches = [match for match in possible_matches if pd.notna(match['Price'])]
            
            if valid_price_matches:
                # Sort by price and select the lowest
                best_match = min(valid_price_matches, key=lambda x: float(x['Price']))
            else:
                # If no valid prices, just take the first match
                best_match = possible_matches[0]
            
            matches.append({
                'scraped_row': best_match,
                'matching_hplib_rows': [hplib_row],
                'is_combined': False
            })
    
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
    hplib_file = 'data/interim/filtered_hplib_GbM/Viessmann_Climate_Solutions_SE.csv'
    scraped_file = 'data/interim/scraped_data_standardized_GbM/Viessmann Climate Solutions SE.csv'
    output_file = 'data/processed/matched_manufacturers/viessmann_matched.csv'
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Read the files
    hplib_df = pd.read_csv(hplib_file)
    scraped_df = pd.read_csv(scraped_file)
    
    print("=== Finding Matches ===")
    # Find matches by checking if elements of Titel are in Model/Type
    matches = find_matches(hplib_df, scraped_df)
    
    # Track which hplib titles have already been matched
    matched_hplib_titles = set()
    
    for match in matches:
        for hplib_row in match['matching_hplib_rows']:
            matched_hplib_titles.add(hplib_row['Titel'])
    
    # Find unmatched items
    unmatched_hplib = hplib_df[~hplib_df['Titel'].isin(matched_hplib_titles)]
    
    # Create output dataframe
    result_df = create_output_dataframe(matches)
    
    # Save results
    result_df.to_csv(output_file, index=False)
    
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