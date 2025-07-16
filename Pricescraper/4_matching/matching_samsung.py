import pandas as pd
import re
import os
from pathlib import Path

def normalize_string(s):
    """
    Normalize a string by removing spaces, +, &, and converting to uppercase.
    Also removes any text in parentheses.
    """
    if pd.isna(s):
        return ""
    
    # Remove text in parentheses
    s = re.sub(r'\(.*?\)', '', str(s))
    
    # Remove spaces, +, &, and convert to uppercase
    s = s.replace(' ', '').replace('+', '').replace('&', '').replace('/', '').upper()
    
    # Remove /EU suffix
    s = re.sub(r'/EU$', '', s)
    
    return s

def find_matches(hplib_df, scraped_df):
    """
    Find matches by checking if all parts of the normalized Titel 
    exist in the normalized Model/Type.
    """
    matches = []
    
    for _, hplib_row in hplib_df.iterrows():
        hplib_title = str(hplib_row['Titel']) if pd.notna(hplib_row['Titel']) else ""
        
        # Skip if no title
        if not hplib_title:
            continue
        
        # Normalize the database title
        normalized_title = normalize_string(hplib_title)
        
        # Get individual parts (split by +)
        title_parts = [part.strip() for part in hplib_title.split('+')]
        normalized_parts = [normalize_string(part) for part in title_parts if part.strip()]
        
        if not normalized_parts:
            continue
        
        # Search for matches in scraped data
        for _, scraped_row in scraped_df.iterrows():
            scraped_model = str(scraped_row['Model/Type']) if pd.notna(scraped_row['Model/Type']) else ""
            
            # Skip if no model
            if not scraped_model:
                continue
            
            # Normalize the scraped model
            normalized_model = normalize_string(scraped_model)
            
            # Check if all parts of the title are in the model
            all_parts_match = True
            for part in normalized_parts:
                if part and part not in normalized_model:
                    all_parts_match = False
                    break
            
            if all_parts_match:
                matches.append({
                    'scraped_row': scraped_row,
                    'matching_hplib_rows': [hplib_row]
                })
                break  # Move to next database item once a match is found
    
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
    hplib_file = 'data/interim/filtered_hplib_GbM/Samsung_Electronics_Air_Conditioner_Europe_B.V..csv'
    scraped_file = 'data/interim/scraped_data_standardized_GbM/Samsung Electronics Air Conditioner Europe BV.csv'
    output_file = 'data/processed/matched_manufacturers/samsung_matched.csv'
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Read the files
    hplib_df = pd.read_csv(hplib_file)
    scraped_df = pd.read_csv(scraped_file)
    
    print("=== Finding Matches ===")
    # Find matches using our simplified approach
    matches = find_matches(hplib_df, scraped_df)
    
    # Create output dataframe
    result_df = create_output_dataframe(matches)
    
    # Save results
    result_df.to_csv(output_file, index=False)
    
    # Print statistics
    print("\n=== Matching Statistics ===")
    print(f"Total items in database: {len(hplib_df)}")
    print(f"Total matches found: {len(matches)}")
    
    # Find unmatched database items
    matched_hplib_titles = {match['matching_hplib_rows'][0]['Titel'] for match in matches}
    unmatched_hplib = hplib_df[~hplib_df['Titel'].isin(matched_hplib_titles)]
    print(f"Unmatched database items: {len(unmatched_hplib)}")
    
    # Print unmatched database items
    print("\n=== Unmatched Database Items ===")
    for _, row in unmatched_hplib.iterrows():
        print(f"- {row['Titel']}")

if __name__ == "__main__":
    main() 