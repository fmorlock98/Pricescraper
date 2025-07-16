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

def strip_parentheses(s):
    """
    Remove text in parentheses from a string.
    Example: WH-SDC0309K3E5 (L Series) becomes WH-SDC0309K3E5
    """
    return re.sub(r'\(.*?\)', '', s).strip()

def find_matches(hplib_df, scraped_df):
    """Find matches between database and scraped data."""
    matches = []
    
    for _, hplib_row in hplib_df.iterrows():
        hplib_title = str(hplib_row['Titel']) if pd.notna(hplib_row['Titel']) else ""
        
        # Skip if no title
        if not hplib_title:
            continue
            
        # Process the database title
        clean_title = strip_parentheses(hplib_title)
        # Split by '/' to get different parts of the model code
        title_parts = [part.strip() for part in clean_title.split('/')]
        
        # Search for matches in scraped data
        for _, scraped_row in scraped_df.iterrows():
            scraped_model = clean_string(scraped_row['Model/Type'])
            
            # Skip if no model
            if not scraped_model:
                continue
            
            # Normalize scraped model for comparison
            scraped_model_normalized = normalize_model_code(scraped_model)
            
            # For database entries with multiple parts (e.g., "WH-SDC0509L3E5 / WH-WDG07LE5")
            # Both parts must be present in the scraped model
            if len(title_parts) > 1:
                all_parts_match = True
                for part in title_parts:
                    if not part:  # Skip empty parts
                        continue
                    
                    part_normalized = normalize_model_code(part)
                    if not part_normalized or part_normalized not in scraped_model_normalized:
                        all_parts_match = False
                        break
                
                if all_parts_match:
                    matches.append({
                        'scraped_row': scraped_row,
                        'matching_hplib_rows': [hplib_row],
                        'is_combined': False
                    })
                    break
            # For database entries with a single part
            else:
                part = title_parts[0]
                if not part:  # Skip empty parts
                    continue
                
                part_normalized = normalize_model_code(part)
                if part_normalized and part_normalized in scraped_model_normalized:
                    matches.append({
                        'scraped_row': scraped_row,
                        'matching_hplib_rows': [hplib_row],
                        'is_combined': False
                    })
                    break
    
    return matches

def find_single_matches(unmatched_hplib_df, scraped_df, already_used_scraped_rows):
    """Find matches for single model codes in unmatched database items."""
    matches = []
    
    for _, hplib_row in unmatched_hplib_df.iterrows():
        hplib_title = str(hplib_row['Titel']) if pd.notna(hplib_row['Titel']) else ""
        
        # Skip if no title
        if not hplib_title:
            continue
        
        # Process the database title
        clean_title = strip_parentheses(hplib_title)
        # Split by '/' to get different parts of the model code
        title_parts = [part.strip() for part in clean_title.split('/')]
        
        # For database entries with a single part
        if len(title_parts) == 1:
            part = title_parts[0]
            if not part:  # Skip empty parts
                continue
            
            part_normalized = normalize_model_code(part)
            
            # Search for matches in scraped data
            for idx, scraped_row in scraped_df.iterrows():
                # Skip already used scraped rows
                if idx in already_used_scraped_rows:
                    continue
                
                scraped_model = clean_string(scraped_row['Model/Type'])
                
                # Skip if no model
                if not scraped_model:
                    continue
                
                scraped_model_normalized = normalize_model_code(scraped_model)
                
                if part_normalized and part_normalized in scraped_model_normalized:
                    matches.append({
                        'scraped_row': scraped_row,
                        'matching_hplib_rows': [hplib_row],
                        'is_combined': False
                    })
                    already_used_scraped_rows.add(idx)
                    break
    
    return matches, already_used_scraped_rows

def find_combined_matches(unmatched_hplib_df, scraped_df, already_used_scraped_rows):
    """Find and combine matches for split parts of database items."""
    matches = []
    
    for _, hplib_row in unmatched_hplib_df.iterrows():
        hplib_title = str(hplib_row['Titel']) if pd.notna(hplib_row['Titel']) else ""
        
        # Skip if no title
        if not hplib_title:
            continue
        
        # Process the database title
        clean_title = strip_parentheses(hplib_title)
        # Split by '/' to get different parts of the model code
        title_parts = [part.strip() for part in clean_title.split('/')]
        
        # Only process items with multiple parts
        if len(title_parts) <= 1:
            continue
        
        # Try to find matches for each part
        part_matches = []
        
        for part in title_parts:
            if not part:  # Skip empty parts
                continue
            
            part_normalized = normalize_model_code(part)
            best_match = None
            
            # Search for matches in scraped data
            for idx, scraped_row in scraped_df.iterrows():
                # Skip already used scraped rows
                if idx in already_used_scraped_rows:
                    continue
                
                scraped_model = clean_string(scraped_row['Model/Type'])
                
                # Skip if no model
                if not scraped_model:
                    continue
                
                scraped_model_normalized = normalize_model_code(scraped_model)
                
                if part_normalized and part_normalized in scraped_model_normalized:
                    best_match = (idx, scraped_row)
                    break
            
            if best_match:
                part_matches.append(best_match)
        
        # If we found matches for all parts, combine them
        if len(part_matches) == len([p for p in title_parts if p]):
            # Mark these scraped rows as used
            for idx, _ in part_matches:
                already_used_scraped_rows.add(idx)
            
            # Create a combined model string and price
            combined_model = " & ".join([match[1]['Model/Type'] for match in part_matches])
            combined_price = sum([float(match[1]['Price']) if pd.notna(match[1]['Price']) else 0 for match in part_matches])
            individual_prices = [f"{match[1]['Price']} {match[1]['Currency']}" for match in part_matches if pd.notna(match[1]['Price'])]
            combined_website = " & ".join([f"{match[1]['Website']} ({price})" for match, price in zip(part_matches, individual_prices)])
            
            # Create a combined scraped row
            combined_row = part_matches[0][1].copy()
            combined_row['Model/Type'] = combined_model
            combined_row['Price'] = combined_price
            combined_row['Website'] = combined_website
            
            matches.append({
                'scraped_row': combined_row,
                'matching_hplib_rows': [hplib_row],
                'is_combined': True
            })
    
    return matches, already_used_scraped_rows

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
    hplib_file = 'data/interim/filtered_hplib_GbM/Panasonic_Marketing_Europe_GmbH.csv'
    scraped_file = 'data/interim/scraped_data_standardized_GbM/Panasonic Marketing Europe GmbH.csv'
    output_file = 'data/processed/matched_manufacturers/panasonic_matched.csv'
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Read the files
    hplib_df = pd.read_csv(hplib_file)
    scraped_df = pd.read_csv(scraped_file)
    
    print("=== First Pass: Find Direct Matches ===")
    # First pass: find direct matches
    matches = find_matches(hplib_df, scraped_df)
    
    # Track which hplib titles have already been matched
    matched_hplib_titles = set()
    # Track which scraped rows have already been used
    used_scraped_indices = set()
    
    for match in matches:
        for hplib_row in match['matching_hplib_rows']:
            matched_hplib_titles.add(hplib_row['Titel'])
        
        # Get the index of the scraped row
        for idx, row in scraped_df.iterrows():
            if row.equals(match['scraped_row']):
                used_scraped_indices.add(idx)
                break
    
    # Find unmatched items
    unmatched_hplib = hplib_df[~hplib_df['Titel'].isin(matched_hplib_titles)]
    
    print("\n=== Second Pass: Find Single Matches ===")
    # Second pass: match single model codes in unmatched database items
    single_matches, used_scraped_indices = find_single_matches(unmatched_hplib, scraped_df, used_scraped_indices)
    matches.extend(single_matches)
    
    # Update matched titles
    for match in single_matches:
        for hplib_row in match['matching_hplib_rows']:
            matched_hplib_titles.add(hplib_row['Titel'])
    
    # Find remaining unmatched items
    unmatched_hplib = hplib_df[~hplib_df['Titel'].isin(matched_hplib_titles)]
    
    print("\n=== Third Pass: Find Combined Matches ===")
    # Third pass: find and combine matches for split parts
    combined_matches, _ = find_combined_matches(unmatched_hplib, scraped_df, used_scraped_indices)
    matches.extend(combined_matches)
    
    # Update matched titles
    for match in combined_matches:
        for hplib_row in match['matching_hplib_rows']:
            matched_hplib_titles.add(hplib_row['Titel'])
    
    # Find remaining unmatched items
    unmatched_hplib = hplib_df[~hplib_df['Titel'].isin(matched_hplib_titles)]
    
    # Create output dataframe
    result_df = create_output_dataframe(matches)
    
    # Save results
    result_df.to_csv(output_file, index=False)
    
    # Print statistics
    print("\n=== Matching Statistics ===")
    print(f"Total items in database: {len(hplib_df)}")
    print(f"Direct matches found: {len(matches) - len(single_matches) - len(combined_matches)}")
    print(f"Single model matches found: {len(single_matches)}")
    print(f"Combined matches found: {len(combined_matches)}")
    print(f"Total matches found: {len(matches)}")
    print(f"Unmatched database items: {len(unmatched_hplib)}")
    
    # Print unmatched database items
    print("\n=== Unmatched Database Items ===")
    for _, row in unmatched_hplib.iterrows():
        print(f"- {row['Titel']}")

if __name__ == "__main__":
    main() 