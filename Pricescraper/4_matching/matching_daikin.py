import pandas as pd
import re
import os
from pathlib import Path

def normalize_string(s):
    """
    Normalize a string by removing spaces, hyphens, and converting to uppercase.
    """
    if pd.isna(s):
        return ""
    
    # Remove spaces, hyphens, and convert to uppercase
    return str(s).replace(' ', '').replace('-', '').upper()

def expand_variants_with_brackets(code):
    """
    Expand code patterns with brackets to create all possible combinations
    For example: "ETSH(B)16P50E(7)" will create:
    ["ETSH16P50E", "ETSHB16P50E", "ETSH16P50E7", "ETSHB16P50E7"]
    
    Handles special cases like "(6V/9W)" to create alternative options (6V or 9W)
    """
    if '(' not in code:
        return [code]
    
    # Find all bracket patterns
    bracket_patterns = re.findall(r'\(([^)]+)\)', code)
    if not bracket_patterns:
        return [code]
    
    # Base pattern with placeholders
    base = re.sub(r'\([^)]+\)', '{}', code)
    
    # Generate combinations
    combinations = []
    
    # Process each bracket pattern to handle slashes within them
    expanded_patterns = []
    for pattern in bracket_patterns:
        if '/' in pattern:
            # If pattern contains alternatives like "6V/9W", split them
            alternatives = [alt.strip() for alt in pattern.split('/')]
            expanded_patterns.append(alternatives + [''])  # Include empty option
        else:
            # Regular pattern, can be included or not
            expanded_patterns.append([pattern, ''])
    
    # Generate all combinations using the expanded patterns
    def generate_combinations(base_str, pattern_idx, expanded_patterns):
        if pattern_idx >= len(expanded_patterns):
            combinations.append(base_str)
            return
        
        for alternative in expanded_patterns[pattern_idx]:
            new_base = base_str.replace('{}', alternative, 1)
            generate_combinations(new_base, pattern_idx + 1, expanded_patterns)
    
    generate_combinations(base, 0, expanded_patterns)
    
    return combinations

def clean_title(title):
    """
    Clean the title by removing "+ cooling kit" and similar additions
    """
    if pd.isna(title):
        return ""
    
    title = str(title).strip()
    
    # Remove "+ cooling kit" and similar text
    title = re.sub(r'\s*\+\s*cooling\s+kit\s*', '', title, flags=re.IGNORECASE)
    
    return title

def process_model_code(title):
    """
    Process model codes from the Titel column
    Handles split units separated by "/" and optional parts in brackets
    """
    if pd.isna(title):
        return []
    
    # Clean the title first
    title = clean_title(title)
    
    # Split by "/" to separate indoor and outdoor units
    parts = [part.strip() for part in title.split('/')]
    
    # Process each part to handle brackets
    expanded_parts = []
    
    for part in parts:
        part = part.strip()
        if '_' in part:  # For monoblock units like "EGSAH10D9W _1P"
            part = part.split('_')[0].strip()
        
        # Expand variants for this part
        part_variants = expand_variants_with_brackets(part)
        expanded_parts.extend(part_variants)
    
    return expanded_parts

def find_matches(hplib_df, scraped_df):
    """
    Find matches between database items and scraped data
    """
    matches = []
    
    for _, hplib_row in hplib_df.iterrows():
        title = str(hplib_row['Titel']) if pd.notna(hplib_row['Titel']) else ""
        
        if not title:
            continue
        
        # Clean the title
        clean_titel = clean_title(title)
        
        # Get original parts before any expansion
        original_parts = [part.strip() for part in clean_titel.split('/')]
        
        # Process the model code to get all variants
        model_codes = process_model_code(clean_titel)
        
        if not model_codes:
            continue
        
        # Normalize the model codes
        normalized_codes = [normalize_string(code) for code in model_codes]
        
        # Track all matches for this database item
        item_matches = []
        
        # Search for matches in scraped data
        for _, scraped_row in scraped_df.iterrows():
            scraped_model = str(scraped_row['Model/Type']) if pd.notna(scraped_row['Model/Type']) else ""
            
            if not scraped_model:
                continue
            
            # Normalize the scraped model
            normalized_model = normalize_string(scraped_model)
            
            # Split units need at least one match from each part
            if '/' in clean_titel:
                # Track which parts have been matched
                part_matched = [False] * len(original_parts)
                
                # For each code, check if it's in the scraped model and which part it belongs to
                for i, part in enumerate(original_parts):
                    part_variants = expand_variants_with_brackets(part.strip())
                    part_norm_variants = [normalize_string(v) for v in part_variants]
                    
                    for variant in part_norm_variants:
                        if variant in normalized_model:
                            part_matched[i] = True
                            break
                
                # Check if all parts have at least one match
                if all(part_matched):
                    item_matches.append({
                        'scraped_row': scraped_row,
                        'match_score': sum(part_matched),  # Score based on number of parts matched
                        'price': scraped_row['Price'] if pd.notna(scraped_row['Price']) else float('inf')
                    })
            else:
                # For monoblock units, just need the main code to be in the model
                for code in normalized_codes:
                    if code in normalized_model:
                        item_matches.append({
                            'scraped_row': scraped_row,
                            'match_score': 1,
                            'price': scraped_row['Price'] if pd.notna(scraped_row['Price']) else float('inf')
                        })
                        break
        
        # If we found matches for this database item
        if item_matches:
            # Sort by match score (higher is better) and then by price (lower is better)
            item_matches.sort(key=lambda x: (-x['match_score'], x['price']))
            
            # Take the best match
            best_match = item_matches[0]
            
            matches.append({
                'scraped_row': best_match['scraped_row'],
                'matching_hplib_rows': [hplib_row]
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

def find_combination_matches(unmatched_hplib_df, scraped_df):
    """
    Find matches for unmatched database items by combining individual scraped components
    like separate indoor and outdoor units
    """
    combination_matches = []
    
    # Extract scraped items that appear to be single components
    single_components = []
    for _, row in scraped_df.iterrows():
        model = str(row['Model/Type']) if pd.notna(row['Model/Type']) else ""
        if not model:
            continue
            
        # Add to single components list with normalized model
        normalized_model = normalize_string(model)
        single_components.append({
            'scraped_row': row,
            'normalized_model': normalized_model
        })
    
    # Only try combination matching for database items that are split systems
    for _, hplib_row in unmatched_hplib_df.iterrows():
        title = str(hplib_row['Titel']) if pd.notna(hplib_row['Titel']) else ""
        
        # Only process split systems (containing '/')
        if '/' not in title:
            continue
            
        # Get original parts before any expansion
        clean_titel = clean_title(title)
        original_parts = [part.strip() for part in clean_titel.split('/')]
        
        # We only handle systems with exactly two parts (outdoor/indoor)
        if len(original_parts) != 2:
            continue
        
        # Extract the outdoor unit (typically left of /) and indoor unit (right of /)
        outdoor_part = original_parts[0]
        indoor_part = original_parts[1]
        
        # Generate all possible variants for both parts
        outdoor_variants = [normalize_string(v) for v in expand_variants_with_brackets(outdoor_part)]
        indoor_variants = [normalize_string(v) for v in expand_variants_with_brackets(indoor_part)]
        
        # Find components that uniquely match each part
        outdoor_matches = []
        indoor_matches = []
        
        for component in single_components:
            model = component['normalized_model']
            
            # Check if the component is uniquely an outdoor unit
            is_outdoor = any(variant in model for variant in outdoor_variants)
            contains_indoor = any(variant in model for variant in indoor_variants)
            
            # Check if the component is uniquely an indoor unit
            is_indoor = any(variant in model for variant in indoor_variants)
            contains_outdoor = any(variant in model for variant in outdoor_variants)
            
            # Only include components that uniquely match one part but not both
            if is_outdoor and not contains_indoor:
                outdoor_matches.append(component)
                
            if is_indoor and not contains_outdoor:
                indoor_matches.append(component)
        
        # Only proceed if we have at least one match for each part
        if not outdoor_matches or not indoor_matches:
            continue
        
        # Find the best outdoor and indoor component pair
        best_pair = None
        best_match_score = 0
        
        for outdoor_comp in outdoor_matches:
            for indoor_comp in indoor_matches:
                # Skip if same component used twice
                if outdoor_comp['scraped_row'].equals(indoor_comp['scraped_row']):
                    continue
                
                # Calculate match score based on how well the components match the parts
                outdoor_score = sum(1 for v in outdoor_variants if v in outdoor_comp['normalized_model'])
                indoor_score = sum(1 for v in indoor_variants if v in indoor_comp['normalized_model'])
                total_score = outdoor_score + indoor_score
                
                # Additionally check if the match seems reliable
                reliable_match = False
                
                # Check if main product codes are present
                for outdoor_v in outdoor_variants:
                    if len(outdoor_v) >= 8 and outdoor_v in outdoor_comp['normalized_model']:
                        for indoor_v in indoor_variants:
                            if len(indoor_v) >= 8 and indoor_v in indoor_comp['normalized_model']:
                                reliable_match = True
                                break
                
                # Only consider reliable matches with a good score
                if reliable_match and (best_pair is None or total_score > best_match_score):
                    best_pair = (outdoor_comp, indoor_comp)
                    best_match_score = total_score
        
        # Only add the match if we found a reliable pair
        if best_pair:
            combination_matches.append({
                'scraped_rows': [best_pair[0]['scraped_row'], best_pair[1]['scraped_row']],
                'matching_hplib_rows': [hplib_row]
            })
    
    return combination_matches

def create_combination_output_dataframe(combination_matches):
    """Create output dataframe from combination matches."""
    if not combination_matches:
        return pd.DataFrame()
    
    # Create list of rows for the output dataframe
    rows = []
    for match in combination_matches:
        scraped_rows = match['scraped_rows']
        hplib_row = match['matching_hplib_rows'][0]
        
        # Create row with all database columns
        row = hplib_row.to_dict()
        
        # Join Model/Type fields with "&"
        combined_model = " & ".join([row['Model/Type'] for row in scraped_rows])
        
        # Calculate combined price
        combined_price = sum(float(row['Price']) if pd.notna(row['Price']) else 0 for row in scraped_rows)
        
        # Get currency with individual prices in brackets - format: "(price1 EUR & price2 EUR)"
        currency = scraped_rows[0]['Currency']
        prices_str = " & ".join([f"{float(row['Price']):.2f} EUR" for row in scraped_rows])
        currency_with_prices = f"{currency} ({prices_str})"
        
        # Combine dates with "&"
        combined_dates = " & ".join([str(row['Date_scraped']) for row in scraped_rows])
        
        # Get websites
        combined_websites = " & ".join([str(row['Website']) for row in scraped_rows])
        
        # Add scraped data columns
        row.update({
            'Model/Type': combined_model,
            'Price': combined_price,
            'Currency': currency_with_prices,
            'Website': combined_websites,
            'Date_scraped': combined_dates
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
    hplib_file = 'data/interim/filtered_hplib_GbM/DAIKIN_Europe_N.V..csv'
    scraped_file = 'data/interim/scraped_data_standardized_GbM/DAIKIN Europe NV.csv'
    output_file = 'data/processed/matched_manufacturers/daikin_matched.csv'
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Read the files
    hplib_df = pd.read_csv(hplib_file)
    scraped_df = pd.read_csv(scraped_file)
    
    print("=== Finding Matches ===")
    # Find matches
    matches = find_matches(hplib_df, scraped_df)
    
    # Create output dataframe
    result_df = create_output_dataframe(matches)
    
    # Find unmatched database items
    matched_hplib_titles = {match['matching_hplib_rows'][0]['Titel'] for match in matches}
    unmatched_hplib = hplib_df[~hplib_df['Titel'].isin(matched_hplib_titles)]
    
    print("\n=== Finding Combination Matches ===")
    # Try to match unmatched items by combining individual components
    combination_matches = find_combination_matches(unmatched_hplib, scraped_df)
    
    # Create combination output dataframe
    combination_df = create_combination_output_dataframe(combination_matches)
    
    # Combine both dataframes
    final_df = pd.concat([result_df, combination_df], ignore_index=True)
    
    # Save results
    final_df.to_csv(output_file, index=False)
    
    # Update matched items to include combination matches
    newly_matched_titles = {match['matching_hplib_rows'][0]['Titel'] for match in combination_matches}
    all_matched_titles = matched_hplib_titles.union(newly_matched_titles)
    truly_unmatched = hplib_df[~hplib_df['Titel'].isin(all_matched_titles)]
    
    # Print statistics
    print("\n=== Matching Statistics ===")
    print(f"Total items in database: {len(hplib_df)}")
    print(f"Regular matches found: {len(matches)}")
    print(f"Combination matches found: {len(combination_matches)}")
    print(f"Total matches found: {len(matches) + len(combination_matches)}")
    print(f"Unmatched database items: {len(truly_unmatched)}")
    
    # Print all unmatched database items
    print("\n=== All Unmatched Database Items ===")
    for _, row in truly_unmatched.iterrows():
        print(f"- {row['Titel']}")

if __name__ == "__main__":
    main() 