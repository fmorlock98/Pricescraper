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

def extract_parentheses_content(s):
    """
    Extract text inside parentheses from a string and split by semicolons.
    Example: "CHA-07/400V + CEW-2-200 (CHC-Monoblock 07/200 ; CHC-Monoblock 07/200-35)" 
    returns ["CHC-Monoblock 07/200", "CHC-Monoblock 07/200-35"]
    """
    if pd.isna(s):
        return []
    
    # Find text within parentheses
    matches = re.findall(r'\((.*?)\)', s)
    if not matches:
        return []
    
    # Join all matches and split by semicolons
    all_content = ' ; '.join(matches)
    # Split by semicolon and clean each part
    parts = [part.strip() for part in all_content.split(';')]
    return [part for part in parts if part]  # Remove empty parts

def strip_parentheses(s):
    """
    Remove text in parentheses from a string.
    Example: CHA-07/400V (123456) becomes CHA-07/400V
    """
    return re.sub(r'\(.*?\)', '', s).strip()

def find_matches(hplib_df, scraped_df):
    """Find matches between database and scraped data."""
    matches = []
    used_scraped_indices = set()  # Track which scraped items have been matched
    
    # Debug: Print all scraped models containing CHA
    print("Debug: Scraped models containing CHA:")
    for idx, row in scraped_df.iterrows():
        model = row['Model/Type']
        if 'CHA' in model:
            print(f"  {model}")
            
    # Debug: Print all scraped models containing FHA
    print("\nDebug: Scraped models containing FHA:")
    for idx, row in scraped_df.iterrows():
        model = row['Model/Type']
        if 'FHA' in model:
            print(f"  {model}")
    
    # Keywords indicating additional components (which may increase price)
    additional_component_keywords = [
        "Warmwasserspeicher", "Speicher", "Bedienmodul", "Zubehör", "Paket", 
        "Center", "Trennpufferspeicher", "Reihenpufferspeicher", "Pufferspeicher"
    ]
    
    for _, hplib_row in hplib_df.iterrows():
        hplib_title = str(hplib_row['Titel']) if pd.notna(hplib_row['Titel']) else ""
        
        # Skip if no title
        if not hplib_title:
            continue
            
        # Extract content in parentheses if any
        parentheses_content = extract_parentheses_content(hplib_title)
        
        # Process the database title (remove parentheses)
        clean_title = strip_parentheses(hplib_title)
        
        # Track all possible matches for this database item
        possible_matches = []
        
        # Case 1: Simple CHA items (e.g., "CHA-07/400V")
        if "CHA" in clean_title and not parentheses_content and "+" not in clean_title:
            print(f"Debug: Processing simple CHA item: {clean_title}")
            
            # Determine if it's CHA-07 or CHA-10
            is_cha07 = "07" in clean_title
            is_cha10 = "10" in clean_title
            
            # Search for matches in scraped data
            for idx, scraped_row in scraped_df.iterrows():
                if idx in used_scraped_indices:
                    continue
                    
                scraped_model = clean_string(scraped_row['Model/Type'])
                if not scraped_model:
                    continue
                    
                # Skip items containing "Schlammabscheider" for simple CHA items
                if "Schlammabscheider" in scraped_model:
                    continue
                
                # Simple matching for CHA-07 and CHA-10 models
                # Match needs to be specific for just the standalone CHA unit
                if is_cha07 and "CHA-Monoblock 07/400V" in scraped_model and "CHC-Monoblock" not in scraped_model:
                    print(f"Debug: Matched CHA-07 with {scraped_model}")
                    # Count additional components in the scraped model
                    component_count = sum(1 for keyword in additional_component_keywords if keyword.lower() in scraped_model.lower())
                    possible_matches.append((idx, scraped_row, component_count))
                elif is_cha10 and "CHA-Monoblock 10/400V" in scraped_model and "CHC-Monoblock" not in scraped_model:
                    print(f"Debug: Matched CHA-10 with {scraped_model}")
                    # Count additional components in the scraped model
                    component_count = sum(1 for keyword in additional_component_keywords if keyword.lower() in scraped_model.lower())
                    possible_matches.append((idx, scraped_row, component_count))
        
        # Case 2: CHA items with CEW-2-200 or SEW-2-300 and parentheses
        elif "CHA" in clean_title and ("CEW-2-200" in clean_title or "SEW-2-300" in clean_title) and parentheses_content:
            print(f"Debug: Processing CHA+CEW/SEW item: {clean_title}")
            
            # Determine if this is a SEW-2-300 item (special case)
            is_sew_item = "SEW-2-300" in clean_title
            
            # Extract the base part and power variant from the title
            # e.g., "CHA-07/400V" from "CHA-07/400V + SEW-2-300 (...)"
            base_match = re.search(r'(CHA-\d+/\d+V)', clean_title)
            power_variant = None
            if base_match:
                base_part = base_match.group(1)
                power_variant = base_part.split('-')[1].split('/')[0]  # e.g., "07" from "CHA-07/400V"
            
            # We look at the content in parentheses
            for content in parentheses_content:
                # Split by semicolon
                parts = [part.strip() for part in content.split(';')]
                
                # Search for matches in scraped data
                for idx, scraped_row in scraped_df.iterrows():
                    if idx in used_scraped_indices:
                        continue
                        
                    scraped_model = clean_string(scraped_row['Model/Type'])
                    if not scraped_model:
                        continue
                    
                    # For SEW-2-300 items, we allow Schlammabscheider and prioritize CHC items
                    if is_sew_item:
                        # Try to match any of the bracket models
                        for part in parts:
                            if part in scraped_model:
                                print(f"Debug: Matched SEW-2-300 item with {scraped_model}")
                                # Count additional components in the scraped model
                                component_count = sum(1 for keyword in additional_component_keywords if keyword.lower() in scraped_model.lower())
                                possible_matches.append((idx, scraped_row, component_count))
                                break
                    else:
                        # Skip items containing "Schlammabscheider" for CEW-2-200 items
                        if "Schlammabscheider" in scraped_model:
                            continue
                            
                        # Try first part before semicolon
                        if parts[0] in scraped_model:
                            # Count additional components in the scraped model
                            component_count = sum(1 for keyword in additional_component_keywords if keyword.lower() in scraped_model.lower())
                            possible_matches.append((idx, scraped_row, component_count))
                            break
                        # If no match, try second part after semicolon
                        if len(parts) > 1 and parts[1] in scraped_model:
                            # Count additional components in the scraped model
                            component_count = sum(1 for keyword in additional_component_keywords if keyword.lower() in scraped_model.lower())
                            possible_matches.append((idx, scraped_row, component_count))
                            break
        
        # Case 3: Simple FHA items (e.g., "FHA-08/10-230V-M2 FS-B2")
        elif "FHA" in clean_title and not parentheses_content and "+" not in clean_title:
            print(f"Debug: Processing simple FHA item: {clean_title}")
            
            # Extract key identifiers from the FHA model
            # Example: "FHA-05/06-230V-M2 FS-B2" → power="05/06", voltage="230V"
            power_match = re.search(r'FHA[- ](\d+/\d+)', clean_title)
            voltage_match = re.search(r'(\d+V)', clean_title)
            
            # Check if this model has an e6 designation (electric heating element)
            has_e6 = "-e6-" in clean_title or " e6 " in clean_title
            
            if power_match:
                power = power_match.group(1)  # e.g., "05/06"
                voltage = voltage_match.group(1) if voltage_match else ""
                
                print(f"Debug: FHA power={power}, voltage={voltage}, has_e6={has_e6}")
                
                # Special case for FHA-11/14 and FHA-14/17
                is_special_fha = power in ["11/14", "14/17"]
                
                # Search for matches in scraped data
                for idx, scraped_row in scraped_df.iterrows():
                    if idx in used_scraped_indices:
                        continue
                        
                    scraped_model = clean_string(scraped_row['Model/Type'])
                    if not scraped_model:
                        continue
                    
                    # Special handling for FHA-11/14 and FHA-14/17 models
                    if is_special_fha:
                        # For these specific models, we don't exclude "Schlammabscheider"
                        # We just need to match the power rating and FHA
                        if f"FHA-{power}" in scraped_model or f"FHA {power}" in scraped_model:
                            print(f"Debug: Matched special FHA with {scraped_model}")
                            # Count additional components in the scraped model
                            component_count = sum(1 for keyword in additional_component_keywords if keyword.lower() in scraped_model.lower())
                            possible_matches.append((idx, scraped_row, component_count))
                    else:
                        # Skip items containing "Schlammabscheider" for regular FHA models
                        if "Schlammabscheider" in scraped_model:
                            continue
                            
                        # Skip FHA-Center items - they should only match with database items containing parentheses
                        if "FHA-Center" in scraped_model or "FHA-Center" in scraped_model.lower():
                            continue
                        
                        # Check if the scraped model has an electric heating element
                        scraped_has_e6 = "E-Heizelement" in scraped_model or "e6" in scraped_model.lower()
                        
                        # Match if it's an FHA model with the same power and voltage,
                        # and matching electric heating element status
                        if "FHA" in scraped_model and power in scraped_model:
                            # Check voltage if it was found
                            if not voltage or voltage in scraped_model:
                                # Both the database and scraped model should match regarding heating element
                                if has_e6 == scraped_has_e6:
                                    print(f"Debug: Matched FHA with {scraped_model}")
                                    # Count additional components in the scraped model
                                    component_count = sum(1 for keyword in additional_component_keywords if keyword.lower() in scraped_model.lower())
                                    possible_matches.append((idx, scraped_row, component_count))
        
        # Case 4: FHA items with CEW-2-200 and parentheses
        elif "FHA" in clean_title and ("CEW-2-200" in clean_title or "SEW" in clean_title) and parentheses_content:
            print(f"Debug: Processing FHA+CEW item: {clean_title}")
            
            # Check if this model has an e6 designation (electric heating element)
            has_e6 = "-e6-" in clean_title or " e6 " in clean_title
            
            # Extract power rating from the title
            power_match = re.search(r'FHA[- ](\d+/\d+)', clean_title)
            power = power_match.group(1) if power_match else None
            
            # We only care about the content in parentheses
            for content in parentheses_content:
                # Split by semicolon
                parts = [part.strip() for part in content.split(';')]
                
                # Search for matches in scraped data
                for idx, scraped_row in scraped_df.iterrows():
                    if idx in used_scraped_indices:
                        continue
                        
                    scraped_model = clean_string(scraped_row['Model/Type'])
                    if not scraped_model:
                        continue
                        
                    # Skip items containing "Schlammabscheider"
                    if "Schlammabscheider" in scraped_model:
                        continue
                    
                    # Check if the scraped model has an electric heating element
                    scraped_has_e6 = "E-Heizelement" in scraped_model or "e6" in scraped_model.lower()
                    
                    # Try first part before semicolon
                    if parts[0] in scraped_model:
                        # Both the database and scraped model should match regarding heating element
                        if has_e6 == scraped_has_e6:
                            print(f"Debug: Matched FHA+CEW first part with {scraped_model}")
                            # Count additional components in the scraped model
                            component_count = sum(1 for keyword in additional_component_keywords if keyword.lower() in scraped_model.lower())
                            possible_matches.append((idx, scraped_row, component_count))
                            break
                    # If no match, try second part after semicolon
                    if len(parts) > 1 and parts[1] in scraped_model:
                        # Both the database and scraped model should match regarding heating element
                        if has_e6 == scraped_has_e6:
                            print(f"Debug: Matched FHA+CEW second part with {scraped_model}")
                            # Count additional components in the scraped model
                            component_count = sum(1 for keyword in additional_component_keywords if keyword.lower() in scraped_model.lower())
                            possible_matches.append((idx, scraped_row, component_count))
                            break
                    
                    # Special case for FHA-Center items - these should match with items that have parentheses
                    if ("FHA-Center" in scraped_model or "FHA Center" in scraped_model) and power:
                        if power in scraped_model:
                            print(f"Debug: Matched FHA-Center with {scraped_model}")
                            # Count additional components in the scraped model
                            component_count = sum(1 for keyword in additional_component_keywords if keyword.lower() in scraped_model.lower())
                            possible_matches.append((idx, scraped_row, component_count))
                            break
        
        # If we found matches, choose the best one based on component count and price
        if possible_matches:
            print(f"Debug: Found {len(possible_matches)} matches for {clean_title}")
            
            # Filter out rows with NaN prices
            valid_price_matches = [(idx, row, component_count) for idx, row, component_count in possible_matches if pd.notna(row['Price'])]
            
            # If no valid prices, just take the first match
            if not valid_price_matches:
                best_idx, best_match, _ = possible_matches[0]
            else:
                # Sort matches - first by component count (ascending), then by price (ascending)
                # This prioritizes simpler models with fewer components
                sorted_matches = sorted(valid_price_matches, key=lambda x: (x[2], float(x[1]['Price'])))
                
                # Print the sorted matches for debugging
                print("Debug: Sorted matches (component_count, price):")
                for idx, row, comp_count in sorted_matches:
                    print(f"  {comp_count} components, {row['Price']} {row['Currency']}: {row['Model/Type']}")
                
                # Take the first (simplest/cheapest) match
                best_idx, best_match, _ = sorted_matches[0]
            
            # Mark this scraped item as used
            used_scraped_indices.add(best_idx)
            
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
    hplib_file = 'data/interim/filtered_hplib_GbM/WOLF_GmbH.csv'
    scraped_file = 'data/interim/scraped_data_standardized_GbM/WOLF GmbH.csv'
    output_file = 'data/processed/matched_manufacturers/wolf_matched.csv'
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Read the files
    hplib_df = pd.read_csv(hplib_file)
    scraped_df = pd.read_csv(scraped_file)
    
    print("=== Finding Matches ===")
    # Find matches using more flexible matching criteria
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