import pandas as pd
import os
from datetime import datetime
import re

def extract_model_series(s):
    """Extract the base model series from a string"""
    if pd.isna(s):
        return ""
    # Convert to lowercase and remove special characters
    s = str(s).lower()
    
    # Try to match CS3400iAWS patterns
    patterns = [
        r'cs3400iaws\s*\d+\s*(?:e|m|b|or[-\s]?[s|t]|ore[-\s]?[s|t]|orm[-\s]?[s|t]|ormb[-\s]?[s|t])',
        r'cs3400iaws\s*\d+\s*(?:e|m|b|or[-\s]?[s|t]|ore[-\s]?[s|t]|orm[-\s]?[s|t]|ormb[-\s]?[s|t])',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, s)
        if match:
            return match.group(0)
            
    # Try to match CS5800iAW patterns
    patterns = [
        r'cs5800iaw\s*\d+\s*(?:e|m|b|or[-\s]?[s|t]|ore[-\s]?[s|t]|orm[-\s]?[s|t]|ormb[-\s]?[s|t])',
        r'cs5800iaw\s*\d+\s*(?:e|m|b|or[-\s]?[s|t]|ore[-\s]?[s|t]|orm[-\s]?[s|t]|ormb[-\s]?[s|t])',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, s)
        if match:
            return match.group(0)
            
    return ""

def extract_power(s):
    """Extract power rating from a string"""
    if pd.isna(s):
        return None
    # Look for patterns like "6,9 kW" or "6.9 kW"
    match = re.search(r'(\d+[.,]\d+)\s*kw', str(s).lower())
    if match:
        return float(match.group(1).replace(',', '.'))
    return None

def extract_indoor_type(s):
    """Extract indoor unit type (E, M, B) from a string"""
    if pd.isna(s):
        return None
    s = str(s).lower()
    
    # Look for specific indoor unit patterns
    if 'inneneinheit' in s or 'mb' in s or 'm' in s or 'e' in s:
        if 'monoenerg' in s or 'e' in s:
            return 'E'
        elif 'speicher' in s or 'm' in s:
            return 'M'
        elif 'bivalent' in s or 'b' in s or 'mb' in s:
            return 'B'
            
    # Look for patterns in package strings
    if 'cs5800iaw' in s:
        if 'e' in s:
            return 'E'
        elif 'm' in s and 'b' not in s:
            return 'M'
        elif 'mb' in s:
            return 'B'
            
    return None

def extract_outdoor_type(s):
    """Extract outdoor unit type (OR-S, OR-T, etc.) from a string"""
    if pd.isna(s):
        return None
    s = str(s).lower()
    
    # Look for specific outdoor unit patterns
    patterns = [
        r'or[-\s]?s',  # OR-S
        r'or[-\s]?t',  # OR-T
        r'ore[-\s]?s',  # ORE-S
        r'ore[-\s]?t',  # ORE-T
        r'orm[-\s]?s',  # ORM-S
        r'orm[-\s]?t',  # ORM-T
        r'ormb[-\s]?s',  # ORMB-S
        r'ormb[-\s]?t'   # ORMB-T
    ]
    
    for pattern in patterns:
        if re.search(pattern, s):
            return re.search(pattern, s).group(0).upper()
            
    # Look for patterns in package strings
    if 'cs5800iaw' in s:
        if 'or-s' in s:
            return 'OR-S'
        elif 'or-t' in s:
            return 'OR-T'
            
    return None

def extract_size(s):
    """Extract size number from model (e.g., 10 from CS3400iAWS 10)"""
    if pd.isna(s):
        return None
    # Try CS3400iAWS pattern
    match = re.search(r'cs3400iaws\s*(\d+)', str(s).lower())
    if match:
        return int(match.group(1))
        
    # Try CS5800iAW pattern
    match = re.search(r'cs5800iaw\s*(\d+)', str(s).lower())
    if match:
        return int(match.group(1))
        
    # Try to extract size from BOPA model number
    match = re.search(r'bopa\s*cs\d+\s*aw\s*(\d+)', str(s).lower())
    if match:
        return int(match.group(1))
    
    # Try to extract size from OR pattern
    match = re.search(r'(\d+)\s*or[-\s]?[s|t]', str(s).lower())
    if match:
        return int(match.group(1))
        
    return None

def parse_package_string(s):
    """Parse a package string to extract components for CS5800iAW/CS6800iAW models"""
    if pd.isna(s):
        return None, None, None
    
    s = str(s).lower()
    parts = re.split(r'[,;+]', s)
    
    size = None
    indoor_type = None
    outdoor_type = None
    base_model = None
    
    # Determine which model series is in the string
    if 'cs6800iaw' in s:
        base_model = 'CS6800iAW'
    elif 'cs5800iaw' in s:
        base_model = 'CS5800iAW'
    
    # Step 1: Extract size from any pattern containing AW X OR-S/T
    # This is the most reliable source for size
    match = re.search(r'aw\s*(\d+)\s*or[-\s]?[s|t]', s)
    if match:
        size = int(match.group(1))
    else:
        # Fallback to other size extraction methods
        for part in parts:
            # Try BOPA CS### AW X format
            match = re.search(r'bopa\s*cs\d+\s*aw\s*(\d+)', part)
            if match:
                size = int(match.group(1))
                break
                
            # Try BOPA7## AW X format 
            match = re.search(r'bopa\s*\d+\s*(?:wärmepumpe\s*)?aw\s*(\d+)', part)
            if match:
                size = int(match.group(1))
                break
    
    # Step 2: Extract outdoor type from OR-S/OR-T
    if 'or-s' in s:
        outdoor_type = 'S'
    elif 'or-t' in s:
        outdoor_type = 'T'
    
    # Step 3: Find the indoor type (E, M, or MB)
    for part in parts:
        if 'cs5800iaw' in part or 'cs6800iaw' in part:
            # Check for MB first (to prevent matching M before MB)
            if ' mb' in part or part.endswith(' mb') or ',mb' in part or part.endswith(',mb'):
                indoor_type = 'B'  # MB corresponds to ORMB
                break
            elif ' m ' in part or part.endswith(' m') or ',m' in part or part.endswith(',m') or 'hydrobox/speicher' in part:
                indoor_type = 'M'  # M corresponds to ORM
                break
            elif ' e ' in part or part.endswith(' e') or ',e' in part or part.endswith(',e') or 'monovalent' in part:
                indoor_type = 'E'  # E corresponds to ORE
                break
    
    # If we have the necessary components and don't have a base model yet, determine it from the rest
    if size and indoor_type and outdoor_type and not base_model:
        base_model = 'CS5800iAW'  # Default to CS5800iAW if not explicitly found
    
    return size, indoor_type, outdoor_type, base_model

def find_matches(hplib_df, scraped_df):
    """Find matches between hplib and scraped data"""
    matches = []
    used_hplib_indices = set()
    
    # Create an explicit mapping for specific package strings that don't match correctly
    package_to_model = {
        # Original mappings
        "BOSCH Luftwärmepumpen-Paket BOPA CS761 AW 4 OR-S, CS5800iAW 12 M, 7739622385": "CS5800iAW 4 ORM-S",
        "BOSCH Luftwärmepumpen-Paket BOPA CS763 AW 7 OR-S, CS5800iAW 12 M, 7739622388": "CS5800iAW 7 ORM-S",
        "BOSCH Luftwärmepumpen-Paket BOPA CS806 AW 12 OR-T, CS5800iAW 12 MB, WH 290, 7739624195": "CS5800iAW 12 ORMB-T",
        "BOSCH Luftwärmepumpen-Paket BOPA CS776 AW 4 OR-S, CS5800iAW 12 MB, SWDP300-2O C, 7739622865": "CS5800iAW 4 ORMB-S",
        "BOSCH Luftwärmepumpen-Paket BOPA CS762 AW 5 OR-S, CS5800iAW 12 M, 7739622387": "CS5800iAW 5 ORM-S",
        "BOSCH Luftwärmepumpen-Paket BOPA CS774 AW 5 OR-S, CS5800iAW 12 MB, HR300, 7739622864": "CS5800iAW 5 ORMB-S",
        "BOSCH Luftwärmepumpen-Paket BOPA CS789 AW 12 OR-T, CS5800iAW 12 M, 7739623436": "CS5800iAW 12 ORM-T",
        
        # Additional mappings for models with (60°C)
        "BOSCH Luftwärmepumpen-Paket BOPA CS767 AW 4 OR-S, CS5800iAW 12 E, BST 50,HR300, 7739622384": "CS5800iAW 4 ORE-S (60°C)",
        "BOSCH Luftwärmepumpen-Paket BOPA CS768 AW 5 OR-S, CS5800iAW 12 E, BST 50,HR300, 7739622386": "CS5800iAW 5 ORE-S (60°C)",
        "BOSCH Luftwärmepumpen-Paket BOPA CS769 AW 7 OR-S, CS5800iAW 12 E, BST 50,HR300, 7739622390": "CS5800iAW 7 ORE-S (60°C)",
        "BOSCH Luftwärmepumpen-Paket BOPA CS773 AW 4 OR-S, CS5800iAW 12 MB, HR300, 7739622863": "CS5800iAW 4 ORMB-S (60°C)",
        "BOSCH Luftwärmepumpen-Paket BOPA CS774 AW 5 OR-S, CS5800iAW 12 MB, HR300, 7739622864": "CS5800iAW 5 ORMB-S (60°C)",
        "BOSCH Luftwärmepumpen-Paket BOPA CS775 AW 7 OR-S, CS5800iAW 12 MB, HR300, 7739622867": "CS5800iAW 7 ORMB-S (60°C)",
        "BOSCH Luftwärmepumpen-Paket BOPA CS788 AW 10 OR-T, CS5800iAW 12 M, 7739623429": "CS5800iAW 10 ORM-T (60°C)",
        "BOSCH Luftwärmepumpen-Paket BOPA CS792 AW 10 OR-T, CS5800iAW 12 E, BH120,WH290, 7739623435": "CS5800iAW 10 ORE-T (60°C)",
        "BOSCH Luftwärmepumpen-Paket BOPA CS793 AW 12 OR-T, CS5800iAW 12 E, BH120,WH290, 7739623442": "CS5800iAW 12 ORE-T (60°C)",
        "BOSCH Luftwärmepumpen-Paket BOPA CS763 AW 7 OR-S, CS5800iAW 12 M, 7739622388": "CS5800iAW 7 ORM-S (60°C)",
        
        # More variants with different packages
        "Bosch CS766 AW 7 OR-S / CS6800iAW 12 M (7739622393)": "CS5800iAW 7 ORM-S",
        "Bosch Set BOPA CS772 AW 7 OR-S CS6800iAW 12 E + BH120/WH290 (7739622395)": "CS6800iAW 7 ORE-S",
        
        # Also handle the match for different idealo format
        "Bosch Set BOPA CS772 AW 7 OR-S CS6800iAW 12 E + BH120/WH290 (7739622395)": "CS6800iAW 7 ORE-S"
    }
    
    # Create bidirectional mapping for CS5800iAW models with/without (60°C)
    stripped_to_original = {}
    for model in hplib_df['Titel']:
        if 'CS5800iAW' in model and '(60°C)' in model:
            stripped = model.replace(' (60°C)', '')
            stripped_to_original[stripped.lower()] = model
            stripped_to_original[model.lower()] = model
    
    # First handle the explicit mappings
    for scraped_idx, scraped_row in scraped_df.iterrows():
        model_type = scraped_row['Model/Type']
        if pd.isna(model_type):
            continue
            
        # Check if this is a known package that needs special handling
        if model_type in package_to_model:
            expected_model = package_to_model[model_type]
            
            # Find matching hplib entry
            for hplib_idx, hplib_row in hplib_df.iterrows():
                if hplib_idx in used_hplib_indices:
                    continue
                
                hplib_title = hplib_row['Titel'].lower()
                # Remove (60°C) from hplib title
                hplib_title_stripped = hplib_title.replace(' (60°c)', '')
                
                # Check if this is the right model - either exact match or with/without (60°C)
                if (expected_model.lower() == hplib_title or 
                    expected_model.lower() == hplib_title_stripped or
                    expected_model.lower().replace(' (60°c)', '') == hplib_title_stripped):
                    matches.append({
                        'scraped_row': scraped_row,
                        'matching_hplib_rows': [hplib_row],
                        'is_combined': False
                    })
                    used_hplib_indices.add(hplib_idx)
                    break
    
    # First, identify indoor units for different categories
    indoor_units = {}
    cs3400_indoor_units = []  # Special list for CS3400iAWS indoor units
    
    for idx, row in scraped_df.iterrows():
        model_type = row['Model/Type']
        if pd.isna(model_type):
            continue
            
        # Track CS3400iAWS indoor units separately
        if 'cs3400iaws' in model_type.lower() and 'inneneinheit' in model_type.lower():
            cs3400_indoor_units.append(row)
            continue
            
        # Regular indoor units for other models
        indoor_type = extract_indoor_type(model_type)
        if indoor_type:
            size = extract_size(model_type)
            if size:
                key = (size, indoor_type)
                if key not in indoor_units or row['Price'] < indoor_units[key]['Price']:
                    indoor_units[key] = row
    
    # First try direct matches
    for scraped_idx, scraped_row in scraped_df.iterrows():
        model_type = scraped_row['Model/Type']
        if pd.isna(model_type) or model_type in package_to_model:  # Skip if already handled
            continue
            
        # Try to find direct match in hplib
        for hplib_idx, hplib_row in hplib_df.iterrows():
            if hplib_idx in used_hplib_indices:
                continue
                
            if hplib_row['Titel'].lower() in model_type.lower():
                matches.append({
                    'scraped_row': scraped_row,
                    'matching_hplib_rows': [hplib_row],
                    'is_combined': False
                })
                used_hplib_indices.add(hplib_idx)
                break
    
    # Handle CS3400iAWS models - special logic to combine any outdoor unit with any compatible indoor unit
    for scraped_idx, scraped_row in scraped_df.iterrows():
        model_type = scraped_row['Model/Type']
        if pd.isna(model_type) or 'cs3400iaws' not in model_type.lower() or 'inneneinheit' in model_type.lower() or model_type in package_to_model:  # Skip if already handled
            continue
            
        # This is an outdoor unit - extract its size
        size = extract_size(model_type)
        if not size:
            continue
            
        # Find compatible indoor units (CS3400iAWS B units - we don't need to match sizes)
        for indoor_row in cs3400_indoor_units:
            indoor_type = extract_indoor_type(indoor_row['Model/Type'])
            if not indoor_type:
                continue
                
            # Create combined model type
            combined_model = f"{model_type} & {indoor_row['Model/Type']}"
            expected_model = f"CS3400iAWS {size} OR{indoor_type}-S"
            
            # Find matching hplib entry
            for hplib_idx, hplib_row in hplib_df.iterrows():
                if hplib_idx in used_hplib_indices:
                    continue
                    
                # Check if this is the right model
                if expected_model.lower() in hplib_row['Titel'].lower() or hplib_row['Titel'].lower() == f"cs3400iaws {size} or{indoor_type.lower()}-s":
                    # Create combined row
                    combined_row = scraped_row.copy()
                    combined_row['Model/Type'] = combined_model
                    combined_row['Price'] = scraped_row['Price'] + indoor_row['Price']
                    combined_row['Website'] = f"{scraped_row['Website']} ({scraped_row['Price']} {scraped_row['Currency']}) & {indoor_row['Website']} ({indoor_row['Price']} {indoor_row['Currency']})"
                    combined_row['Date_scraped'] = f"{scraped_row['Date_scraped']} & {indoor_row['Date_scraped']}"
                    
                    matches.append({
                        'scraped_row': combined_row,
                        'matching_hplib_rows': [hplib_row],
                        'is_combined': True
                    })
                    used_hplib_indices.add(hplib_idx)
                    break
    
    # Handle CS5800iAW/CS6800iAW models in package strings
    for scraped_idx, scraped_row in scraped_df.iterrows():
        model_type = scraped_row['Model/Type']
        if pd.isna(model_type) or model_type in package_to_model or ('cs5800iaw' not in model_type.lower() and 'cs6800iaw' not in model_type.lower() and 'bopa' not in model_type.lower()):  # Skip if already handled
            continue
        
        # Parse the package string to extract components
        size, indoor_type, outdoor_type, base_model = parse_package_string(model_type)
        
        if size and indoor_type and outdoor_type:
            # Determine the exact model from the package string
            if 'cs6800iaw' in model_type.lower():
                base_model = 'CS6800iAW'
            elif 'cs5800iaw' in model_type.lower():
                base_model = 'CS5800iAW'
            
            # Only proceed if we have a specific model type
            if not base_model:
                continue
                
            # Construct the base model name without (60°C)
            base_model_name = f"{base_model} {size} OR{indoor_type}-{outdoor_type}"
            
            # Find matching hplib entry
            found_match = False
            for hplib_idx, hplib_row in hplib_df.iterrows():
                if hplib_idx in used_hplib_indices:
                    continue
                
                hplib_title = hplib_row['Titel'].lower()
                
                # For CS5800iAW models, try matching without (60°C)
                if base_model == 'CS5800iAW':
                    # Try matching with the base model name (without 60°C)
                    if base_model_name.lower() == hplib_title.replace(' (60°c)', ''):
                        matches.append({
                            'scraped_row': scraped_row,
                            'matching_hplib_rows': [hplib_row],
                            'is_combined': False
                        })
                        used_hplib_indices.add(hplib_idx)
                        found_match = True
                        break
                    
                    # If no match found, try matching with the exact model name
                    if base_model_name.lower() == hplib_title:
                        matches.append({
                            'scraped_row': scraped_row,
                            'matching_hplib_rows': [hplib_row],
                            'is_combined': False
                        })
                        used_hplib_indices.add(hplib_idx)
                        found_match = True
                        break
                else:
                    # For other models, try both with and without (60°C)
                    if base_model_name.lower() == hplib_title or f"{base_model_name} (60°c)".lower() == hplib_title:
                        matches.append({
                            'scraped_row': scraped_row,
                            'matching_hplib_rows': [hplib_row],
                            'is_combined': False
                        })
                        used_hplib_indices.add(hplib_idx)
                        found_match = True
                        break
    
    return matches

def create_output_dataframe(matches):
    """Create the output dataframe from matches"""
    output_rows = []
    
    # Define the column order
    columns = [
        'Manufacturer', 'Model', 'Titel', 'Date', 'Model/Type', 'Price', 'Currency', 
        'Website', 'Date_scraped', 'Type', 'Subtype', 'Group', 'Rated Power low T [kW]', 
        'Rated Power medium T [kW]', 'Refrigerant', 'Mass of Refrigerant [kg]', 
        'SPL indoor low Power [dBA]', 'SPL outdoor low Power [dBA]', 'SPL indoor high Power [dBA]', 
        'SPL outdoor high Power [dBA]', 'Bivalence temperature [°C]', 'Tolerance temperature [°C]', 
        'Max. water heating temperature [°C]', 'Power heating rod low T [kW]', 
        'Power heating rod medium T [kW]', 'Poff [W]', 'PTOS [W]', 'PSB [W]', 'PCKS [W]', 
        'eta low T [%]', 'eta medium T [%]', 'SCOP', 'SEER low T', 'SEER medium T', 
        'P_th_h_ref [W]', 'P_th_c_ref [W]', 'P_el_h_ref [W]', 'P_el_c_ref [W]', 'COP_ref', 
        'EER_c_ref'
    ]
    
    for match in matches:
        scraped_row = match['scraped_row']
        for hplib_row in match['matching_hplib_rows']:
            output_row = {}
            
            # First add all hplib columns
            for col in hplib_row.index:
                output_row[col] = hplib_row[col]
            
            # Then update with scraped data
            output_row['Model/Type'] = scraped_row['Model/Type']
            if pd.notna(scraped_row['Price']):
                output_row['Price'] = scraped_row['Price']
            if pd.notna(scraped_row['Currency']):
                output_row['Currency'] = scraped_row['Currency']
            if pd.notna(scraped_row['Website']):
                output_row['Website'] = scraped_row['Website']
            if pd.notna(scraped_row['Date_scraped']):
                output_row['Date_scraped'] = scraped_row['Date_scraped']
            
            output_rows.append(output_row)
    
    # Create DataFrame and reorder columns
    df = pd.DataFrame(output_rows)
    df = df[columns]
    return df

def create_manual_matches(hplib_df, scraped_df):
    """Create manual matches for specific combinations"""
    manual_matches = []
    
    # Define the combinations for CS3400iAWS models
    combinations = [
        {
            'hplib_title': 'CS3400iAWS 10 ORB-S',
            'outdoor': 'Junkers Bosch Split Luft/Wasser-Wärmepumpe Compress CS3400iAWS 10 OR-S, 6,9 kW (8750722683)',
            'indoor': 'Junkers Bosch CS3400iAWS 10 B Luft/Wasser-Wärmepumpe Split Inneneinheit, wandh. (8738212151)'
        },
        {
            'hplib_title': 'CS3400iAWS 6 ORB-S',
            'outdoor': 'Junkers Bosch Split Luft/Wasser-Wärmepumpe Compress CS3400iAWS 6 OR-S, 5,1 kW (8750722681)',
            'indoor': 'Junkers Bosch CS3400iAWS 10 B Luft/Wasser-Wärmepumpe Split Inneneinheit, wandh. (8738212151)'
        },
        {
            'hplib_title': 'CS3400iAWS 8 ORB-S',
            'outdoor': 'Junkers Bosch Split Luft/Wasser-Wärmepumpe Compress CS3400iAWS 8 OR-S, 6,2 kW (8750722682)',
            'indoor': 'Junkers Bosch CS3400iAWS 10 B Luft/Wasser-Wärmepumpe Split Inneneinheit, wandh. (8738212151)'
        },
        {
            'hplib_title': 'CS3400iAWS 4 ORB-S',
            'outdoor': 'Junkers Bosch Split Luft/Wasser-Wärmepumpe Compress CS3400iAWS 4 OR-S, 4,3 kW (8750722680)',
            'indoor': 'Junkers Bosch CS3400iAWS 10 B Luft/Wasser-Wärmepumpe Split Inneneinheit, wandh. (8738212151)'
        }
    ]
    
    # Define the direct matches for specific CS5800iAW/CS6800iAW models
    cs5800_direct_matches = [
        {
            'hplib_title': 'CS5800iAW 10 ORMB-T',
            'scraped': 'BOSCH Luftwärmepumpen-Paket BOPA CS807 AW 10 OR-T, CS5800iAW 12 MB, HR300, 7739624189'
        },
        {
            'hplib_title': 'CS5800iAW 10 ORMB-T (60°C)',
            'scraped': 'BOSCH Luftwärmepumpen-Paket BOPA CS807 AW 10 OR-T, CS5800iAW 12 MB, HR300, 7739624189'
        },
        {
            'hplib_title': 'CS5800iAW 12 ORMB-T',
            'scraped': 'BOSCH Luftwärmepumpen-Paket BOPA CS806 AW 12 OR-T, CS5800iAW 12 MB, WH 290, 7739624195'
        },
        {
            'hplib_title': 'CS5800iAW 12 ORMB-T (60°C)',
            'scraped': 'BOSCH Luftwärmepumpen-Paket BOPA CS806 AW 12 OR-T, CS5800iAW 12 MB, WH 290, 7739624195'
        },
        {
            'hplib_title': 'CS6800iAW 10 ORE-T',
            'scraped': 'BOSCH Luftwärmepumpen-Paket BOPA CS794 AW 10 OR-T, CS6800iAW 12 E, BH200,WH370, 7739623431'
        },
        {
            'hplib_title': 'CS6800iAW 10 ORM-T',
            'scraped': 'BOSCH Luftwärmepumpen-Paket BOPA CS790 AW 10 OR-T, CS6800iAW 12 M, 7739623430'
        },
        {
            'hplib_title': 'CS6800iAW 10 ORMB-T',
            'scraped': 'BOSCH Luftwärmepumpen-Paket BOPA CS800 AW 10 OR-T, CS6800iAW 12 MB, SWDP 300-2, 7739623434'
        },
        {
            'hplib_title': 'CS6800iAW 12 ORE-T',
            'scraped': 'BOSCH Luftwärmepumpen-Paket BOPA CS795 AW 12 OR-T, CS6800iAW 12 E, BH200,WH370, 7739623438'
        },
        {
            'hplib_title': 'CS6800iAW 12 ORM-T',
            'scraped': 'BOSCH Luftwärmepumpen-Paket BOPA CS791 AW 12 OR-T, CS6800iAW 12 M, 7739623437'
        },
        {
            'hplib_title': 'CS6800iAW 12 ORMB-T',
            'scraped': 'BOSCH Luftwärmepumpen-Paket BOPA CS797 AW 12 OR-T, CS6800iAW 12 MB, WH 370, 7739623440'
        },
        {
            'hplib_title': 'CS5800iAW 4 ORMB-S (60°C)',
            'scraped': 'BOSCH Luftwärmepumpen-Paket BOPA CS773 AW 4 OR-S, CS5800iAW 12 MB, HR300, 7739622863'
        },
        {
            'hplib_title': 'CS5800iAW 5 ORMB-S (60°C)',
            'scraped': 'BOSCH Luftwärmepumpen-Paket BOPA CS774 AW 5 OR-S, CS5800iAW 12 MB, HR300, 7739622864'
        },
        {
            'hplib_title': 'CS5800iAW 7 ORMB-S (60°C)',
            'scraped': 'BOSCH Luftwärmepumpen-Paket BOPA CS775 AW 7 OR-S, CS5800iAW 12 MB, HR300, 7739622867'
        },
        {
            'hplib_title': 'CS6800iAW 4 ORMB-S',
            'scraped': 'BOSCH Luftwärmepumpen-Paket BOPA CS782 AW 4 OR-S, CS6800iAW 12 MB, SWDP300-2O C, 7739622870'
        },
        {
            'hplib_title': 'CS6800iAW 5 ORMB-S',
            'scraped': 'BOSCH Luftwärmepumpen-Paket BOPA CS783 AW 5 OR-S, CS6800iAW 12 MB, SWDP 300-2, 7739622873'
        },
        {
            'hplib_title': 'CS6800iAW 7 ORMB-S',
            'scraped': 'BOSCH Luftwärmepumpen-Paket BOPA CS781 AW 7 OR-S, CS6800iAW 12 MB, WH 290, 7739622872'
        },
        # Add the remaining (60°C) variants with appropriate match lines
        {
            'hplib_title': 'CS5800iAW 10 ORE-T (60°C)',
            'scraped': 'BOSCH Luftwärmepumpen-Paket BOPA CS792 AW 10 OR-T, CS5800iAW 12 E, BH120,WH290, 7739623435'  # Line 11
        },
        {
            'hplib_title': 'CS5800iAW 10 ORM-T (60°C)',
            'scraped': 'BOSCH Luftwärmepumpen-Paket BOPA CS788 AW 10 OR-T, CS5800iAW 12 M, 7739623429'  # Line 16
        },
        {
            'hplib_title': 'CS5800iAW 12 ORE-T (60°C)',
            'scraped': 'BOSCH Luftwärmepumpen-Paket BOPA CS793 AW 12 OR-T, CS5800iAW 12 E, BH120,WH290, 7739623442'  # Line 12
        },
        {
            'hplib_title': 'CS5800iAW 12 ORM-T (60°C)',
            'scraped': 'BOSCH Luftwärmepumpen-Paket BOPA CS789 AW 12 OR-T, CS5800iAW 12 M, 7739623436'  # Line 9
        },
        {
            'hplib_title': 'CS5800iAW 4 ORM-S (60°C)',
            'scraped': 'BOSCH Luftwärmepumpen-Paket BOPA CS761 AW 4 OR-S, CS5800iAW 12 M, 7739622385'  # Line 4
        },
        {
            'hplib_title': 'CS5800iAW 5 ORM-S (60°C)',
            'scraped': 'BOSCH Luftwärmepumpen-Paket BOPA CS762 AW 5 OR-S, CS5800iAW 12 M, 7739622387'  # Line 7
        },
        # Fix for line 2 - ensure CS5800iAW 7 ORM-S matches with CS763
        {
            'hplib_title': 'CS5800iAW 7 ORM-S',
            'scraped': 'BOSCH Luftwärmepumpen-Paket BOPA CS763 AW 7 OR-S, CS5800iAW 12 M, 7739622388'
        },
        # Add match for CS5800iAW 7 ORM-S (60°C) using the same package as CS5800iAW 7 ORM-S
        {
            'hplib_title': 'CS5800iAW 7 ORM-S (60°C)',
            'scraped': 'BOSCH Luftwärmepumpen-Paket BOPA CS763 AW 7 OR-S, CS5800iAW 12 M, 7739622388'
        }
    ]
    
    # Process CS3400iAWS combinations
    for combo in combinations:
        # Find the hplib row
        hplib_rows = hplib_df[hplib_df['Titel'] == combo['hplib_title']]
        if len(hplib_rows) == 0:
            continue
        hplib_row = hplib_rows.iloc[0]
        
        # Find the outdoor and indoor units in scraped data
        outdoor_rows = scraped_df[scraped_df['Model/Type'] == combo['outdoor']]
        indoor_rows = scraped_df[scraped_df['Model/Type'] == combo['indoor']]
        
        if len(outdoor_rows) == 0 or len(indoor_rows) == 0:
            continue
            
        outdoor_row = outdoor_rows.iloc[0]
        indoor_row = indoor_rows.iloc[0]
        
        # Create combined row
        combined_row = outdoor_row.copy()
        combined_row['Model/Type'] = f"{outdoor_row['Model/Type']} & {indoor_row['Model/Type']}"
        combined_row['Price'] = outdoor_row['Price'] + indoor_row['Price']
        combined_row['Website'] = f"{outdoor_row['Website']} ({outdoor_row['Price']} {outdoor_row['Currency']}) & {indoor_row['Website']} ({indoor_row['Price']} {indoor_row['Currency']})"
        combined_row['Date_scraped'] = f"{outdoor_row['Date_scraped']} & {indoor_row['Date_scraped']}"
        
        manual_matches.append({
            'scraped_row': combined_row,
            'matching_hplib_rows': [hplib_row],
            'is_combined': True
        })
    
    # Process direct matches for CS5800iAW/CS6800iAW models
    for match in cs5800_direct_matches:
        hplib_rows = hplib_df[hplib_df['Titel'] == match['hplib_title']]
        if len(hplib_rows) == 0:
            # Look for the version without (60°C)
            if "(60°C)" in match['hplib_title']:
                base_model = match['hplib_title'].replace(" (60°C)", "")
                hplib_rows = hplib_df[hplib_df['Titel'] == base_model]
                
            if len(hplib_rows) == 0:
                continue
        
        scraped_rows = scraped_df[scraped_df['Model/Type'] == match['scraped']]
        if len(scraped_rows) == 0:
            continue
        
        manual_matches.append({
            'scraped_row': scraped_rows.iloc[0],
            'matching_hplib_rows': [hplib_rows.iloc[0]],
            'is_combined': False
        })
    
    return manual_matches

def main():
    # Load data
    hplib_file = 'data/interim/filtered_hplib_GbM/Bosch_Thermotechnik_GmbH.csv'
    scraped_file = 'data/interim/scraped_data_standardized_GbM/Bosch Thermotechnik GmbH.csv'
    output_file = 'data/processed/matched_manufacturers/bosch_matched.csv'
    
    # Read the files
    hplib_df = pd.read_csv(hplib_file)
    scraped_df = pd.read_csv(scraped_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Find automatic matches
    matches = find_matches(hplib_df, scraped_df)
    
    # Track which hplib titles have already been matched
    matched_hplib_titles = set()
    for match in matches:
        for hplib_row in match['matching_hplib_rows']:
            matched_hplib_titles.add(hplib_row['Titel'])
    
    # Add manual matches, but avoid duplicates
    manual_matches = create_manual_matches(hplib_df, scraped_df)
    for match in manual_matches:
        # Check if any of the hplib rows are already matched
        is_duplicate = False
        for hplib_row in match['matching_hplib_rows']:
            if hplib_row['Titel'] in matched_hplib_titles:
                is_duplicate = True
                break
        
        # Only add if not a duplicate
        if not is_duplicate:
            matches.append(match)
            # Update matched titles
            for hplib_row in match['matching_hplib_rows']:
                matched_hplib_titles.add(hplib_row['Titel'])
    
    # Create output dataframe
    result_df = create_output_dataframe(matches)
    
    # Save results
    result_df.to_csv(output_file, index=False)
    
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