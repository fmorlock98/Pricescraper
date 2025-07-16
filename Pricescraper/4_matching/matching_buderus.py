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
    
    # Try to match common model series patterns for Buderus heat pumps
    patterns = [
        # WLW series with all variations
        r'wlw\s*\d+[a-z]*[-]?\d*\s*(?:ar)?\s*(?:\d+)?\s*(?:e|tp70|t180)?',  # Base pattern with optional parts
        
        # More specific patterns for different combinations
        r'wlw\s*\d+[a-z]*[-]?\d*\s*ar\s*\d+\s*e',     # AR with E (wall-mounted)
        r'wlw\s*\d+[a-z]*[-]?\d*\s*ar\s*tp70',         # AR with TP70 (floor-standing)
        r'wlw\s*\d+[a-z]*[-]?\d*\s*ar\s*t180',         # AR with T180 (tank)
        r'wlw\s*\d+[a-z]*[-]?\d*\s*ar\s*\d+',          # AR with power number
        
        # CS series patterns
        r'cs\s*\d+[a-z]*[-]?\d*',
        
        # WPLS patterns
        r'wpls\s*\d+[a-z]*[-]?\d*',
        
        # Logaplus patterns
        r'logaplus\s*wlw\s*\d+[a-z]*[-]?\d*\s*(?:ar)?\s*(?:\d+)?\s*(?:e|tp70|t180)?',
        
        # SP AR patterns
        r'wlw\s*\d+[a-z]*[-]?\d*\s*sp\s*ar\s*(?:b|e|t190)?'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, s)
        if match:
            # Remove 'logaplus' and 'sp ar' from the match if present
            result = match.group(0)
            if 'logaplus' in result:
                result = result.replace('logaplus', '').strip()
            if 'sp ar' in result:
                result = result.replace('sp ar', '').strip()
            return result
    return ""

def extract_numbers(s):
    """Extract numbers from a string"""
    if pd.isna(s):
        return []
    numbers = re.findall(r'\d+(?:[.,]\d+)?', str(s))
    return [float(n.replace(',', '.')) for n in numbers]

def clean_string(s):
    """Clean string by removing special characters and extra spaces"""
    if pd.isna(s):
        return ""
    # Convert to string and remove special characters
    s = str(s).lower()
    # Remove (60°C) pattern
    s = re.sub(r'\(60°c\)', '', s)
    # Replace Logaplus with Logatherm
    s = s.replace('logaplus', 'logatherm')
    # Remove SP AR pattern
    s = re.sub(r'sp\s*ar\s*', '', s)
    # Remove special characters and extra spaces
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    # Remove extra spaces
    s = ' '.join(s.split())
    return s

def normalize_brand_names(s):
    """Normalize brand name variations"""
    if pd.isna(s):
        return ""
    s = str(s).lower()
    replacements = {
        'buderus': 'buderus',
        'bosch': 'bosch',
        'thermotechnik': 'thermotechnik',
        'logaplus': 'logatherm'
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s

def interpret_model_type(s):
    """Interpret specific aspects of the model type"""
    if pd.isna(s):
        return set()
    
    s = str(s).lower()
    features = set()
    
    # Remove 'w' suffix from the string for feature detection
    s_clean = re.sub(r'\s*w$', '', s)
    
    # Check for mounting type
    if ('e' in s_clean.split() or '-e' in s_clean or 
        'ar 12 e' in s_clean or 'ar12e' in s_clean or 
        'ar 12e' in s_clean or 'sp ar e' in s_clean):
        features.add('wall_mounted')
    if ('tp70' in s_clean or 'tp 70' in s_clean or 
        'pufferspeicher 70' in s_clean):
        features.add('floor_standing')
        
    # Check for tank
    if ('t180' in s_clean or 't 180' in s_clean or 
        'warmwasserspeicher 180' in s_clean or 
        't190' in s_clean or 'warmwasserspeicher' in s_clean):
        features.add('tank_180l')
        
    # Check for power rating after AR
    power_match = re.search(r'ar\s*(\d+)', s_clean)
    if power_match:
        features.add(f'ar_power_{power_match.group(1)}')
    
    # Check for operation type
    if 'b' in s_clean.split() or 'bivalenten' in s_clean:
        features.add('bivalent')
    if 'e' in s_clean.split() or 'monoenergetischen' in s_clean:
        features.add('mono')
    
    return features

def is_matching_pair(title, model_type):
    """Check if title and model_type are a matching pair"""
    if pd.isna(title) or pd.isna(model_type):
        return False
        
    # Clean and normalize strings
    title_clean = clean_string(title)
    model_type_clean = clean_string(model_type)
    
    # Normalize brand names
    title_clean = normalize_brand_names(title_clean)
    model_type_clean = normalize_brand_names(model_type_clean)
    
    # Extract model series
    title_series = extract_model_series(title_clean)
    model_type_series = extract_model_series(model_type_clean)
    
    # Extract features from both title and model type
    title_features = interpret_model_type(title_clean)
    model_features = interpret_model_type(model_type_clean)
    
    # Check for other feature matches
    # If one has a feature, the other must have it too
    other_features = {'wall_mounted', 'floor_standing', 'tank_180l', 'bivalent', 'mono'}
    for feature in other_features:
        title_has_feature = feature in title_features
        model_has_feature = feature in model_features
        
        # If one has the feature and the other doesn't, they don't match
        if title_has_feature != model_has_feature:
            return False
            
    # Check AR power rating if present
    title_power = next((f for f in title_features if f.startswith('ar_power_')), None)
    model_power = next((f for f in model_features if f.startswith('ar_power_')), None)
    if title_power and model_power and title_power != model_power:
        return False
    
    # If we found model series in both, they should match
    if title_series and model_type_series:
        if title_series != model_type_series:
            return False
    
    # Extract and compare numbers
    title_numbers = extract_numbers(title)
    model_type_numbers = extract_numbers(model_type)
    
    # If there are numbers in the title, at least one should match in model_type
    if title_numbers:
        number_match = False
        for title_num in title_numbers:
            if any(abs(title_num - model_num) < 0.1 for model_num in model_type_numbers):
                number_match = True
                break
        if not number_match:
            return False
    
    # Check if all words from title appear in model_type in order
    # Remove 'w' suffix from both strings for comparison
    title_words = re.sub(r'\s*w$', '', title_clean).split()
    model_type_words = re.sub(r'\s*w$', '', model_type_clean).split()
    
    i = 0  # index in title_words
    j = 0  # index in model_type_words
    
    while i < len(title_words) and j < len(model_type_words):
        if title_words[i] in model_type_words[j]:
            i += 1
        j += 1
        
        if j == len(model_type_words) and i < len(title_words):
            return False
            
    return i == len(title_words)

def load_and_prepare_data():
    # Read the source files
    hplib_file = 'data/interim/filtered_hplib_GbM/Bosch_Thermotechnik_GmbH_(Buderus).csv'
    scraped_file = 'data/interim/scraped_data_standardized_GbM/Bosch Thermotechnik GmbH (Buderus).csv'
    output_file = 'data/processed/matched_manufacturers/buderus_matched.csv'
    
    # Read the files
    hplib_df = pd.read_csv(hplib_file)
    scraped_df = pd.read_csv(scraped_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    return hplib_df, scraped_df, output_file

def find_matches(hplib_df, scraped_df):
    """Find matches between hplib and scraped data using enhanced matching.
    Each scraped item can match multiple hplib items, but each hplib item can only be matched once."""
    matches = []
    used_hplib_indices = set()  # Track which hplib items have been matched
    
    # Find all potential matches for each scraped item
    for scraped_idx, scraped_row in scraped_df.iterrows():
        model_type = scraped_row['Model/Type']
        if pd.isna(model_type):
            continue
            
        # Find all potential matching hplib rows for this scraped item
        matching_hplib_rows = []
        for hplib_idx, hplib_row in hplib_df.iterrows():
            if hplib_idx in used_hplib_indices:  # Skip already matched hplib items
                continue
                
            titel = hplib_row['Titel']
            if pd.isna(titel):
                continue
                
            if is_matching_pair(titel, model_type):
                matching_hplib_rows.append(hplib_row)
                used_hplib_indices.add(hplib_idx)
        
        if matching_hplib_rows:
            matches.append({
                'scraped_row': scraped_row,
                'matching_hplib_rows': matching_hplib_rows
            })
    
    return matches

def create_output_dataframe(matches):
    """Create the output dataframe from matches with specified column order"""
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

def main():
    # Load data
    hplib_file = 'data/interim/filtered_hplib_GbM/Bosch_Thermotechnik_GmbH_(Buderus).csv'
    scraped_file = 'data/interim/scraped_data_standardized_GbM/Bosch Thermotechnik GmbH Buderus.csv'
    output_file = 'data/processed/matched_manufacturers/buderus_matched.csv'
    
    # Read the files
    hplib_df = pd.read_csv(hplib_file)
    scraped_df = pd.read_csv(scraped_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Find matches
    matches = find_matches(hplib_df, scraped_df)
    
    # Create output dataframe
    result_df = create_output_dataframe(matches)
    
    # Save results
    result_df.to_csv(output_file, index=False)
    
    # Track which items have been matched
    matched_hplib_titles = set()
    matched_scraped_indices = set()
    
    for match in matches:
        # Track matched scraped items
        matched_scraped_indices.add(match['scraped_row'].name)
        # Track matched hplib items
        for hplib_row in match['matching_hplib_rows']:
            matched_hplib_titles.add(hplib_row['Titel'])
    
    # Find unmatched items
    unmatched_hplib = hplib_df[~hplib_df['Titel'].isin(matched_hplib_titles)]
    unmatched_scraped = scraped_df[~scraped_df.index.isin(matched_scraped_indices)]
    
    # Print statistics
    print("\n=== Matching Statistics ===")
    print(f"Total items in scraped data: {len(scraped_df)}")
    print(f"Total items in hplib data: {len(hplib_df)}")
    print(f"Total matches found: {len(matches)}")
    print(f"Total rows in output: {len(result_df)}")
    print(f"Unmatched hplib items: {len(unmatched_hplib)}")
    print(f"Unmatched scraped items: {len(unmatched_scraped)}")
    
    # Print detailed matching information
    print("\n=== Detailed Matching Information ===")
    for match in matches:
        print(f"\nScraped item: {match['scraped_row']['Model/Type']}")
        print("Matches with:")
        for hplib_row in match['matching_hplib_rows']:
            print(f"- {hplib_row['Titel']}")
    
    # Ask user if they want to see unmatched items
    print("\nWould you like to see unmatched items? (Enter 'y' for yes):")
    print("1. Show unmatched database items")
    print("2. Show unmatched scraped items")
    print("3. Show both")
    print("4. Skip (press Enter)")
    
    user_input = input().strip().lower()
    
    if user_input == '1' or user_input == '3':
        print("\n=== Unmatched Items from Filtered Database ===")
        for _, row in unmatched_hplib.iterrows():
            print(f"- {row['Titel']}")
            
    if user_input == '2' or user_input == '3':
        print("\n=== Unmatched Items from Scraped Data ===")
        for _, row in unmatched_scraped.iterrows():
            print(f"- {row['Model/Type']}")

if __name__ == "__main__":
    main() 