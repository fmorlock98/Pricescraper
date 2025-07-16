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
    # Keep dots for model numbers (like 8.2)
    return s.strip().upper()

def convert_to_eur(amount, currency):
    """Convert amount from given currency to EUR using current exchange rates."""
    # Current exchange rates (as of April 2023)
    rates = {
        'EUR': 1.0,
        'CHF': 1.03,  # 1 CHF = 1.03 EUR (example rate, replace with actual)
        'GBP': 1.17,  # 1 GBP = 1.17 EUR
        '€': 1.0      # Euro symbol is same as EUR
    }
    
    # Return converted amount or original if currency not found
    converted = amount * rates.get(currency, 1.0)
    # Round to 2 decimal places to avoid floating point precision issues
    return round(converted, 2)

def find_matches(hplib_df, scraped_df):
    """Find matches between database and scraped data, keeping only the cheapest match for each item."""
    matches = []
    
    for _, hplib_row in hplib_df.iterrows():
        hplib_title = str(hplib_row['Titel']) if pd.notna(hplib_row['Titel']) else ""
        
        # Skip if no title
        if not hplib_title:
            continue
        
        # Split by '+' to get parts (outdoor unit, indoor unit, boiler)
        parts = [part.strip() for part in hplib_title.split('+')]
        
        # For single unit systems
        if len(parts) == 1:
            matching_scraped_rows = []
            for _, scraped_row in scraped_df.iterrows():
                scraped_model = clean_string(scraped_row['Model/Type'])
                if not scraped_model:
                    continue
                
                if clean_string(parts[0]) in scraped_model:
                    matching_scraped_rows.append(scraped_row)
            
            if matching_scraped_rows:
                # Find the cheapest match (handling potential non-numeric prices)
                cheapest_row = find_cheapest_row(matching_scraped_rows)
                matches.append({
                    'scraped_rows': [cheapest_row],
                    'matching_hplib_row': hplib_row,
                    'is_combined': False
                })
        
        # For split systems (outdoor + indoor units + possibly storage)
        else:
            # Initialize a dictionary to track matching parts
            matched_parts = {part: [] for part in parts}
            
            # Try to find matches for each part
            for part in parts:
                clean_part = clean_string(part)
                for _, scraped_row in scraped_df.iterrows():
                    scraped_model = clean_string(scraped_row['Model/Type'])
                    if not scraped_model:
                        continue
                    
                    # Special handling for VIH RW 300/3 BR storage unit
                    if "VIH RW 300/3 BR" in part and "VIH RW 300/3 BR" in scraped_model:
                        matched_parts[part].append(scraped_row)
                        continue
                    
                    # Exact part number match for VWL units (more strict)
                    part_number = re.search(r'VWL\s+\d+(/\d+\.\d+)?\s+[A-Z]+', clean_part)
                    if part_number and part_number.group() in scraped_model:
                        matched_parts[part].append(scraped_row)
                        continue
                    
                    # Fallback to partial match
                    if clean_part in scraped_model:
                        matched_parts[part].append(scraped_row)
            
            # Check if all parts have matches
            all_parts_matched = all(len(matches_for_part) > 0 for matches_for_part in matched_parts.values())
            
            if all_parts_matched:
                # Find the cheapest option for each part
                cheapest_matches = [find_cheapest_row(matches_for_part) for matches_for_part in matched_parts.values()]
                
                matches.append({
                    'scraped_rows': cheapest_matches,
                    'matching_hplib_row': hplib_row,
                    'is_combined': True
                })
    
    return matches

def find_cheapest_row(rows):
    """Find the row with the lowest price among a list of rows."""
    valid_rows = []
    
    for row in rows:
        try:
            # Try to convert price to float and normalize to EUR
            price = float(row['Price'])
            currency = row['Currency']
            price_eur = convert_to_eur(price, currency)
            # If conversion succeeded, it's a valid numeric price
            valid_rows.append((price_eur, row))
        except (ValueError, TypeError):
            # Skip rows with non-numeric prices
            continue
    
    if valid_rows:
        # Sort by price and get the cheapest
        valid_rows.sort(key=lambda x: x[0])
        return valid_rows[0][1]
    else:
        # If no valid prices, return the first row
        return rows[0]

def combine_scraped_rows(scraped_rows):
    """Combine multiple scraped rows into one."""
    combined = {}
    
    # Combine Model/Type and Date_scraped with '&'
    for field in ['Model/Type', 'Date_scraped']:
        combined[field] = ' & '.join(str(row[field]) for row in scraped_rows)
    
    # Combine Website and Price with format "website1 (price1) & website2 (price2)"
    websites = []
    for row in scraped_rows:
        website = str(row['Website'])
        price = str(row['Price'])
        currency = str(row['Currency'])
        websites.append(f"{website} ({price} {currency})")
    combined['Website'] = ' & '.join(websites)
    
    # Calculate total price by converting all to EUR first
    total_price_eur = 0
    all_numeric = True
    
    for row in scraped_rows:
        try:
            price = float(row['Price'])
            currency = row['Currency']
            price_eur = convert_to_eur(price, currency)
            total_price_eur += price_eur
        except (ValueError, TypeError):
            all_numeric = False
            break
    
    if all_numeric:
        # Round the total price to 2 decimal places
        combined['Price'] = round(total_price_eur, 2)
        combined['Currency'] = 'EUR'
    else:
        # If can't convert all prices, keep original format
        combined['Price'] = ' & '.join(str(row['Price']) for row in scraped_rows)
        combined['Currency'] = ' & '.join(str(row['Currency']) for row in scraped_rows)
    
    return combined

def create_output_dataframe(matches):
    """Create output dataframe from matches."""
    if not matches:
        return pd.DataFrame()
    
    # Create list of rows for the output dataframe
    rows = []
    for match in matches:
        hplib_row = match['matching_hplib_row']
        
        # Create row with all database columns
        row = hplib_row.to_dict()
        
        # Add scraped data columns
        if match['is_combined']:
            combined = combine_scraped_rows(match['scraped_rows'])
            for key, value in combined.items():
                row[key] = value
        else:
            scraped_row = match['scraped_rows'][0]
            row['Model/Type'] = scraped_row['Model/Type']
            row['Price'] = scraped_row['Price']
            row['Currency'] = scraped_row['Currency']
            row['Website'] = scraped_row['Website']
            row['Date_scraped'] = scraped_row['Date_scraped']
        
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
    hplib_file = 'data/interim/filtered_hplib_GbM/Vaillant_GmbH.csv'
    scraped_file = 'data/interim/scraped_data_standardized_GbM/Vaillant GmbH.csv'
    output_file = 'data/processed/matched_manufacturers/vaillant_matched.csv'
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Read the files
    hplib_df = pd.read_csv(hplib_file)
    scraped_df = pd.read_csv(scraped_file)
    
    # Add manually specified item
    manual_item = {
        'Model/Type': 'Vaillant uniSTOR plus VIH RW 300/3 BR Wärmepumpenspeicher',
        'Price': 2590.0,
        'Currency': 'CHF',
        'Website': 'MANUALLY',
        'Date_scraped': '2025-03-13'
    }
    scraped_df = pd.concat([scraped_df, pd.DataFrame([manual_item])], ignore_index=True)
    
    # Find matches
    matches = find_matches(hplib_df, scraped_df)
    
    # Create output dataframe
    result_df = create_output_dataframe(matches)
    
    # Save results
    result_df.to_csv(output_file, index=False)
    
    # Track which hplib titles have already been matched
    matched_hplib_titles = set()
    for match in matches:
        matched_hplib_titles.add(match['matching_hplib_row']['Titel'])
    
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