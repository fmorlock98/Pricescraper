import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import re
import os
from datetime import datetime
import random

class EibmarktScraper:
    def __init__(self):
        self.options = Options()
        self.options.add_argument('--headless')
        self.options.add_argument('--disable-gpu')
        self.options.add_argument('--no-sandbox')
        self.options.add_argument('--disable-dev-shm-usage')
        self.options.add_argument('--lang=de-DE')
        
        # Add random user agent
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
        ]
        self.options.add_argument(f'user-agent={random.choice(user_agents)}')
        
        chromedriver_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'chromedriver')
        service = Service(executable_path=chromedriver_path)
        
        self.driver = webdriver.Chrome(service=service, options=self.options)
        self.base_url = "https://www.eibmarkt.com/cgi-bin/eibmarkt.storefront/67b8474e0005b9ea2746ac1e040205da/Ext/Custom/Search?query=W%E4rmepumpe&filterCatalogs=EC000393"
        
    def scrape_page(self, page_number):
        """Scrape a specific page"""
        try:
            url = f"{self.base_url}&page={page_number}"
            
            # Random delay between requests (2-5 seconds)
            time.sleep(random.uniform(2, 5))
            
            self.driver.get(url)
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            products = []
            
            # Find all product containers
            product_containers = soup.find_all('td', {'width': '50%', 'valign': 'top', 'style': 'padding:15px'})
            
            print(f"Found {len(product_containers)} products on page {page_number}")
            
            for container in product_containers:
                try:
                    # Find the inner table that contains product info
                    product_table = container.find('table')
                    if not product_table:
                        continue
                    
                    # Extract manufacturer from ProductPreviewImage cell
                    manufacturer_cell = product_table.find('td', {'class': 'ProductPreviewImage'})
                    if manufacturer_cell:
                        # Get text and split by line breaks, take the first non-empty line
                        manufacturer_lines = [line.strip() for line in manufacturer_cell.get_text().split('\n') if line.strip()]
                        manufacturer = manufacturer_lines[0] if manufacturer_lines else "Unknown"
                    else:
                        manufacturer = "Unknown"
                    
                    # Extract type from productlink
                    type_link = product_table.find('a', {'class': 'productlink'})
                    product_type = type_link.text.strip() if type_link else "Unknown"
                    
                    # Extract price from price span
                    price_span = product_table.find('span', {'class': 'price'})
                    price = None
                    currency = None
                    if price_span:
                        price_link = price_span.find('a')
                        if price_link:
                            price_text = price_link.text.strip()
                            # Extract price and currency
                            price_match = re.search(r'([\d.,]+)\s*(EUR|€)', price_text)
                            if price_match:
                                price = float(price_match.group(1).replace('.', '').replace(',', '.'))
                                currency = price_match.group(2)
                    
                    print(f"Found product: {manufacturer} - {product_type} - {price} {currency}")
                    
                    products.append({
                        'manufacturer': manufacturer,
                        'type': product_type,
                        'price': price,
                        'currency': currency,
                        'page': page_number,
                        'date_scraped': datetime.now().strftime('%Y-%m-%d')
                    })
                    
                except Exception as e:
                    print(f"Error processing product: {str(e)}")
                    continue
            
            return products
            
        except Exception as e:
            print(f"Error scraping page {page_number}: {str(e)}")
            # Save screenshot for debugging
            self.driver.save_screenshot(os.path.join('data', 'raw', 'scraped_data', f'error_page_{page_number}.png'))
            return []
    
    def close(self):
        self.driver.quit()

def main():
    # Create necessary directories if they don't exist
    os.makedirs(os.path.join('data', 'raw', 'scraped_data'), exist_ok=True)
    
    # Create filename with current date
    current_date = datetime.now().strftime('%Y%m%d')
    filename = f'eibmarkt_heatpumps_{current_date}.csv'
    
    scraper = EibmarktScraper()
    all_products = []
    
    try:
        # Set total pages to 18 (known value)
        total_pages = 18
        print(f"Starting to scrape {total_pages} pages")
        
        # Scrape each page
        for page in range(1, total_pages + 1):
            print(f"Scraping page {page}/{total_pages}")
            
            products = scraper.scrape_page(page)
            all_products.extend(products)
            
            # Save progress every 5 pages
            if page % 5 == 0:
                df = pd.DataFrame(all_products)
                df.to_csv(os.path.join('data', 'raw', 'scraped_data', filename), index=False)
                print(f"Progress saved: {len(all_products)} products found")
            
            # Random longer delay between pages (5-10 seconds)
            time.sleep(random.uniform(5, 10))
        
        # Final save
        if all_products:
            df = pd.DataFrame(all_products)
            df.to_csv(os.path.join('data', 'raw', 'scraped_data', filename), index=False)
            print(f"\nScraping completed. Found {len(all_products)} products.")
            print(f"Results saved to data/raw/scraped_data/{filename}")
        else:
            print("\nNo products found")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
        # Save whatever we have if there's an error
        if all_products:
            error_filename = f'eibmarkt_heatpumps_{current_date}_partial.csv'
            df = pd.DataFrame(all_products)
            df.to_csv(os.path.join('data', 'raw', 'scraped_data', error_filename), index=False)
            print(f"Saved {len(all_products)} products to data/raw/scraped_data/{error_filename}")
    
    finally:
        scraper.close()

if __name__ == "__main__":
    main() 