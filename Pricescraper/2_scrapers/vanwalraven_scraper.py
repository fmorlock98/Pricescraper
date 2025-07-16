import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re
import os
from datetime import datetime
import random

class VanWalravenScraper:
    def __init__(self):
        self.options = Options()
        self.options.add_argument('--headless')
        self.options.add_argument('--disable-gpu')
        self.options.add_argument('--no-sandbox')
        self.options.add_argument('--disable-dev-shm-usage')
        
        # Add random user agent
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
        ]
        self.options.add_argument(f'user-agent={random.choice(user_agents)}')
        
        self.service = Service()
        self.driver = webdriver.Chrome(service=self.service, options=self.options)
        self.base_url = "https://www.vanwalraven.com/en/catalog/heating-cooling/heat-pumps/groups/g+c+view"
        self.all_products = []
        self.current_page = 1

    def accept_cookies(self):
        """Accept cookies on the website"""
        try:
            # Wait for cookie banner and accept button
            cookie_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button[name='submit-button']"))
            )
            cookie_button.click()
            print("Accepted cookies")
            time.sleep(2)
            return True
        except Exception as e:
            print(f"Error accepting cookies: {str(e)}")
            return False

    def get_manufacturer_links(self):
        """Get all manufacturer links from the main heat pumps page"""
        try:
            manufacturer_links = []
            
            # List of manufacturers to exclude
            excluded_manufacturers = [
                "Itho Daalderop",
                "Intergas",
                "Inventum",
                "Atlantic",
                "Masterwatt"
            ]
            
            # Find all manufacturer cards
            manufacturer_cards = self.driver.find_elements(By.CSS_SELECTOR, "div.catalog-card-group.js-link")
            
            for card in manufacturer_cards:
                try:
                    # Get the link and manufacturer name
                    link_elem = card.find_element(By.CSS_SELECTOR, "h2 a")
                    href = link_elem.get_attribute('href')
                    name = link_elem.text.strip()
                    
                    # Skip "Heat pumps accessories" and excluded manufacturers
                    if ("accessories" in name.lower() or 
                        any(mfr.lower() in name.lower() for mfr in excluded_manufacturers)):
                        print(f"Skipping excluded manufacturer: {name}")
                        continue
                    
                    if href and name:
                        manufacturer_links.append((name, href))
                        print(f"Found manufacturer: {name}")
                except Exception as e:
                    print(f"Error processing manufacturer card: {str(e)}")
                    continue
            
            return manufacturer_links
        except Exception as e:
            print(f"Error getting manufacturer links: {str(e)}")
            return []

    def get_model_links(self, manufacturer_url):
        """Get all heat pump model links from a manufacturer's page"""
        try:
            model_links = []
            self.driver.get(manufacturer_url)
            time.sleep(random.uniform(2, 4))
            
            # Find all model cards
            model_cards = self.driver.find_elements(By.CSS_SELECTOR, "div.catalog-card-group.js-link")
            
            for card in model_cards:
                try:
                    # Get the link and model name
                    link_elem = card.find_element(By.CSS_SELECTOR, "h2 a")
                    href = link_elem.get_attribute('href')
                    name = link_elem.text.strip()
                    
                    # Skip if it contains "accessories"
                    if "accessories" in name.lower():
                        print(f"Skipping accessories: {name}")
                        continue
                    
                    if href and name:
                        model_links.append((name, href))
                        print(f"Found model: {name}")
                except Exception as e:
                    print(f"Error processing model card: {str(e)}")
                    continue
            
            return model_links
        except Exception as e:
            print(f"Error getting model links: {str(e)}")
            return []

    def scrape_products_from_model(self, model_url, manufacturer):
        """Scrape all product variants from a model page"""
        try:
            self.driver.get(model_url)
            time.sleep(random.uniform(2, 4))
            
            # First try to find product rows in table
            product_rows = self.driver.find_elements(By.CSS_SELECTOR, "tr.odd, tr.even")
            
            if product_rows:
                # Handle table-based product page
                for row in product_rows:
                    try:
                        # Extract product information
                        product_link = row.find_element(By.CSS_SELECTOR, "td.spec.spec-1.description a")
                        product_type = product_link.text.strip()
                        
                        # Extract price
                        price_elem = row.find_element(By.CSS_SELECTOR, "td.listprice")
                        price_text = price_elem.text.strip()
                        
                        # Convert price to float
                        price_match = re.search(r'€\s*([\d.,]+)', price_text)
                        if price_match:
                            price = float(price_match.group(1).replace('.', '').replace(',', '.'))
                            currency = 'EUR'
                        else:
                            price = None
                            currency = None
                        
                        product = {
                            'manufacturer': manufacturer,
                            'type': product_type,
                            'price': price,
                            'currency': currency,
                            'page': self.current_page,
                            'date_scraped': datetime.now().strftime('%Y-%m-%d')
                        }
                        
                        self.all_products.append(product)
                        print(f"Found product: {product_type} - {price} {currency}")
                        
                    except Exception as e:
                        print(f"Error processing product row: {str(e)}")
                        continue
            else:
                # Handle single product page
                try:
                    # Get product type from title
                    title_elem = self.driver.find_element(By.CSS_SELECTOR, "div.titlebar h1 span")
                    product_type = title_elem.text.strip()
                    
                    # Get price
                    price_elem = self.driver.find_element(By.CSS_SELECTOR, "span.price.price-final")
                    price_text = price_elem.text.strip()
                    
                    # Convert price to float
                    price_match = re.search(r'€\s*([\d.,]+)', price_text)
                    if price_match:
                        price = float(price_match.group(1).replace('.', '').replace(',', '.'))
                        currency = 'EUR'
                    else:
                        price = None
                        currency = None
                    
                    product = {
                        'manufacturer': manufacturer,
                        'type': product_type,
                        'price': price,
                        'currency': currency,
                        'page': self.current_page,
                        'date_scraped': datetime.now().strftime('%Y-%m-%d')
                    }
                    
                    self.all_products.append(product)
                    print(f"Found single product: {product_type} - {price} {currency}")
                    
                except Exception as e:
                    print(f"Error processing single product page: {str(e)}")
            
            # Increment page counter
            self.current_page += 1
                
        except Exception as e:
            print(f"Error scraping products from model page: {str(e)}")

    def save_progress(self):
        """Save current progress to CSV"""
        if self.all_products:
            current_date = datetime.now().strftime('%Y%m%d')
            filename = f'vanwalraven_heatpumps_{current_date}.csv'
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.join('data', 'raw', 'scraped_data'), exist_ok=True)
            
            df = pd.DataFrame(self.all_products)
            df.to_csv(os.path.join('data', 'raw', 'scraped_data', filename), index=False)
            print(f"Progress saved: {len(self.all_products)} products found")

    def scrape_all(self):
        """Scrape all products from all manufacturers"""
        try:
            # Navigate to main page and accept cookies
            self.driver.get(self.base_url)
            time.sleep(random.uniform(2, 4))
            
            self.accept_cookies()
            
            # Get all manufacturer links
            manufacturer_links = self.get_manufacturer_links()
            
            for manufacturer_name, manufacturer_url in manufacturer_links:
                print(f"\nProcessing manufacturer: {manufacturer_name}")
                
                # Get all model links for this manufacturer
                model_links = self.get_model_links(manufacturer_url)
                
                for model_name, model_url in model_links:
                    print(f"\nProcessing model: {model_name}")
                    
                    # Scrape all products for this model
                    self.scrape_products_from_model(model_url, manufacturer_name)
                    
                    # Save progress after each model
                    self.save_progress()
                    
                    # Random delay between models
                    time.sleep(random.uniform(2, 4))
                
                # Random delay between manufacturers
                time.sleep(random.uniform(3, 5))
            
        except Exception as e:
            print(f"Error in scrape_all: {str(e)}")
            self.driver.save_screenshot(os.path.join('data', 'raw', 'scraped_data', 'scrape_all_error.png'))

    def close(self):
        """Close the browser"""
        self.driver.quit()

def main():
    """Main function to run the scraper"""
    try:
        scraper = VanWalravenScraper()
        scraper.scrape_all()
        
        # Final save
        scraper.save_progress()
        print(f"\nScraping completed. Found {len(scraper.all_products)} products.")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    finally:
        scraper.close()

if __name__ == "__main__":
    main() 