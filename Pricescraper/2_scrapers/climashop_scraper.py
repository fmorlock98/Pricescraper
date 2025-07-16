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

class ClimashopScraper:
    def __init__(self):
        self.options = Options()
        self.options.add_argument('--headless')
        self.options.add_argument('--disable-gpu')
        self.options.add_argument('--no-sandbox')
        self.options.add_argument('--disable-dev-shm-usage')
        self.options.add_argument('--lang=nl-NL')
        
        # Add random user agent
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
        ]
        self.options.add_argument(f'user-agent={random.choice(user_agents)}')
        
        self.service = Service()
        self.driver = webdriver.Chrome(service=self.service, options=self.options)
        self.base_url = "https://climashop.nl/product-category/warmtepompen/"
        self.all_products = []

    def accept_cookies(self):
        """Accept cookies on the website"""
        try:
            print("Waiting for page to load completely...")
            time.sleep(5)
            
            # Try multiple approaches to find and click cookie button
            cookie_selectors = [
                "button.cc-accept-all",
                "button.consent-accept",
                "button.cookie-accept",
                "#cookie-accept",
                "#accept-cookies",
                "button[contains(text(), 'Accepteren')]",
                "button[contains(text(), 'Accept all')]"
            ]
            
            for selector in cookie_selectors:
                try:
                    button = WebDriverWait(self.driver, 3).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    button.click()
                    print("Clicked cookie button")
                    time.sleep(2)
                    return True
                except:
                    continue
            
            print("Could not find cookie accept button, trying to continue anyway...")
            return False
            
        except Exception as e:
            print(f"Error in cookie acceptance: {str(e)}")
            return False

    def scrape_products_from_page(self, page_number):
        """Scrape all products from the current page"""
        try:
            # Wait for product containers to be present
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "li.product"))
            )
            
            # Find all product containers
            product_containers = self.driver.find_elements(By.CSS_SELECTOR, "li.product")
            
            if not product_containers:
                print("No product containers found on page")
                return []
            
            products = []
            for container in product_containers:
                try:
                    # Extract product type and manufacturer
                    try:
                        type_elem = container.find_element(By.CSS_SELECTOR, "h3 a")
                        full_type = type_elem.text.strip()
                        # Extract manufacturer (first word)
                        manufacturer = full_type.split()[0]
                        product_type = full_type
                    except:
                        print("Could not find product type")
                        continue
                    
                    # Extract price
                    try:
                        price_elem = container.find_element(By.CSS_SELECTOR, "span.card__price-new")
                        price_text = price_elem.text.strip()
                        
                        # Convert price to float (handle format like "€5.199,00")
                        price_match = re.search(r'€([\d.,]+)', price_text)
                        if price_match:
                            price_str = price_match.group(1)
                            # Remove thousands separator and replace decimal comma
                            price = float(price_str.replace('.', '').replace(',', '.'))
                            currency = 'EUR'
                        else:
                            price = None
                            currency = None
                    except:
                        print("Could not find price")
                        continue
                    
                    product = {
                        'manufacturer': manufacturer,
                        'type': product_type,
                        'price': price,
                        'currency': currency,
                        'page': page_number,
                        'date_scraped': datetime.now().strftime('%Y-%m-%d')
                    }
                    
                    products.append(product)
                    print(f"Found product: {manufacturer} - {product_type} - {price} {currency} (Page {page_number})")
                    
                except Exception as e:
                    print(f"Error processing product container: {str(e)}")
                    continue
            
            return products
            
        except Exception as e:
            print(f"Error scraping products from page: {str(e)}")
            return []

    def has_next_page(self):
        """Check if there is a next page"""
        try:
            next_link = self.driver.find_element(By.CSS_SELECTOR, "a.next.page-numbers")
            return bool(next_link.get_attribute('href'))
        except:
            return False

    def get_next_page_url(self):
        """Get the URL of the next page"""
        try:
            next_link = self.driver.find_element(By.CSS_SELECTOR, "a.next.page-numbers")
            return next_link.get_attribute('href')
        except:
            return None

    def go_to_next_page(self):
        """Navigate to the next page"""
        try:
            next_url = self.get_next_page_url()
            if next_url:
                self.driver.get(next_url)
                time.sleep(random.uniform(2, 4))
                return True
            return False
        except Exception as e:
            print(f"Error navigating to next page: {str(e)}")
            return False

    def scrape_all(self):
        """Scrape all products from all pages"""
        try:
            # Start with the first page
            self.driver.get(self.base_url)
            time.sleep(random.uniform(2, 4))
            
            # Try cookie acceptance
            self.accept_cookies()
            
            page_number = 1
            while True:
                print(f"\nProcessing page {page_number}")
                
                # Scrape current page
                products = self.scrape_products_from_page(page_number)
                self.all_products.extend(products)
                
                # Save progress after each page
                self.save_progress()
                
                # Check for next page
                if not self.has_next_page():
                    print("No more pages to process")
                    break
                
                # Go to next page
                if not self.go_to_next_page():
                    print("Failed to navigate to next page")
                    break
                
                page_number += 1
                time.sleep(random.uniform(2, 4))
                
        except Exception as e:
            print(f"Error in scrape_all: {str(e)}")

    def save_progress(self):
        """Save current progress to CSV"""
        if self.all_products:
            current_date = datetime.now().strftime('%Y%m%d')
            filename = f'climashop_heatpumps_{current_date}.csv'
            
            # Create scraped_data directory if it doesn't exist
            os.makedirs(os.path.join('data', 'raw', 'scraped_data'), exist_ok=True)
            
            df = pd.DataFrame(self.all_products)
            df.to_csv(os.path.join('data', 'raw', 'scraped_data', filename), index=False)
            print(f"Progress saved: {len(self.all_products)} products found")

    def close(self):
        """Close the browser"""
        self.driver.quit()

def main():
    """Main function to run the scraper"""
    try:
        scraper = ClimashopScraper()
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