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

class ClimaMarketScraper:
    def __init__(self):
        self.options = Options()
        self.options.add_argument('--headless')
        self.options.add_argument('--disable-gpu')
        self.options.add_argument('--no-sandbox')
        self.options.add_argument('--disable-dev-shm-usage')
        self.options.add_argument('--disable-blink-features=AutomationControlled')
        self.options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.options.add_experimental_option('useAutomationExtension', False)
        
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
        ]
        self.options.add_argument(f'user-agent={random.choice(user_agents)}')
        
        self.service = Service()
        self.driver = webdriver.Chrome(service=self.service, options=self.options)
        
        self.urls = [
            "https://www.climamarket.eu/en/monobloc-air-to-water-heat-pump",
            "https://www.climamarket.eu/en/bibloc-air-to-water-heat-pump",
            "https://www.climamarket.eu/en/water-heater-heat-pump"
        ]
        
        self.all_products = []

    def accept_cookies(self):
        """Accept cookies on the website"""
        try:
            time.sleep(3)
            cookie_button = self.driver.find_element(By.CSS_SELECTOR, "button.cookie-accept-all")
            cookie_button.click()
            time.sleep(2)
        except:
            print("No cookie banner found or already accepted")

    def scrape_products_from_page(self, url):
        """Scrape all products from a page"""
        try:
            products = []
            print(f"Scraping products from: {url}")
            
            # Wait for product grid to load
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.products.row.product_content.grid"))
            )
            
            # Additional wait for dynamic content
            time.sleep(5)
            
            # Use JavaScript to get all products
            product_elements = self.driver.execute_script("""
                return Array.from(document.querySelectorAll('div.products.row.product_content.grid > div.item-product')).map(container => {
                    const titleElem = container.querySelector('a.product_name');
                    const priceElem = container.querySelector('span.price.price_sale');
                    return {
                        title: titleElem ? titleElem.getAttribute('title') : null,
                        price: priceElem ? priceElem.textContent.trim() : null
                    };
                });
            """)
            
            print(f"Found {len(product_elements)} products via JavaScript")
            
            for product in product_elements:
                try:
                    if product['title'] and product['price']:
                        # Extract manufacturer (first word) and type
                        manufacturer = product['title'].split()[0]
                        product_type = product['title']
                        
                        # Parse price
                        price_match = re.search(r'([\d.,]+)', product['price'])
                        if price_match:
                            price = float(price_match.group(1).replace(',', ''))
                            
                            product_data = {
                                'manufacturer': manufacturer,
                                'type': product_type,
                                'price': price,
                                'currency': 'EUR',
                                'page': url,
                                'date_scraped': datetime.now().strftime('%Y-%m-%d')
                            }
                            
                            products.append(product_data)
                            print(f"Found product: {manufacturer} - {product_type} - €{price}")
                
                except Exception as e:
                    print(f"Error processing product: {str(e)}")
                    continue
            
            return products
            
        except Exception as e:
            print(f"Error scraping products: {str(e)}")
            self.driver.save_screenshot(os.path.join('data', 'raw', 'scraped_data', 'scrape_error.png'))
            return []

    def get_next_page(self):
        """Check if there's a next page and return its URL"""
        try:
            # Wait for pagination to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "ul.page-list"))
            )
            
            # Look specifically for the next page link
            next_button = self.driver.find_element(
                By.CSS_SELECTOR, 
                "a.next[rel='next']"
            )
            next_url = next_button.get_attribute('href')
            print(f"Found next page URL: {next_url}")
            return next_url
        except:
            print("No next page found")
            return None

    def scrape_all(self):
        """Main scraping function"""
        try:
            # Create scraped_data directory if it doesn't exist
            os.makedirs(os.path.join('data', 'raw', 'scraped_data'), exist_ok=True)
            
            # Process each main URL
            for base_url in self.urls:
                print(f"\nProcessing URL: {base_url}")
                current_url = base_url
                page_number = 1
                visited_urls = set()  # Keep track of visited URLs
                
                while current_url and current_url not in visited_urls:
                    print(f"\nScraping page {page_number}")
                    visited_urls.add(current_url)  # Add current URL to visited set
                    
                    self.driver.get(current_url)
                    time.sleep(random.uniform(2, 4))
                    
                    # Accept cookies on first page
                    if page_number == 1:
                        self.accept_cookies()
                        time.sleep(2)
                    
                    # Scrape products from current page
                    products = self.scrape_products_from_page(current_url)
                    if products:
                        self.all_products.extend(products)
                        # Save progress after each page
                        self.save_progress()
                    
                    # Get next page URL
                    next_url = self.get_next_page()
                    
                    if next_url and next_url not in visited_urls:
                        current_url = next_url
                        page_number += 1
                        print(f"Moving to page {page_number}: {current_url}")
                        time.sleep(random.uniform(2, 4))
                    else:
                        print("Reached last page or detected loop")
                        current_url = None
                
                print(f"Finished processing URL: {base_url}")
                print(f"Total pages scraped: {page_number}")
            
        except Exception as e:
            print(f"Error in scrape_all: {str(e)}")
            self.driver.save_screenshot(os.path.join('data', 'raw', 'scraped_data', 'scrape_all_error.png'))

    def save_progress(self):
        """Save current progress to CSV"""
        if self.all_products:
            current_date = datetime.now().strftime('%Y%m%d')
            filename = f'climamarket_heatpumps_{current_date}.csv'
            
            df = pd.DataFrame(self.all_products)
            df.to_csv(os.path.join('data', 'raw', 'scraped_data', filename), index=False)
            print(f"Progress saved: {len(self.all_products)} products found")

    def close(self):
        """Close the browser"""
        self.driver.quit()

def main():
    """Main function to run the scraper"""
    try:
        scraper = ClimaMarketScraper()
        scraper.scrape_all()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
    finally:
        scraper.close()

if __name__ == "__main__":
    main() 