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

class SelfioScraper:
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
        self.base_url = "https://www.selfio.de/heizung/waermeerzeuger/waermepumpe/?o=8&n=168"
        self.all_products = []

    def accept_cookies(self):
        """Accept cookies on the website"""
        try:
            time.sleep(3)
            # Add different selectors for cookie acceptance button
            cookie_selectors = [
                "button.cookie-permission--accept-button",
                "button[data-cookieman-save]",
                ".cookie-consent--button button"
            ]
            
            for selector in cookie_selectors:
                try:
                    button = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    button.click()
                    print("Accepted cookies")
                    time.sleep(2)
                    return
                except:
                    continue
                    
            print("No cookie banner found or already accepted")
        except:
            print("Error accepting cookies")

    def scrape_products_from_page(self, url):
        """Scrape all products from a page"""
        try:
            products = []
            print(f"Scraping products from: {url}")
            
            # Wait for product grid to load
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.product--box.box--minimal"))
            )
            
            # Additional wait for dynamic content
            time.sleep(3)
            
            # Get all product containers
            product_containers = self.driver.find_elements(By.CSS_SELECTOR, "div.product--box.box--minimal")
            print(f"Found {len(product_containers)} products on page")
            
            for container in product_containers:
                try:
                    # Get product title
                    title_elem = container.find_element(By.CSS_SELECTOR, "a.product--title")
                    title = title_elem.get_attribute('title').strip()
                    
                    # Extract manufacturer (first word) and type
                    manufacturer = title.split()[0]
                    product_type = title
                    
                    # Get price
                    try:
                        price_elem = container.find_element(By.CSS_SELECTOR, "span.price--default.is--nowrap")
                        price_text = price_elem.text.strip()
                        # Remove 'ab' if present and extract price
                        price_text = price_text.replace('ab', '').strip()
                        price_match = re.search(r'([\d.,]+)', price_text)
                        
                        if price_match:
                            price = float(price_match.group(1).replace('.', '').replace(',', '.'))
                            
                            product = {
                                'manufacturer': manufacturer,
                                'type': product_type,
                                'price': price,
                                'currency': 'EUR',
                                'page': url,
                                'date_scraped': datetime.now().strftime('%Y-%m-%d')
                            }
                            
                            products.append(product)
                            print(f"Found product: {manufacturer} - {product_type} - €{price}")
                            
                    except Exception as e:
                        print(f"Error extracting price: {str(e)}")
                        continue
                        
                except Exception as e:
                    print(f"Error processing product: {str(e)}")
                    continue
            
            return products
            
        except Exception as e:
            print(f"Error scraping products: {str(e)}")
            self.driver.save_screenshot(os.path.join('data', 'raw', 'scraped_data', 'scrape_error.png'))
            return []

    def get_next_page(self):
        """Get the URL of the next page"""
        try:
            next_link = self.driver.find_element(
                By.CSS_SELECTOR, 
                "a.paging--next[title='Nächste Seite']"
            )
            return next_link.get_attribute('href')
        except:
            print("No next page found")
            return None

    def scrape_all(self):
        """Main scraping function"""
        try:
            # Create scraped_data directory if it doesn't exist
            os.makedirs(os.path.join('data', 'raw', 'scraped_data'), exist_ok=True)
            
            # Start with base URL and ensure 168 items per page
            current_url = "https://www.selfio.de/heizung/waermeerzeuger/waermepumpe/"
            page_number = 1
            total_pages = 3  # We know there are exactly 3 pages
            
            while page_number <= total_pages:
                print(f"\nScraping page {page_number} of {total_pages}")
                
                # Construct page URL
                if page_number == 1:
                    page_url = f"{current_url}?n=168"
                else:
                    page_url = f"{current_url}?p={page_number}&n=168"
                    
                self.driver.get(page_url)
                time.sleep(random.uniform(2, 4))
                
                # Accept cookies on first page
                if page_number == 1:
                    self.accept_cookies()
                    time.sleep(2)
                
                # Scrape products from current page
                products = self.scrape_products_from_page(page_url)
                if products:
                    self.all_products.extend(products)
                    # Save progress after each page
                    self.save_progress()
                
                # Move to next page
                if page_number < total_pages:
                    page_number += 1
                    print(f"Moving to page {page_number}")
                    time.sleep(random.uniform(2, 4))
                else:
                    print("Reached last page")
                    break
            
            print(f"Finished scraping. Total pages: {page_number}")
            print(f"Total products found: {len(self.all_products)}")
            
        except Exception as e:
            print(f"Error in scrape_all: {str(e)}")
            self.driver.save_screenshot(os.path.join('data', 'raw', 'scraped_data', 'scrape_all_error.png'))

    def save_progress(self):
        """Save current progress to CSV"""
        if self.all_products:
            current_date = datetime.now().strftime('%Y%m%d')
            filename = f'selfio_heatpumps_{current_date}.csv'
            
            df = pd.DataFrame(self.all_products)
            df.to_csv(os.path.join('data', 'raw', 'scraped_data', filename), index=False)
            print(f"Progress saved: {len(self.all_products)} products found")

    def close(self):
        """Close the browser"""
        self.driver.quit()

def main():
    """Main function to run the scraper"""
    try:
        scraper = SelfioScraper()
        scraper.scrape_all()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
    finally:
        scraper.close()

if __name__ == "__main__":
    main() 