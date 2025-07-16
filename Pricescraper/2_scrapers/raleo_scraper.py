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

class RaleoScraper:
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
        
        self.service = Service()
        self.driver = webdriver.Chrome(service=self.service, options=self.options)
        self.base_url = "https://raleo.de/category/Heizung/Wärmepumpen?l=60"

    def accept_cookies(self):
        """Accept cookies on the website"""
        try:
            print("Waiting for page to load completely...")
            time.sleep(5)  # Give page more time to load initially
            
            # First check if we need to switch to an iframe
            try:
                iframes = self.driver.find_elements(By.TAG_NAME, "iframe")
                for iframe in iframes:
                    try:
                        self.driver.switch_to.frame(iframe)
                        print("Switched to iframe")
                    except:
                        continue
            except:
                print("No iframes found")
            
            # Save page source and screenshot for debugging
            with open(os.path.join('data', 'raw', 'scraped_data', 'before_cookies.html'), 'w', encoding='utf-8') as f:
                f.write(self.driver.page_source)
            self.driver.save_screenshot(os.path.join('data', 'raw', 'scraped_data', 'before_cookies.png'))
            
            # Try to find and click the button
            try:
                # Try JavaScript click
                button = self.driver.execute_script("""
                    return document.querySelector('button.v-btn.v-btn--text.theme--light.v-size--default.primary--text') ||
                           document.evaluate("//button[contains(text(), 'Alle akzeptieren')]", 
                                          document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                """)
                if button:
                    self.driver.execute_script("arguments[0].click();", button)
                    print("Clicked cookie button using JavaScript")
                    time.sleep(2)
                    return True
            except Exception as e:
                print(f"JavaScript click failed: {str(e)}")
            
            # Switch back to default content if we were in an iframe
            self.driver.switch_to.default_content()
            
            print("Could not find cookie accept button, trying to continue anyway...")
            return False
            
        except Exception as e:
            print(f"Error in cookie acceptance: {str(e)}")
            self.driver.save_screenshot(os.path.join('data', 'raw', 'scraped_data', 'cookie_error.png'))
            return False

    def scrape_page(self, page_number):
        """Scrape a specific page"""
        try:
            # Update URL construction to include the p parameter for pagination
            url = f"{self.base_url}&p={page_number}" if page_number > 1 else self.base_url
            print(f"\nScraping page {page_number}")
            
            # Random delay between requests (2-5 seconds)
            time.sleep(random.uniform(2, 5))
            
            self.driver.get(url)
            
            # Accept cookies on first page
            if page_number == 1:
                self.accept_cookies()
            
            # Wait for products to load
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.grid-items"))
            )
            
            products = []
            
            # Find all product cards within the grid-items container
            product_elements = self.driver.find_elements(By.CSS_SELECTOR, "div.grid-items > a.grid-card")
            
            for element in product_elements:
                try:
                    # Extract manufacturer
                    manufacturer_elem = element.find_element(By.CSS_SELECTOR, "div.brand.reactive")
                    manufacturer = manufacturer_elem.text.replace("Hersteller:", "").strip()
                    
                    # Extract model/type - make sure we get the full text content
                    type_elem = element.find_element(By.CSS_SELECTOR, "p.word_break.description_short.reactive")
                    product_type = type_elem.get_attribute('textContent').strip()
                    
                    # Extract price
                    price_elem = element.find_element(By.CSS_SELECTOR, "span.text-h5.primary--text")
                    price_text = price_elem.text.strip()
                    
                    # Convert price to float
                    price_match = re.search(r'([\d.,]+)\s*€', price_text)
                    if price_match:
                        price = float(price_match.group(1).replace('.', '').replace(',', '.'))
                        currency = '€'
                    else:
                        price = None
                        currency = None
                    
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
            
            print(f"Found {len(products)} products on this page")
            return products
            
        except Exception as e:
            print(f"Error scraping page {page_number}: {str(e)}")
            self.driver.save_screenshot(os.path.join('data', 'raw', 'scraped_data', f'error_page_{page_number}.png'))
            return []

    def close(self):
        """Close the browser"""
        self.driver.quit()

def main():
    """Main function to run the scraper"""
    try:
        scraper = RaleoScraper()
        all_products = []
        current_date = datetime.now().strftime('%Y%m%d')
        filename = f'raleo_heatpumps_{current_date}.csv'
        
        # Create scraped_data directory if it doesn't exist
        os.makedirs(os.path.join('data', 'raw', 'scraped_data'), exist_ok=True)
        
        page = 1
        while True:
            products = scraper.scrape_page(page)
            if not products:  # If no products found, we've reached the end
                break
                
            all_products.extend(products)
            
            # Save progress after each page
            if all_products:
                df = pd.DataFrame(all_products)
                df.to_csv(os.path.join('data', 'raw', 'scraped_data', filename), index=False)
                print(f"Progress saved: {len(all_products)} products found")
            
            # Random longer delay between pages (5-10 seconds)
            time.sleep(random.uniform(5, 10))
            page += 1
        
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
            error_filename = f'raleo_heatpumps_{current_date}_partial.csv'
            df = pd.DataFrame(all_products)
            df.to_csv(os.path.join('data', 'raw', 'scraped_data', error_filename), index=False)
            print(f"Saved {len(all_products)} products to data/raw/scraped_data/{error_filename}")
    
    finally:
        scraper.close()

if __name__ == "__main__":
    main() 