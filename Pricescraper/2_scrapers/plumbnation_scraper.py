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

class PlumbnationScraper:
    def __init__(self):
        self.options = Options()
        self.options.add_argument('--headless')
        self.options.add_argument('--disable-gpu')
        self.options.add_argument('--no-sandbox')
        self.options.add_argument('--disable-dev-shm-usage')
        self.options.add_argument('--lang=en-GB')
        
        # Add random user agent
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
        ]
        self.options.add_argument(f'user-agent={random.choice(user_agents)}')
        
        self.service = Service()
        self.driver = webdriver.Chrome(service=self.service, options=self.options)
        self.base_url = "https://www.plumbnation.co.uk/air-source-heat-pumps-111-0000"
        self.all_products = []

    def accept_cookies(self):
        """Accept cookies on the website"""
        try:
            print("Waiting for page to load completely...")
            time.sleep(5)
            
            # Try multiple approaches to find and click cookie button
            cookie_selectors = [
                "button.accept-cookies",
                "#cookie-accept",
                "button[contains(text(), 'Accept and Continue')]",
                "button[contains(text(), 'Accept')]",
                ".cc-compliance .cc-allow"
            ]
            
            for selector in cookie_selectors:
                try:
                    button = WebDriverWait(self.driver, 3).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    button.click()
                    print(f"Clicked cookie button with selector: {selector}")
                    time.sleep(2)
                    return True
                except:
                    continue
            
            print("Could not find cookie accept button, trying to continue anyway...")
            return False
            
        except Exception as e:
            print(f"Error in cookie acceptance: {str(e)}")
            self.driver.save_screenshot(os.path.join('data', 'raw', 'scraped_data', 'cookie_error.png'))
            return False

    def scrape_products_from_page(self, url):
        """Scrape all products from a page"""
        try:
            self.driver.get(url)
            time.sleep(random.uniform(3, 5))
            
            # Wait for product containers
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.product-card"))
            )
            
            # Find all product containers
            product_containers = self.driver.find_elements(By.CSS_SELECTOR, "div.product-card")
            
            products = []
            for container in product_containers:
                try:
                    # Get the data-analytics-object attribute
                    analytics_data = container.get_attribute('data-analytics-object')
                    if analytics_data:
                        # Parse the JSON-like string
                        analytics_data = analytics_data.replace('&quot;', '"')
                        import json
                        data = json.loads(analytics_data)
                        
                        # Extract manufacturer and type from analytics data
                        manufacturer = data.get('item_brand', 'Unknown')
                        product_type = data.get('item_name', '')
                        
                        # Find price
                        try:
                            price_elem = container.find_element(By.CSS_SELECTOR, "span.amount")
                            price_text = price_elem.text.split('\n')[0].strip()  # Get first line before VAT
                            
                            # Convert price to float
                            price_match = re.search(r'£([\d.,]+)', price_text)
                            if price_match:
                                price = float(price_match.group(1).replace(',', ''))
                                currency = 'GBP'
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
                            'date_scraped': datetime.now().strftime('%Y-%m-%d')
                        }
                        
                        products.append(product)
                        print(f"Found product: {manufacturer} - {product_type} - {price} {currency}")
                
                except Exception as e:
                    print(f"Error processing product: {str(e)}")
                    continue
            
            return products
            
        except Exception as e:
            print(f"Error scraping products from page: {str(e)}")
            self.driver.save_screenshot(os.path.join('data', 'raw', 'scraped_data', 'error_page.png'))
            return []

    def scrape_all(self):
        """Scrape all products"""
        try:
            self.driver.get(self.base_url)
            time.sleep(random.uniform(2, 4))
            
            # Accept cookies
            self.accept_cookies()
            
            # Get all page URLs
            try:
                pagination = self.driver.find_elements(By.CSS_SELECTOR, "div.pagenumbers a")
                page_urls = [self.base_url]  # First page is base URL
                
                for page_link in pagination:
                    href = page_link.get_attribute('href')
                    if href and href not in page_urls:
                        page_urls.append(href)
                
                print(f"Found {len(page_urls)} pages to scrape")
                
            except Exception as e:
                print(f"Error getting pagination: {str(e)}")
                page_urls = [self.base_url]  # Fall back to just first page
            
            # Scrape each page
            for i, url in enumerate(page_urls, 1):
                print(f"\nProcessing page {i} of {len(page_urls)}")
                
                products = self.scrape_products_from_page(url)
                self.all_products.extend(products)
                
                # Save progress after each page
                self.save_progress()
                
                # Random delay between pages
                if i < len(page_urls):  # Don't wait after last page
                    time.sleep(random.uniform(2, 4))
                
        except Exception as e:
            print(f"Error in scrape_all: {str(e)}")
            self.driver.save_screenshot(os.path.join('data', 'raw', 'scraped_data', 'scrape_all_error.png'))

    def save_progress(self):
        """Save current progress to CSV"""
        if self.all_products:
            current_date = datetime.now().strftime('%Y%m%d')
            filename = f'plumbnation_heatpumps_{current_date}.csv'
            
            df = pd.DataFrame(self.all_products)
            df.to_csv(os.path.join('data', 'raw', 'scraped_data', filename), index=False)
            print(f"Progress saved: {len(self.all_products)} products found")

    def close(self):
        """Close the browser"""
        self.driver.quit()

def main():
    """Main function to run the scraper"""
    try:
        # Create scraped_data directory if it doesn't exist
        os.makedirs(os.path.join('data', 'raw', 'scraped_data'), exist_ok=True)
        
        scraper = PlumbnationScraper()
        scraper.scrape_all()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
    finally:
        scraper.close()

if __name__ == "__main__":
    main() 