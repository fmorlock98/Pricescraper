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

class IdealoScraper:
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
        self.base_url = "https://www.idealo.de/preisvergleich/ProductCategory/18459.html"
        self.all_products = []
        self.current_page = 1
        self.total_pages = None

    def get_total_pages(self):
        """Get the total number of pages from the pagination"""
        try:
            # Find all page elements
            page_elements = self.driver.find_elements(By.CSS_SELECTOR, "div.sr-pagination__numbers_rff8h a.sr-pageElement_S1HzJ")
            
            # Get the last page element (excluding the "Next Page" button)
            last_page = None
            for element in page_elements:
                aria_label = element.get_attribute('aria-label')
                if aria_label and aria_label.isdigit():
                    last_page = int(aria_label)
            
            if last_page:
                self.total_pages = last_page
                print(f"Total pages to scrape: {self.total_pages}")
                return self.total_pages
            else:
                print("Could not find last page number")
                return None
                
        except Exception as e:
            print(f"Error getting total pages: {str(e)}")
            return None

    def scrape_products_from_page(self):
        """Scrape all products from the current page"""
        try:
            # Find all product items
            product_items = self.driver.find_elements(By.CSS_SELECTOR, "div.sr-resultList__item_m6xdA")
            
            for item in product_items:
                try:
                    # Get product title
                    title_elem = item.find_element(By.CSS_SELECTOR, "div.sr-productSummary__title_f5flP")
                    title = title_elem.text.strip()
                    
                    # Extract manufacturer (first word of title)
                    manufacturer = title.split()[0]
                    
                    # Get price
                    price_elem = item.find_element(By.CSS_SELECTOR, "div.sr-detailedPriceInfo__price_sYVmx")
                    price_text = price_elem.text.strip()
                    
                    # Convert price to float (remove "ab" prefix and convert German number format)
                    price_match = re.search(r'(\d+[.,]\d+)', price_text)
                    if price_match:
                        price = float(price_match.group(1).replace('.', '').replace(',', '.'))
                        currency = 'EUR'
                    else:
                        price = None
                        currency = None
                    
                    product = {
                        'manufacturer': manufacturer,
                        'type': title,
                        'price': price,
                        'currency': currency,
                        'page': self.current_page,
                        'date_scraped': datetime.now().strftime('%Y-%m-%d')
                    }
                    
                    self.all_products.append(product)
                    print(f"Found product: {title} - {price} {currency}")
                    
                except Exception as e:
                    print(f"Error processing product item: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error scraping products from page: {str(e)}")

    def go_to_next_page(self):
        """Navigate to the next page"""
        try:
            # Find the next page button
            next_button = self.driver.find_element(By.CSS_SELECTOR, "a.sr-pageArrow_HufQY[aria-label='Nächste Seite']")
            
            # Get the href attribute
            next_page_url = next_button.get_attribute('href')
            
            if next_page_url:
                # Navigate directly to the next page URL
                self.driver.get(next_page_url)
                time.sleep(random.uniform(2, 4))
                return True
            else:
                print("Could not find next page URL")
                return False
                
        except Exception as e:
            print(f"Error navigating to next page: {str(e)}")
            return False

    def save_progress(self):
        """Save current progress to CSV"""
        if self.all_products:
            current_date = datetime.now().strftime('%Y%m%d')
            filename = f'idealo_heatpumps_{current_date}.csv'
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.join('data', 'raw', 'scraped_data'), exist_ok=True)
            
            df = pd.DataFrame(self.all_products)
            df.to_csv(os.path.join('data', 'raw', 'scraped_data', filename), index=False)
            print(f"Progress saved: {len(self.all_products)} products found")

    def scrape_all(self):
        """Scrape all products from all pages"""
        try:
            # Navigate to main page
            self.driver.get(self.base_url)
            time.sleep(random.uniform(2, 4))
            
            # Get total number of pages
            self.get_total_pages()
            
            if not self.total_pages:
                print("Could not determine total pages. Exiting.")
                return
            
            # Scrape each page
            while self.current_page <= self.total_pages:
                print(f"\nProcessing page {self.current_page} of {self.total_pages}")
                
                # Scrape products from current page
                self.scrape_products_from_page()
                
                # Save progress after each page
                self.save_progress()
                
                # Go to next page if not on last page
                if self.current_page < self.total_pages:
                    if not self.go_to_next_page():
                        print("Failed to navigate to next page. Exiting.")
                        break
                
                # Increment page counter
                self.current_page += 1
                
                # Random delay between pages
                time.sleep(random.uniform(2, 4))
            
        except Exception as e:
            print(f"Error in scrape_all: {str(e)}")
            self.driver.save_screenshot(os.path.join('data', 'raw', 'scraped_data', 'scrape_all_error.png'))

    def close(self):
        """Close the browser"""
        self.driver.quit()

def main():
    """Main function to run the scraper"""
    try:
        scraper = IdealoScraper()
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