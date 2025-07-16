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
import json

class ClimamarketITScraper:
    def __init__(self):
        self.options = Options()
        self.options.add_argument('--headless')
        self.options.add_argument('--disable-gpu')
        self.options.add_argument('--no-sandbox')
        self.options.add_argument('--disable-dev-shm-usage')
        self.options.add_argument('--lang=it-IT')
        
        # Add random user agent
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
        ]
        self.options.add_argument(f'user-agent={random.choice(user_agents)}')
        
        self.service = Service()
        self.driver = webdriver.Chrome(service=self.service, options=self.options)
        self.base_url = "https://www.climamarket.it/pompe-di-calore?order_by=current_price&per_page=24"
        self.all_products = []

    def accept_cookies(self):
        """Accept cookies on the website"""
        try:
            print("Waiting for page to load completely...")
            time.sleep(5)
            
            # Try multiple approaches to find and click cookie button
            cookie_selectors = [
                "button#acceptCookies",
                "button.accept-cookies",
                "#cookie-accept",
                "button[contains(text(), 'Accetta')]",
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
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.productGridView"))
            )
            
            # Find all product containers
            product_containers = self.driver.find_elements(By.CSS_SELECTOR, "div.productGridView")
            
            if not product_containers:
                print("No product containers found on page")
                return []
            
            products = []
            for container in product_containers:
                try:
                    # Extract manufacturer
                    try:
                        brand_elem = container.find_element(By.CSS_SELECTOR, "div.productBrand a")
                        manufacturer = brand_elem.text.strip()
                    except:
                        print("Could not find manufacturer")
                        continue
                    
                    # Extract product type from data-items attribute
                    try:
                        type_elem = container.find_element(By.CSS_SELECTOR, "a.jsSelectableItem")
                        data_items = type_elem.get_attribute('data-items')
                        items_data = json.loads(data_items)
                        product_type = items_data[0]['item_name']
                    except:
                        print("Could not find product type")
                        continue
                    
                    # Extract price
                    try:
                        # Find the price container
                        price_container = container.find_element(By.CSS_SELECTOR, "div.wrapperPriceListing")
                        
                        # Initialize variables to store the lowest price
                        lowest_price = float('inf')
                        lowest_price_text = None
                        lowest_decimal = None
                        
                        # Check flyerPrice (special offer price)
                        try:
                            flyer_price = price_container.find_element(By.CSS_SELECTOR, "div.flyerPrice p:not(.spLabel)")
                            main_price = flyer_price.text.replace('€', '').strip()
                            decimal_part = flyer_price.find_element(By.CSS_SELECTOR, "span.apice").text.strip()
                            
                            # Remove the decimal part from main price
                            main_price = main_price.split(',')[0].strip()
                            
                            # Calculate full price
                            full_price = float(f"{main_price.replace('.', '')}.{decimal_part}")
                            
                            if full_price < lowest_price:
                                lowest_price = full_price
                                lowest_price_text = main_price
                                lowest_decimal = decimal_part
                        except Exception as e:
                            print(f"No flyerPrice found: {str(e)}")
                        
                        # Check priceBig sconto
                        try:
                            sconto_price = price_container.find_element(By.CSS_SELECTOR, "p.priceBig.sconto")
                            main_price = sconto_price.text.replace('€', '').strip()
                            decimal_part = sconto_price.find_element(By.CSS_SELECTOR, "span.apice").text.strip()
                            
                            # Remove the decimal part from main price
                            main_price = main_price.split(',')[0].strip()
                            
                            # Calculate full price
                            full_price = float(f"{main_price.replace('.', '')}.{decimal_part}")
                            
                            if full_price < lowest_price:
                                lowest_price = full_price
                                lowest_price_text = main_price
                                lowest_decimal = decimal_part
                        except Exception as e:
                            print(f"No priceBig sconto found: {str(e)}")
                        
                        if lowest_price == float('inf'):
                            print("Could not find any valid price")
                            continue
                            
                        price = lowest_price
                        currency = 'EUR'
                        print(f"Found lowest price: {price} EUR (main: {lowest_price_text}, decimal: {lowest_decimal})")
                            
                    except Exception as e:
                        print(f"Error extracting price: {str(e)}")
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

    def get_total_pages(self):
        """Get the total number of pages"""
        try:
            # Wait for pagination element to be present
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div#wrapperPaginazione span"))
            )
            
            pagination_text = self.driver.find_element(By.CSS_SELECTOR, "div#wrapperPaginazione span").text
            print(f"Found pagination text: {pagination_text}")
            
            # Extract the last number from the text
            numbers = [int(s) for s in re.findall(r'\d+', pagination_text)]
            if numbers:
                total_pages = numbers[-1]  # Get the last number
                print(f"Total pages found: {total_pages}")
                return total_pages
            else:
                print("Could not extract total pages number from text")
                return 1
        except Exception as e:
            print(f"Error getting total pages: {str(e)}")
            return 1

    def get_next_page_url(self, current_page):
        """Construct the URL for the next page"""
        return f"https://www.climamarket.it/pompe-di-calore?order_by=current_price&page={current_page}&per_page=24"

    def scrape_all(self):
        """Scrape all products from all pages"""
        try:
            # Start with the first page
            print(f"Starting with URL: {self.base_url}")
            self.driver.get(self.base_url)
            time.sleep(random.uniform(2, 4))
            
            # Try cookie acceptance
            self.accept_cookies()
            
            # Get total number of pages
            total_pages = self.get_total_pages()
            print(f"Found {total_pages} pages to scrape")
            
            current_page = 1
            while current_page <= total_pages:
                print(f"\nProcessing page {current_page} of {total_pages}")
                
                # Wait for page to load
                try:
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "div.productGridView"))
                    )
                except Exception as e:
                    print(f"Error waiting for page {current_page} to load: {str(e)}")
                    # Try reloading the page
                    self.driver.refresh()
                    time.sleep(5)
                
                # Scrape current page
                products = self.scrape_products_from_page(current_page)
                if products:
                    self.all_products.extend(products)
                    print(f"Found {len(products)} products on page {current_page}")
                else:
                    print(f"No products found on page {current_page}, retrying...")
                    # Try to reload the page once
                    self.driver.refresh()
                    time.sleep(5)
                    products = self.scrape_products_from_page(current_page)
                    if products:
                        self.all_products.extend(products)
                        print(f"After retry: Found {len(products)} products on page {current_page}")
                
                # Save progress after each page
                self.save_progress()
                
                # Go to next page if not on last page
                if current_page < total_pages:
                    next_url = self.get_next_page_url(current_page + 1)
                    print(f"Going to next page: {next_url}")
                    self.driver.get(next_url)
                    time.sleep(random.uniform(4, 6))  # Increased delay between pages
                
                current_page += 1
                
        except Exception as e:
            print(f"Error in scrape_all: {str(e)}")
            # Save progress even if error occurs
            self.save_progress()

    def save_progress(self):
        """Save current progress to CSV"""
        if self.all_products:
            current_date = datetime.now().strftime('%Y%m%d')
            filename = f'climamarketIT_heatpumps_{current_date}.csv'
            
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
        scraper = ClimamarketITScraper()
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