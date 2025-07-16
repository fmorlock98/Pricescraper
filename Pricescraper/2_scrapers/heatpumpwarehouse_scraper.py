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

class HeatPumpWarehouseScraper:
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
        self.base_url = "https://www.theheatpumpwarehouse.co.uk/product-category/heat-pumps/air-source-heat-pumps/"
        self.all_products = []

    def accept_cookies(self):
        """Accept cookies on the website"""
        try:
            print("Waiting for cookie consent...")
            time.sleep(5)
            
            cookie_selectors = [
                "#cookie_action_close_header",
                ".cli-accept-all-btn",
                ".accept-cookies",
                "#cookie-law-info-bar button"
            ]
            
            for selector in cookie_selectors:
                try:
                    button = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    button.click()
                    print("Accepted cookies")
                    time.sleep(2)
                    return True
                except:
                    continue
            
            return False
            
        except Exception as e:
            print(f"Error accepting cookies: {str(e)}")
            return False

    def get_manufacturer_links(self):
        """Get all manufacturer links from the main page"""
        try:
            print("Getting manufacturer links...")
            
            # Define target manufacturers
            target_manufacturers = [
                'Ebac',
                'Trianco Activair',
                'CTC',
                'BAXI',
                'Hisense',
                'Hitachi',
                'Samsung',
                'Vaillant',
                'LG',
                'Panasonic',
                'Mitsubishi'
            ]
            
            # Wait for the category list to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "li.cat-item.current-cat.cat-parent"))
            )
            
            # Find the air source heat pumps category container
            air_source_category = self.driver.find_element(
                By.CSS_SELECTOR, 
                "li.cat-item.current-cat.cat-parent"
            )
            
            # Get all manufacturer list items within this category
            manufacturers = air_source_category.find_elements(
                By.CSS_SELECTOR, 
                "ul.children > li.cat-item"
            )
            
            manufacturer_links = []
            for manufacturer in manufacturers:
                try:
                    # Get manufacturer name and link
                    link = manufacturer.find_element(By.CSS_SELECTOR, "a")
                    name = link.text.strip()
                    
                    # Only process if it's in our target list
                    if name in target_manufacturers:
                        url = link.get_attribute('href')
                        has_submodels = 'cat-parent' in manufacturer.get_attribute('class')
                        
                        manufacturer_links.append({
                            'name': name,
                            'url': url,
                            'has_submodels': has_submodels
                        })
                        print(f"Found manufacturer: {name} {'(with submodels)' if has_submodels else ''}")
                    
                except Exception as e:
                    print(f"Error processing manufacturer: {str(e)}")
                    continue
            
            return manufacturer_links
            
        except Exception as e:
            print(f"Error getting manufacturer links: {str(e)}")
            self.driver.save_screenshot(os.path.join('data', 'raw', 'scraped_data', 'manufacturer_error.png'))
            return []

    def get_submodel_links(self, manufacturer_url):
        """Get submodel links for a manufacturer"""
        try:
            self.driver.get(manufacturer_url)
            time.sleep(random.uniform(2, 4))
            
            # Wait for submodel list to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "ul.children"))
            )
            
            # Find the submodels container
            submodels_container = self.driver.find_element(By.CSS_SELECTOR, "ul.children")
            
            # Get all submodel links
            submodels = submodels_container.find_elements(
                By.CSS_SELECTOR, 
                "li.cat-item > a"
            )
            
            submodel_links = []
            for submodel in submodels:
                name = submodel.text.strip()
                url = submodel.get_attribute('href')
                if name and url:
                    submodel_links.append({
                        'name': name,
                        'url': url
                    })
                    print(f"Found submodel: {name}")
            
            return submodel_links
            
        except Exception as e:
            print(f"Error getting submodel links: {str(e)}")
            self.driver.save_screenshot(os.path.join('data', 'raw', 'scraped_data', 'submodel_error.png'))
            return []

    def scrape_products_from_page(self, url, manufacturer, submodel=None):
        """Scrape all products from a page"""
        try:
            self.driver.get(url)
            time.sleep(random.uniform(3, 5))
            
            # Wait for product containers
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "li.product-col"))
            )
            
            products = []
            product_containers = self.driver.find_elements(By.CSS_SELECTOR, "li.product-col")
            
            for container in product_containers:
                try:
                    # Get product name
                    name_elem = container.find_element(By.CSS_SELECTOR, "h3.woocommerce-loop-product__title")
                    product_type = name_elem.text.strip()
                    
                    # Get price (excluding VAT)
                    price_elem = container.find_element(By.CSS_SELECTOR, "div.inc-vat span.woocommerce-Price-amount")
                    price_text = price_elem.text.strip()
                    price_match = re.search(r'£([\d.,]+)', price_text)
                    
                    if price_match:
                        price = float(price_match.group(1).replace(',', ''))
                        
                        product = {
                            'manufacturer': manufacturer,
                            'type': product_type,
                            'price': price,
                            'currency': 'GBP',
                            'page': url,
                            'date_scraped': datetime.now().strftime('%Y-%m-%d')
                        }
                        
                        products.append(product)
                        print(f"Found product: {manufacturer} - {product_type} - £{price}")
                    
                except Exception as e:
                    print(f"Error processing product: {str(e)}")
                    continue
                
            return products
            
        except Exception as e:
            print(f"Error scraping products: {str(e)}")
            self.driver.save_screenshot(os.path.join('data', 'raw', 'scraped_data', 'scrape_error.png'))
            return []

    def scrape_all(self):
        """Main scraping function"""
        try:
            self.driver.get(self.base_url)
            time.sleep(random.uniform(2, 4))
            
            # Accept cookies
            self.accept_cookies()
            time.sleep(2)
            
            # Get all manufacturer links
            manufacturers = self.get_manufacturer_links()
            
            # Process each manufacturer
            for manufacturer in manufacturers:
                print(f"\nProcessing manufacturer: {manufacturer['name']}")
                
                if manufacturer['has_submodels']:
                    # Get and process submodels
                    submodels = self.get_submodel_links(manufacturer['url'])
                    for submodel in submodels:
                        print(f"Processing submodel: {submodel['name']}")
                        products = self.scrape_products_from_page(
                            submodel['url'],
                            manufacturer['name'],
                            submodel['name']
                        )
                        self.all_products.extend(products)
                        self.save_progress()
                        time.sleep(random.uniform(2, 4))
                else:
                    # Process manufacturer directly
                    products = self.scrape_products_from_page(
                        manufacturer['url'],
                        manufacturer['name']
                    )
                    self.all_products.extend(products)
                    self.save_progress()
                    time.sleep(random.uniform(2, 4))
                
        except Exception as e:
            print(f"Error in scrape_all: {str(e)}")
            self.driver.save_screenshot(os.path.join('data', 'raw', 'scraped_data', 'scrape_all_error.png'))

    def save_progress(self):
        """Save current progress to CSV"""
        if self.all_products:
            current_date = datetime.now().strftime('%Y%m%d')
            filename = f'heatpumpwarehouse_heatpumps_{current_date}.csv'
            
            df = pd.DataFrame(self.all_products)
            df.to_csv(os.path.join('data', 'raw', 'scraped_data', filename), index=False)
            print(f"Progress saved: {len(self.all_products)} products found")

    def close(self):
        """Close the browser"""
        self.driver.quit()

def main():
    """Main function to run the scraper"""
    try:
        os.makedirs(os.path.join('data', 'raw', 'scraped_data'), exist_ok=True)
        
        scraper = HeatPumpWarehouseScraper()
        scraper.scrape_all()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
    finally:
        scraper.close()

if __name__ == "__main__":
    main() 