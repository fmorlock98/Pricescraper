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

class HeizungsdiscountScraper:
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
        self.base_url = "https://www.heizungsdiscount24.de/waermepumpen/"
        self.all_products = []

    def get_brand_links(self):
        """Get all brand links from the main page"""
        try:
            self.driver.get(self.base_url)
            time.sleep(random.uniform(2, 4))
            
            # Find the brands list
            brand_elements = self.driver.find_elements(By.CSS_SELECTOR, "li.os_prod_content_actp_1 ul li a")
            brand_links = []
            
            for element in brand_elements[:10]:  # First 10 are manufacturer brands
                href = element.get_attribute('href')
                name = element.text.strip()
                if href and name:
                    brand_links.append((name, href))
                    print(f"Found brand: {name}")
            
            return brand_links
            
        except Exception as e:
            print(f"Error getting brand links: {str(e)}")
            return []

    def get_model_links(self, brand_url):
        """Get all model links for a brand"""
        try:
            self.driver.get(brand_url)
            time.sleep(random.uniform(2, 4))
            
            # Find the models list
            model_elements = self.driver.find_elements(By.CSS_SELECTOR, "li.os_prod_content_actp_2 ul li a")
            model_links = []
            
            # Skip categories we don't want
            skip_categories = ['Zubehör', 'Zubehoer', 'Accessories', 'Ersatzteile', 'Spare Parts']
            
            for element in model_elements:
                name = element.text.strip()
                href = element.get_attribute('href')
                
                # Skip if the name matches any of our skip categories
                if any(category.lower() in name.lower() for category in skip_categories):
                    print(f"Skipping category: {name}")
                    continue
                
                if href and name:
                    model_links.append((name, href))
                    print(f"Found model: {name}")
            
            return model_links
            
        except Exception as e:
            print(f"Error getting model links: {str(e)}")
            return []

    def get_submodel_links(self, model_url):
        """Get all submodel links for a model"""
        try:
            self.driver.get(model_url)
            time.sleep(random.uniform(2, 4))
            
            # First check if there are direct products on this page
            try:
                product_container = self.driver.find_element(By.CSS_SELECTOR, "div#os_list_prod")
                if product_container:
                    # If we find products directly, return empty list to trigger direct product scraping
                    return []
            except:
                pass
            
            # Try to find submodels with different selectors
            submodel_elements = []
            selectors = [
                "li.os_prod_content_actp_3 ul li a",  # For deeper navigation
                "li.os_prod_content_actp_4 ul li a",  # Alternative navigation level
                "div.os_list_title a.os_list_link1"   # Direct product links
            ]
            
            for selector in selectors:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    submodel_elements.extend(elements)
                    print(f"Found elements with selector: {selector}")
                    break  # Use first successful selector
            
            submodel_links = []
            for element in submodel_elements:
                href = element.get_attribute('href')
                name = element.text.strip()
                if href and name:
                    submodel_links.append((name, href))
                    print(f"Found submodel: {name}")
            
            return submodel_links
            
        except Exception as e:
            print(f"Error getting submodel links: {str(e)}")
            return []

    def accept_cookies(self):
        """Accept cookies on the website"""
        try:
            print("Waiting for page to load completely...")
            time.sleep(5)  # Give page more time to load initially
            
            # First check for iframes that might contain cookie consent
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
            
            # Try multiple approaches to find and click cookie button
            try:
                # List of possible selectors for cookie buttons
                cookie_selectors = [
                    "button#uc-btn-accept-banner",  # UserCentrics
                    "button.consent-accept",
                    "button.accept-cookies",
                    "button.cookie-accept",
                    "#cookie-accept",
                    "#accept-cookies",
                    "button[contains(text(), 'Alle akzeptieren')]",
                    "button[contains(text(), 'Akzeptieren')]",
                    "button.cc-button",
                    "button.cc-accept-all",
                    ".cc-compliance .cc-allow"
                ]
                
                # Try CSS selectors first
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
                
                # Try XPath as fallback
                xpath_patterns = [
                    "//button[contains(., 'Alle akzeptieren')]",
                    "//button[contains(., 'Akzeptieren')]",
                    "//button[contains(., 'Accept all')]",
                    "//button[contains(., 'Accept')]",
                    "//a[contains(., 'Alle akzeptieren')]"
                ]
                
                for xpath in xpath_patterns:
                    try:
                        button = WebDriverWait(self.driver, 3).until(
                            EC.element_to_be_clickable((By.XPATH, xpath))
                        )
                        button.click()
                        print(f"Clicked cookie button with xpath: {xpath}")
                        time.sleep(2)
                        return True
                    except:
                        continue
                
                # Try JavaScript click as last resort
                js_scripts = [
                    "document.querySelector('.cc-compliance .cc-allow').click();",
                    "document.querySelector('#cookie-accept').click();",
                    "document.querySelector('.consent-accept').click();",
                    """
                    var buttons = document.getElementsByTagName('button');
                    for(var i = 0; i < buttons.length; i++) {
                        if(buttons[i].textContent.includes('akzeptieren') || 
                           buttons[i].textContent.includes('Akzeptieren') ||
                           buttons[i].textContent.includes('Accept')) {
                            buttons[i].click();
                            break;
                        }
                    }
                    """
                ]
                
                for script in js_scripts:
                    try:
                        self.driver.execute_script(script)
                        print("Clicked cookie button using JavaScript")
                        time.sleep(2)
                        return True
                    except:
                        continue
                
            except Exception as e:
                print(f"Error in cookie button click attempts: {str(e)}")
            
            # Switch back to default content if we were in an iframe
            try:
                self.driver.switch_to.default_content()
            except:
                pass
            
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
            time.sleep(random.uniform(3, 5))  # Increased initial wait time
            
            # Wait for any of these elements to be present
            wait_selectors = [
                "div.os_list_wrap_all",
                "div.os_list_title",
                "div.os_list_text",
                "div.os_list_price2"
            ]
            
            for selector in wait_selectors:
                try:
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    print(f"Found element with selector: {selector}")
                    break
                except:
                    continue
            
            # Save page source for debugging
            with open(os.path.join('data', 'raw', 'scraped_data', 'last_page.html'), 'w', encoding='utf-8') as f:
                f.write(self.driver.page_source)
            
            # Try different container selectors
            product_containers = []
            container_selectors = [
                "div.os_list_wrap_all",
                "div.os_content_corner",
                "div.col-lg-6.col-md-12.col-sm-12.col-xs-12"
            ]
            
            for selector in container_selectors:
                product_containers = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if product_containers:
                    print(f"Found {len(product_containers)} products with selector: {selector}")
                    break
            
            if not product_containers:
                print("No product containers found on page")
                self.driver.save_screenshot(os.path.join('data', 'raw', 'scraped_data', 'no_products.png'))
                return []
            
            products = []
            for container in product_containers:
                try:
                    # Extract manufacturer
                    try:
                        manufacturer_elem = container.find_element(By.CSS_SELECTOR, "div.os_list_text")
                        manufacturer = manufacturer_elem.text.replace("Hersteller:", "").split('\n')[0].strip()
                    except:
                        print("Could not find manufacturer, trying alternative method")
                        manufacturer = "Unknown"
                    
                    # Extract product type and article number
                    try:
                        type_elem = container.find_element(By.CSS_SELECTOR, "div.os_list_title a.os_list_link1, a.os_list_link1")
                        product_type = type_elem.text.strip()
                    except:
                        print("Could not find product type")
                        continue
                    
                    # Try to get article number
                    try:
                        art_num = manufacturer_elem.text.split('Art.Nr.:')[1].split('\n')[0].strip()
                        product_type = f"{product_type} ({art_num})"
                    except:
                        pass
                    
                    # Extract price
                    try:
                        price_elem = container.find_element(By.CSS_SELECTOR, "div.os_list_price2")
                        price_text = price_elem.text.strip()
                        
                        # Convert price to float
                        price_match = re.search(r'([\d.,]+)\s*EUR', price_text)
                        if price_match:
                            price = float(price_match.group(1).replace('.', '').replace(',', '.'))
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
                        'date_scraped': datetime.now().strftime('%Y-%m-%d')
                    }
                    
                    products.append(product)
                    print(f"Found product: {manufacturer} - {product_type} - {price} {currency}")
                    
                except Exception as e:
                    print(f"Error processing product container: {str(e)}")
                    continue
            
            return products
            
        except Exception as e:
            print(f"Error scraping products from page: {str(e)}")
            self.driver.save_screenshot(os.path.join('data', 'raw', 'scraped_data', f'error_page.png'))
            return []

    def scrape_all(self):
        """Scrape all products from all brands, models and submodels"""
        try:
            # Get all brand links
            self.driver.get(self.base_url)
            time.sleep(random.uniform(2, 4))
            
            # Try cookie acceptance multiple times
            max_cookie_attempts = 3
            for attempt in range(max_cookie_attempts):
                if self.accept_cookies():
                    break
                print(f"Cookie acceptance attempt {attempt + 1} failed, retrying...")
                time.sleep(2)
            
            brand_links = self.get_brand_links()
            
            for brand_name, brand_url in brand_links:
                print(f"\nProcessing brand: {brand_name}")
                
                # Get all model links for this brand
                model_links = self.get_model_links(brand_url)
                
                if not model_links:
                    # If no model links found, try to scrape products directly from brand page
                    print("No model links found, checking for direct products...")
                    products = self.scrape_products_from_page(brand_url)
                    self.all_products.extend(products)
                    self.save_progress()
                    continue
                
                for model_name, model_url in model_links:
                    print(f"\nProcessing model: {model_name}")
                    
                    # Get all submodel links for this model
                    submodel_links = self.get_submodel_links(model_url)
                    
                    if not submodel_links:
                        # If no submodel links found, try to scrape products directly from model page
                        print("No submodel links found, checking for direct products...")
                        products = self.scrape_products_from_page(model_url)
                        self.all_products.extend(products)
                        self.save_progress()
                        continue
                    
                    for submodel_name, submodel_url in submodel_links:
                        print(f"Processing submodel: {submodel_name}")
                        products = self.scrape_products_from_page(submodel_url)
                        self.all_products.extend(products)
                        
                        # Random delay between submodels
                        time.sleep(random.uniform(2, 4))
                    
                    # Save progress after each model
                    self.save_progress()
                
        except Exception as e:
            print(f"Error in scrape_all: {str(e)}")
            self.driver.save_screenshot(os.path.join('data', 'raw', 'scraped_data', 'scrape_all_error.png'))

    def save_progress(self):
        """Save current progress to CSV"""
        if self.all_products:
            current_date = datetime.now().strftime('%Y%m%d')
            filename = f'heizungsdiscount_heatpumps_{current_date}.csv'
            
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
        
        scraper = HeizungsdiscountScraper()
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