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

class Breeze24Scraper:
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
        self.base_url = "https://www.breeze24.com/waermepumpen"
        self.all_products = []

    def get_category_links(self):
        """Get all category links from the main page"""
        try:
            self.driver.get(self.base_url)
            time.sleep(random.uniform(2, 4))
            
            # Find the Wärmepumpen category specifically
            category_elements = self.driver.find_elements(By.CSS_SELECTOR, "ul.sidebar--navigation.categories--navigation li.navigation--entry.is--active.has--sub-categories.has--sub-children a.navigation--link")
            category_links = []
            
            for element in category_elements:
                href = element.get_attribute('href')
                name = element.text.strip()
                if href and name and "Wärmepumpen" in name:
                    category_links.append((name, href))
                    print(f"Found category: {name}")
                    break  # We only want the first matching category
            
            return category_links
            
        except Exception as e:
            print(f"Error getting category links: {str(e)}")
            return []

    def get_subcategory_links(self, category_url, level=1):
        """Get all subcategory links for a category, handling multiple levels of navigation"""
        try:
            self.driver.get(category_url)
            time.sleep(random.uniform(2, 4))
            
            subcategory_links = []
            
            # List of sections to exclude
            excluded_sections = [
                "Schwimmbad",
                "Solarthermie",
                "Speicher",
                "Hybrid-WP / Gas",
                "Multi+ Wärmepumpen"
            ]
            
            # Find all navigation lists at the current level
            nav_lists = self.driver.find_elements(By.CSS_SELECTOR, f"ul.sidebar--navigation.categories--navigation.navigation--list.is--level{level}")
            
            for nav_list in nav_lists:
                # Find all navigation entries in this list
                entries = nav_list.find_elements(By.CSS_SELECTOR, "li.navigation--entry")
                
                for entry in entries:
                    try:
                        # Check if this entry has sub-categories
                        has_sub_categories = "has--sub-categories" in entry.get_attribute("class")
                        has_sub_children = "has--sub-children" in entry.get_attribute("class")
                        
                        # Get the link element
                        link = entry.find_element(By.CSS_SELECTOR, "a.navigation--link")
                        name = link.text.strip()
                        href = link.get_attribute('href')
                        
                        # Skip excluded sections
                        if any(excluded in name for excluded in excluded_sections):
                            print(f"Skipping excluded section: {name}")
                            continue
                        
                        if href and name:
                            subcategory_links.append((name, href, has_sub_categories or has_sub_children))
                            print(f"Found {'subcategory' if not (has_sub_categories or has_sub_children) else 'category with subcategories'}: {name}")
                    except Exception as e:
                        print(f"Error processing navigation entry: {str(e)}")
                        continue
            
            return subcategory_links
            
        except Exception as e:
            print(f"Error getting subcategory links: {str(e)}")
            return []

    def accept_cookies(self):
        """Accept cookies on the website"""
        try:
            print("Waiting for page to load completely...")
            time.sleep(5)  # Give page more time to load initially
            
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
                
            except Exception as e:
                print(f"Error in cookie button click attempts: {str(e)}")
            
            print("Could not find cookie accept button, trying to continue anyway...")
            return False
            
        except Exception as e:
            print(f"Error in cookie acceptance: {str(e)}")
            self.driver.save_screenshot(os.path.join('data', 'raw', 'scraped_data', 'cookie_error.png'))
            return False

    def get_max_page_number(self):
        """Get the maximum page number from pagination"""
        try:
            # Look for the last page number in pagination
            page_elements = self.driver.find_elements(By.CSS_SELECTOR, "span.paging--display strong")
            if page_elements:
                max_page = int(page_elements[-1].text.strip())
                print(f"Found max page number: {max_page}")
                return max_page
            return 1
        except Exception as e:
            print(f"Error getting max page number: {str(e)}")
            return 1

    def scrape_products_from_page(self, url):
        """Scrape all products from a page and its subsequent pages"""
        try:
            all_products = []
            page = 1
            
            while True:
                # Construct URL for current page
                if page > 1:
                    # Handle URL parameters properly
                    base_url = url.split('?')[0]  # Remove any existing parameters
                    page_url = f"{base_url}?p={page}&o=3&n=36"  # Add standard parameters
                else:
                    page_url = url
                
                print(f"\nScraping page {page}...")
                print(f"URL: {page_url}")
                
                self.driver.get(page_url)
                time.sleep(random.uniform(3, 5))  # Increased initial wait time
                
                # Wait for product container to be present
                try:
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "div.product--box"))
                    )
                except:
                    print("No product container found on page")
                    break
                
                # Find all product boxes
                product_containers = self.driver.find_elements(By.CSS_SELECTOR, "div.product--box")
                
                if not product_containers:
                    print("No product containers found on page")
                    break
                
                for container in product_containers:
                    try:
                        # Extract product name and manufacturer
                        try:
                            name_elem = container.find_element(By.CSS_SELECTOR, "a.product--title")
                            # Get the full title from the title attribute
                            product_name = name_elem.get_attribute('title').strip()
                            # Get manufacturer from first word of title
                            manufacturer = product_name.split()[0]
                        except:
                            print("Could not find product name")
                            continue
                        
                        # Extract price
                        try:
                            price_elem = container.find_element(By.CSS_SELECTOR, "span.price--default")
                            price_text = price_elem.text.strip()
                            
                            # Convert price to float
                            price_match = re.search(r'([\d.,]+)\s*€', price_text)
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
                            'type': product_name,
                            'price': price,
                            'currency': currency,
                            'date_scraped': datetime.now().strftime('%Y-%m-%d'),
                            'page': page
                        }
                        
                        all_products.append(product)
                        print(f"Found product: {manufacturer} - {product_name} - {price} {currency} (Page {page})")
                        
                    except Exception as e:
                        print(f"Error processing product container: {str(e)}")
                        continue
                
                # Check if there are more pages
                max_page = self.get_max_page_number()
                if page >= max_page:
                    print(f"Reached last page ({max_page})")
                    break
                
                page += 1
                # Random delay between pages
                time.sleep(random.uniform(2, 4))
            
            return all_products
            
        except Exception as e:
            print(f"Error scraping products from page: {str(e)}")
            self.driver.save_screenshot(os.path.join('data', 'raw', 'scraped_data', f'error_page_{page}.png'))
            return []

    def scrape_category_recursively(self, category_name, category_url, level=1):
        """Recursively scrape a category and all its subcategories"""
        try:
            print(f"\nProcessing {'subcategory' if level > 1 else 'category'}: {category_name}")
            
            # Get subcategories for current category
            subcategory_links = self.get_subcategory_links(category_url, level)
            
            if not subcategory_links:
                # If no subcategories found, scrape products from current category
                print(f"No subcategories found for {category_name}, scraping products...")
                products = self.scrape_products_from_page(category_url)
                self.all_products.extend(products)
                self.save_progress()
                return
            
            # Process each subcategory
            for subcategory_name, subcategory_url, has_subcategories in subcategory_links:
                if has_subcategories:
                    # Recursively process subcategories
                    self.scrape_category_recursively(subcategory_name, subcategory_url, level + 1)
                else:
                    # Scrape products from leaf category
                    print(f"Scraping products from: {subcategory_name}")
                    products = self.scrape_products_from_page(subcategory_url)
                    self.all_products.extend(products)
                
                # Random delay between subcategories
                time.sleep(random.uniform(2, 4))
            
            # Save progress after processing all subcategories
            self.save_progress()
            
        except Exception as e:
            print(f"Error in recursive category scraping: {str(e)}")
            self.driver.save_screenshot(os.path.join('data', 'raw', 'scraped_data', f'error_category_{category_name}.png'))

    def scrape_all(self):
        """Scrape all products from all categories and subcategories"""
        try:
            # Get all category links
            self.driver.get(self.base_url)
            time.sleep(random.uniform(2, 4))
            
            # Try cookie acceptance multiple times
            max_cookie_attempts = 3
            for attempt in range(max_cookie_attempts):
                if self.accept_cookies():
                    break
                print(f"Cookie acceptance attempt {attempt + 1} failed, retrying...")
                time.sleep(2)
            
            category_links = self.get_category_links()
            
            for category_name, category_url in category_links:
                # Process each category recursively
                self.scrape_category_recursively(category_name, category_url)
                
                # Random delay between main categories
                time.sleep(random.uniform(3, 5))
                
        except Exception as e:
            print(f"Error in scrape_all: {str(e)}")
            self.driver.save_screenshot(os.path.join('data', 'raw', 'scraped_data', 'scrape_all_error.png'))

    def save_progress(self):
        """Save current progress to CSV"""
        if self.all_products:
            current_date = datetime.now().strftime('%Y%m%d')
            filename = f'breeze24_heatpumps_{current_date}.csv'
            
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
        
        scraper = Breeze24Scraper()
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