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

class Heima24Scraper:
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
        self.base_url = "https://www.heima24.de/heizung/waermepumpen/"
        self.all_products = []
        self.all_category_urls = set()  # Track all visited category URLs to avoid duplicates
        self.manufacturers = ["buderus", "lg", "vaillant", "viessmann", "wolf"]
        
        # Direct URLs to product category pages
        self.direct_category_urls = [
            "https://www.heima24.de/heizung/waermepumpen/buderus/wpt/",
            "https://www.heima24.de/heizung/waermepumpen/buderus/wlw176i-ar/einzelgeraete/",
            "https://www.heima24.de/heizung/waermepumpen/buderus/wlw176i-ar/e-mit-integriertem-heizstab/",
            "https://www.heima24.de/heizung/waermepumpen/buderus/wlw176i-ar/tp70-mit-integriertem-pufferspeicher/",
            "https://www.heima24.de/heizung/waermepumpen/buderus/wlw176i-ar/t180-mit-integriertem-warmwasserspeicher/",
            "https://www.heima24.de/heizung/waermepumpen/buderus/wlw186i-ar/einzelgeraete/",
            "https://www.heima24.de/heizung/waermepumpen/buderus/wlw186i-ar/e-mit-integriertem-heizstab/",
            "https://www.heima24.de/heizung/waermepumpen/buderus/wlw186i-ar/tp70-mit-integriertem-pufferspeicher/",
            "https://www.heima24.de/heizung/waermepumpen/buderus/wlw186i-ar/t180-mit-integriertem-warmwasserspeicher/",
            "https://www.heima24.de/heizung/waermepumpen/buderus/wlw196i-ar/e-mit-integriertem-heizstab/",
            "https://www.heima24.de/heizung/waermepumpen/buderus/wlw196i-ar/b-mit-integriertem-mischer-zur-kesseleinbindung/",
            "https://www.heima24.de/heizung/waermepumpen/buderus/wlw196i-ar/tp120-mit-integriertem-pufferspeicher/",
            "https://www.heima24.de/heizung/waermepumpen/buderus/wlw196i-ar/t190-mit-integriertem-warmwasserspeicher/",
            "https://www.heima24.de/heizung/waermepumpen/buderus/wlw196i-ir/e-mit-integriertem-heizstab/",
            "https://www.heima24.de/heizung/waermepumpen/buderus/wlw196i-ir/b-mit-integriertem-mischer-zur-kesseleinbindung/",
            "https://www.heima24.de/heizung/waermepumpen/buderus/wlw196i-ir/tp120-mit-integriertem-pufferspeicher/",
            "https://www.heima24.de/heizung/waermepumpen/buderus/wlw196i-ir/t190-mit-integriertem-warmwasserspeicher/",
            "https://www.heima24.de/heizung/waermepumpen/buderus/wlw196i2-ar-s/e-mit-integriertem-heizstab/",
            "https://www.heima24.de/heizung/waermepumpen/buderus/wlw196i2-ar-s/b-mit-integriertem-mischer-zur-kesseleinbindung/",
            "https://www.heima24.de/heizung/waermepumpen/buderus/wlw196i2-ar-s/tp120-mit-integriertem-pufferspeicher/",
            "https://www.heima24.de/heizung/waermepumpen/buderus/wlw196i2-ar-s/t190-mit-integriertem-warmwasserspeicher/",
            "https://www.heima24.de/heizung/waermepumpen/lg/therma-v-whfs/",
            "https://www.heima24.de/heizung/waermepumpen/lg/therma-v-r290/",
            "https://www.heima24.de/heizung/waermepumpen/lg/therma-v-r32/",
            "https://www.heima24.de/heizung/waermepumpen/vaillant/arostor/",
            "https://www.heima24.de/heizung/waermepumpen/vaillant/fluostor/",
            "https://www.heima24.de/heizung/waermepumpen/vaillant/arotherm-plus/",
            "https://www.heima24.de/heizung/waermepumpen/vaillant/arotherm-plus-mit-unitower/",
            "https://www.heima24.de/heizung/waermepumpen/vaillant/arotherm-split-plus/",
            "https://www.heima24.de/heizung/waermepumpen/vaillant/arotherm-split-plus-mit-unitower/",
            "https://www.heima24.de/heizung/waermepumpen/vaillant/flexocompact-exclusive/",
            "https://www.heima24.de/heizung/waermepumpen/vaillant/flexotherm-exclusive/",
            "https://www.heima24.de/heizung/waermepumpen/vaillant/recocompact-exclusive/",
            "https://www.heima24.de/heizung/waermepumpen/vaillant/versotherm-plus/",
            "https://www.heima24.de/heizung/waermepumpen/viessmann/vitocal-060-a/",
            "https://www.heima24.de/heizung/waermepumpen/viessmann/vitocal-262-a/",
            "https://www.heima24.de/heizung/waermepumpen/viessmann/vitocal-150-a-151-a/vitocal-150-a/",
            "https://www.heima24.de/heizung/waermepumpen/viessmann/vitocal-150-a-151-a/vitocal-151-a-mit-integriertem-speicher/",
            "https://www.heima24.de/heizung/waermepumpen/viessmann/vitocal-200-a-222-a/vitocal-200-a/",
            "https://www.heima24.de/heizung/waermepumpen/viessmann/vitocal-200-a-222-a/vitocal-222-a-mit-integriertem-speicher/",
            "https://www.heima24.de/heizung/waermepumpen/viessmann/vitocal-200-s-222-s-r32-split/vitocal-200-s-r32/",
            "https://www.heima24.de/heizung/waermepumpen/viessmann/vitocal-200-s-222-s-r32-split/vitocal-222-s-r32-mit-integriertem-speicher/",
            "https://www.heima24.de/heizung/waermepumpen/viessmann/vitocal-200-s-222-s-split/vitocal-200-s/",
            "https://www.heima24.de/heizung/waermepumpen/viessmann/vitocal-200-s-222-s-split/vitocal-222-s-mit-integriertem-speicher/",
            "https://www.heima24.de/heizung/waermepumpen/viessmann/vitocal-250-a-252-a/vitocal-250-a/",
            "https://www.heima24.de/heizung/waermepumpen/viessmann/vitocal-250-a-252-a/vitocal-252-a-mit-integriertem-speicher/",
            "https://www.heima24.de/heizung/waermepumpen/wolf/fhs/",
            "https://www.heima24.de/heizung/waermepumpen/wolf/cha-chc-monoblock/cha-mit-integriertem-heizelement/",
            "https://www.heima24.de/heizung/waermepumpen/wolf/cha-chc-monoblock/chc-mit-integriertem-warmwasserspeicher/",
            "https://www.heima24.de/heizung/waermepumpen/wolf/cha-chc-monoblock/chc-mit-integriertem-warmwasserspeicher-pufferspeicher/",
            "https://www.heima24.de/heizung/waermepumpen/wolf/fha-monoblock/fha-mit-integriertem-heizelement/",
            "https://www.heima24.de/heizung/waermepumpen/wolf/fha-monoblock/fha-mit-integriertem-warmwasserspeicher/",
            "https://www.heima24.de/heizung/waermepumpen/wolf/fha-monoblock/fha-mit-integriertem-warmwasserspeicher-pufferspeicher/"
        ]
        
    def accept_cookies(self):
        """Accept cookies on the website"""
        try:
            time.sleep(3)
            # Add different selectors for cookie acceptance button
            cookie_selectors = [
                "button.cookie-permission--accept-button",
                "button[data-cookieman-save]",
                ".cookie-consent--button button",
                "a.privacyBox__accept",
                "#cookiehinweis .btn-primary",
                ".cc-compliance .cc-btn",
                ".cookie-info-screen .btn"
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

    def scrape_direct_url(self, url):
        """Scrape products directly from a given URL"""
        print(f"\n===== Scraping URL: {url} =====")
        try:
            self.driver.get(url)
            time.sleep(random.uniform(4, 6))  # Longer wait time to ensure page loads
            
            # First try to find product links
            product_urls = self.extract_product_urls_from_page()
            
            # Process each product URL
            for product_url in product_urls:
                try:
                    # Add random delay between product page visits
                    time.sleep(random.uniform(2, 4))
                    
                    # Extract product details
                    product = self.extract_product_details(product_url)
                    
                    if product:
                        self.all_products.append(product)
                        print(f"Added product: {product['type']} - €{product['price']}")
                        
                        # Save every few products
                        if len(self.all_products) % 5 == 0:
                            self.save_progress()
                except Exception as e:
                    print(f"Error processing product URL {product_url}: {str(e)}")
                    continue
            
            # Check for pagination
            self.process_pagination()
            
            return len(product_urls)
            
        except Exception as e:
            print(f"Error scraping URL {url}: {str(e)}")
            # Take a screenshot for debugging
            self.driver.save_screenshot(os.path.join('data', 'raw', 'scraped_data', f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
            return 0
    
    def extract_product_urls_from_page(self):
        """Extract product URLs from the current page"""
        product_urls = []
        
        # Try multiple approaches to find product links
        
        # Approach 1: Find through div#os_list_prod.row container
        try:
            product_container = self.driver.find_element(By.CSS_SELECTOR, "div#os_list_prod.row")
            product_items = product_container.find_elements(By.CSS_SELECTOR, "div.col-lg-4.col-md-6.col-sm-6.col-xs-12")
            
            print(f"Found {len(product_items)} product items in main container")
            
            for item in product_items:
                try:
                    # Try to find link in the title section
                    link_element = item.find_element(By.CSS_SELECTOR, "a.os_list_link1")
                    href = link_element.get_attribute("href")
                    if href:
                        # Make URL absolute if needed
                        if not href.startswith("http"):
                            if href.startswith("/"):
                                href = "https://www.heima24.de" + href
                            else:
                                href = "https://www.heima24.de/" + href
                        
                        product_urls.append(href)
                        print(f"Found product link: {href}")
                except:
                    # If not found in title, try the image section
                    try:
                        link_element = item.find_element(By.CSS_SELECTOR, "div.os_list_box1_all a")
                        href = link_element.get_attribute("href")
                        if href:
                            # Make URL absolute if needed
                            if not href.startswith("http"):
                                if href.startswith("/"):
                                    href = "https://www.heima24.de" + href
                                else:
                                    href = "https://www.heima24.de/" + href
                            
                            product_urls.append(href)
                            print(f"Found product link from image area: {href}")
                    except:
                        pass
        except Exception as e:
            print(f"Approach 1 failed: {str(e)}")
            
            # Approach 2: Find links directly
            try:
                product_links = self.driver.find_elements(By.CSS_SELECTOR, "a.os_list_link1")
                for link in product_links:
                    href = link.get_attribute("href")
                    if href:
                        # Make URL absolute if needed
                        if not href.startswith("http"):
                            if href.startswith("/"):
                                href = "https://www.heima24.de" + href
                            else:
                                href = "https://www.heima24.de/" + href
                        
                        product_urls.append(href)
                        print(f"Found product link directly: {href}")
            except Exception as e:
                print(f"Approach 2 failed: {str(e)}")
                
                # Approach 3: Look for any clickable div that might redirect to product
                try:
                    product_divs = self.driver.find_elements(By.CSS_SELECTOR, "div[onclick*='wurl']")
                    for div in product_divs:
                        onclick = div.get_attribute("onclick")
                        if onclick and "wurl" in onclick:
                            # Extract URL from onclick attribute
                            url_match = re.search(r"wurl\('([^']+)'\)", onclick)
                            if url_match:
                                href = url_match.group(1)
                                # Make URL absolute if needed
                                if not href.startswith("http"):
                                    if href.startswith("/"):
                                        href = "https://www.heima24.de" + href
                                    else:
                                        href = "https://www.heima24.de/" + href
                                
                                product_urls.append(href)
                                print(f"Found product link from onclick: {href}")
                except Exception as e:
                    print(f"Approach 3 failed: {str(e)}")
        
        # Remove duplicates
        product_urls = list(set(product_urls))
        print(f"Found {len(product_urls)} unique product URLs")
        return product_urls
    
    def process_pagination(self):
        """Process pagination on the current page"""
        try:
            pagination = self.driver.find_elements(By.CSS_SELECTOR, "div.os_list_pages a")
            for page_link in pagination:
                href = page_link.get_attribute("href")
                text = page_link.text.strip()
                # Check if it's a "next page" link - could be ">" or a number
                if href and text and (text == ">" or text.isdigit() and int(text) > 1):
                    print(f"Found pagination link: {href} - {text}")
                    # Visit the pagination page
                    self.driver.get(href)
                    time.sleep(random.uniform(3, 5))  # Wait for page to load
                    
                    # Extract product URLs from this pagination page
                    product_urls = self.extract_product_urls_from_page()
                    
                    # Process each product URL
                    for product_url in product_urls:
                        try:
                            # Add random delay between product page visits
                            time.sleep(random.uniform(2, 4))
                            
                            # Extract product details
                            product = self.extract_product_details(product_url)
                            
                            if product:
                                self.all_products.append(product)
                                print(f"Added product from pagination: {product['type']} - €{product['price']}")
                                
                                # Save every few products
                                if len(self.all_products) % 5 == 0:
                                    self.save_progress()
                        except Exception as e:
                            print(f"Error processing pagination product URL {product_url}: {str(e)}")
                            continue
        except Exception as e:
            print(f"Error processing pagination: {str(e)}")

    def extract_product_details(self, product_url):
        """Extract details from a product page"""
        try:
            print(f"Extracting details from: {product_url}")
            self.driver.get(product_url)
            time.sleep(random.uniform(3, 5))  # Longer wait time
            
            # Determine manufacturer from URL
            default_manufacturer = None
            for mfr in self.manufacturers:
                if mfr.lower() in product_url.lower():
                    default_manufacturer = mfr
                    print(f"Determined manufacturer from URL: {default_manufacturer}")
                    break
            
            if not default_manufacturer:
                print(f"Product URL doesn't contain any target manufacturer, skipping: {product_url}")
                return None
            
            # Extract type/title - try multiple approaches
            product_type = ""
            
            # Approach 1: Try h1 element
            try:
                type_element = self.driver.find_element(By.CSS_SELECTOR, "h1")
                product_type = type_element.text.strip()
                print(f"Found product type from h1: {product_type}")
            except:
                # Approach 2: Try h2 element
                try:
                    type_element = self.driver.find_element(By.CSS_SELECTOR, "h2")
                    product_type = type_element.text.strip()
                    print(f"Found product type from h2: {product_type}")
                except:
                    # Approach 3: Try article title class
                    try:
                        type_element = self.driver.find_element(By.CSS_SELECTOR, ".os_detail_arttitle")
                        product_type = type_element.text.strip()
                        print(f"Found product type from article title: {product_type}")
                    except:
                        # Approach 4: Extract from page title
                        try:
                            page_title = self.driver.title
                            # Remove website name and similar from title
                            cleaned_title = re.sub(r'\s*[-|]\s*.*$', '', page_title)
                            product_type = cleaned_title.strip()
                            print(f"Found product type from page title: {product_type}")
                        except:
                            # Last resort: Construct from URL
                            try:
                                path_parts = product_url.split('/')
                                if len(path_parts) > 4:
                                    file_name = path_parts[-1].split('?')[0]  # Remove query parameters
                                    product_type = file_name.replace('-', ' ').replace('.html', '')
                                    product_type = f"{default_manufacturer.upper()} {product_type}"
                                    print(f"Constructed product type from URL: {product_type}")
                                else:
                                    product_type = f"{default_manufacturer.capitalize()} Wärmepumpe"
                                    print(f"Using generic product type: {product_type}")
                            except:
                                product_type = f"{default_manufacturer.capitalize()} Wärmepumpe"
                                print(f"Using default product type: {product_type}")
            
            # Extract manufacturer from type (first word) if possible
            manufacturer = default_manufacturer
            if product_type and ' ' in product_type:
                try:
                    first_word = product_type.split()[0].lower()
                    if first_word in [m.lower() for m in self.manufacturers]:
                        manufacturer = first_word
                        print(f"Extracted manufacturer from product type: {manufacturer}")
                    else:
                        print(f"First word '{first_word}' is not a target manufacturer, using URL-derived manufacturer")
                except IndexError:
                    print(f"Error extracting first word from product_type: {product_type}")
                    # Use the default_manufacturer determined from URL
            
            # Extract price - try multiple approaches
            price_text = None
            
            # Approach 1: Try os_detail_price span (detail page)
            try:
                price_element = self.driver.find_element(By.CSS_SELECTOR, "span.os_detail_price")
                price_text = price_element.text.strip()
                print(f"Found price from detail page: {price_text}")
            except:
                # Approach 2: Try os_list_price2 div (list page)
                try:
                    price_div = self.driver.find_element(By.CSS_SELECTOR, "div.os_list_price2")
                    price_span = price_div.find_element(By.CSS_SELECTOR, "span")
                    price_text = price_span.text.strip()
                    print(f"Found price from list page: {price_text}")
                except:
                    # Approach 3: Look for any element containing a price pattern
                    try:
                        # Find any element that might contain a price
                        body_text = self.driver.find_element(By.TAG_NAME, "body").text
                        price_pattern = r'(\d{1,3}(?:\.\d{3})*,\d{2})\s*(?:EUR|€)'
                        price_match = re.search(price_pattern, body_text)
                        if price_match:
                            price_text = price_match.group(0)
                            print(f"Found price from body text: {price_text}")
                        else:
                            print("No price found in body text")
                            return None
                    except Exception as e:
                        print(f"Error finding price: {str(e)}")
                        return None
            
            if not price_text:
                print("No price found for this product")
                return None
            
            # Extract numeric price value
            price_match = re.search(r'([\d.,]+)', price_text)
            if price_match:
                price_str = price_match.group(1).replace('.', '').replace(',', '.')
                try:
                    price = float(price_str)
                    print(f"Extracted price value: €{price}")
                    
                    # Create product dictionary
                    product = {
                        'manufacturer': manufacturer,
                        'type': product_type,
                        'price': price,
                        'currency': 'EUR',
                        'page': product_url,
                        'date_scraped': datetime.now().strftime('%Y-%m-%d')
                    }
                    
                    return product
                except ValueError as e:
                    print(f"Error converting price to float: {str(e)}")
                    return None
            else:
                print("Could not extract price")
                return None
                
        except Exception as e:
            print(f"Error extracting product details: {str(e)}")
            return None

    def scrape_direct_categories(self):
        """Scrape products directly from predefined category URLs"""
        try:
            # Create scraped_data directory if it doesn't exist
            os.makedirs(os.path.join('data', 'raw', 'scraped_data'), exist_ok=True)
            
            # Navigate to base URL first to handle cookies
            self.driver.get(self.base_url)
            time.sleep(random.uniform(3, 5))
            
            # Accept cookies
            self.accept_cookies()
            
            total_products_found = 0
            
            # Process each direct URL
            for url in self.direct_category_urls:
                try:
                    products_found = self.scrape_direct_url(url)
                    total_products_found += products_found
                    
                    print(f"Found {products_found} products from {url}")
                    print(f"Total products found so far: {len(self.all_products)}")
                    
                    # Save progress after each category
                    self.save_progress()
                    
                    # Add random delay between categories
                    time.sleep(random.uniform(3, 6))
                except Exception as e:
                    print(f"Error processing category URL {url}: {str(e)}")
                    continue
            
            print(f"Finished scraping all categories. Total products found: {len(self.all_products)}")
            self.save_progress()  # Save final results
            
        except Exception as e:
            print(f"Error in scrape_direct_categories: {str(e)}")
            self.driver.save_screenshot(os.path.join('data', 'raw', 'scraped_data', 'scrape_error.png'))
            self.save_progress()  # Save progress even if error occurs

    def scrape_all(self):
        """Main scraping function - uses direct URLs now"""
        self.scrape_direct_categories()

    def save_progress(self):
        """Save current progress to CSV"""
        if self.all_products:
            current_date = datetime.now().strftime('%Y%m%d')
            filename = f'heima24_heatpumps_{current_date}.csv'
            
            df = pd.DataFrame(self.all_products)
            df.to_csv(os.path.join('data', 'raw', 'scraped_data', filename), index=False)
            print(f"Progress saved: {len(self.all_products)} products found")

    def close(self):
        """Close the browser"""
        self.driver.quit()

def main():
    """Main function to run the scraper"""
    try:
        scraper = Heima24Scraper()
        scraper.scrape_all()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
    finally:
        if 'scraper' in locals():
            scraper.close()

if __name__ == "__main__":
    main() 