from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import os
import time

def test_chrome():
    try:
        print("Starting Chrome test...")
        
        # Setup Chrome options
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # Run in headless mode
        
        # Get path to local chromedriver
        chromedriver_path = os.path.join(os.path.dirname(__file__), 'chromedriver')
        print(f"ChromeDriver path: {chromedriver_path}")
        
        # Create service object
        service = Service(executable_path=chromedriver_path)
        
        # Initialize the driver with service object
        print("Initializing Chrome driver...")
        driver = webdriver.Chrome(
            service=service,
            options=chrome_options
        )
        
        # Try to load Apple's website
        print("Attempting to load apple.com...")
        driver.get("https://www.apple.com")
        
        # Wait a bit for the page to load
        time.sleep(3)
        
        # Get the page title
        title = driver.title
        print(f"Successfully loaded page. Title: {title}")
        
        # Get current URL
        current_url = driver.current_url
        print(f"Current URL: {current_url}")
        
        # Close the browser
        driver.quit()
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
if __name__ == "__main__":
    test_chrome() 