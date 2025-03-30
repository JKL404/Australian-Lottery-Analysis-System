"""
Universal lottery scraper for OZ lotteries
Supports all lottery types with configurable settings
"""
import requests
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime, timedelta
import os
from pathlib import Path
import logging
import time
import argparse
import pandas as pd
from lotto_core import LOTTERY_TYPES, LotteryValidator
import platform

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('LottoScraper')


class UniversalLottoScraper:
    def __init__(self, lottery_type: str):
        if lottery_type not in LOTTERY_TYPES:
            raise ValueError(f"Invalid lottery type. Choose from: {', '.join(LOTTERY_TYPES.keys())}")
            
        self.lottery_type = lottery_type
        self.config = LOTTERY_TYPES[lottery_type]
        self.validator = LotteryValidator(lottery_type)
        self.results = []
        
    def scrape_with_selenium(self, start_date=None, end_date=None, years_back=3, preferred_state="NSW"):
        """Scrape lottery results using Selenium with date range filtering"""
        from selenium import webdriver
        from selenium.webdriver.firefox.options import Options
        from selenium.webdriver.firefox.service import Service
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from webdriver_manager.firefox import GeckoDriverManager
        
        logger.info(f"Setting up Firefox WebDriver for {self.config.name}...")
        options = Options()
        options.headless = True  # Set to False to show the browser window
        
        # Common Firefox locations on macOS
        mac_firefox_paths = [
            '/Applications/Firefox.app/Contents/MacOS/firefox-bin',
            '/Applications/Firefox.app/Contents/MacOS/firefox',
            '/usr/local/bin/firefox',
            '/opt/homebrew/bin/firefox'
        ]
        
        # Find Firefox binary
        firefox_path = None
        if platform.system() == 'Darwin':  # macOS
            for path in mac_firefox_paths:
                if os.path.exists(path):
                    firefox_path = path
                    break
            if not firefox_path:
                logger.error("Firefox not found. Please install Firefox using: brew install --cask firefox")
                raise RuntimeError("Firefox not found. Install using: brew install --cask firefox")
        elif platform.system() == 'Linux':
            firefox_path = '/usr/bin/firefox'
        elif platform.system() == 'Windows':
            firefox_path = r'C:\Program Files\Mozilla Firefox\firefox.exe'
        
        logger.info(f"Using Firefox binary at: {firefox_path}")
        options.binary_location = firefox_path
        
        # Setup Firefox driver
        service = Service(GeckoDriverManager().install())
        driver = webdriver.Firefox(service=service, options=options)
        
        try:
            # Handle date range
            current_date = end_date or datetime.now()
            start_date = start_date or current_date - timedelta(days=365*years_back)
            
            # Swap dates if reversed
            if start_date > current_date:
                start_date, current_date = current_date, start_date
                
            # Generate URLs for all months in range
            urls = []
            temp_date = start_date.replace(day=1)
            
            while temp_date <= current_date:
                month = temp_date.strftime("%B").lower()
                year = temp_date.year
                urls.append(f"{self.config.base_url}/{month}-{year}")
                
                # Move to next month
                if temp_date.month == 12:
                    temp_date = temp_date.replace(year=temp_date.year+1, month=1)
                else:
                    temp_date = temp_date.replace(month=temp_date.month+1)
            
            logger.info(f"Scraping {len(urls)} months between {start_date.strftime('%Y-%m')} and {current_date.strftime('%Y-%m')}")
            
            for url in urls:
                logger.info(f"Navigating to {url}...")
                try:
                    driver.get(url)
                    time.sleep(5)  # Wait for page to load
                    
                    # Check for state restriction message and handle it
                    self._handle_state_restriction(driver, preferred_state)
                    
                    # Add a longer wait after state handling to ensure the page has updated
                    time.sleep(5)  # Give more time for the page to refresh after state change
                    
                    # Find all date headers and process them individually
                    date_headers = driver.find_elements(By.CSS_SELECTOR, "h4")
                    logger.info(f"Found {len(date_headers)} possible date headers")
                    
                    for header in date_headers:
                        try:
                            self._process_draw_header_selenium(header)
                        except Exception as e:
                            logger.warning(f"Error processing header: {str(e)}")
                            continue
                except Exception as e:
                    logger.error(f"Error processing URL {url}: {str(e)}")
            
            logger.info(f"Extracted {len(self.results)} total results for {self.config.name}")
            
        except Exception as e:
            logger.error(f"Error with Selenium: {str(e)}")
        
        finally:
            driver.quit()
        
        return self._clean_results()
    
    def _handle_state_restriction(self, driver, preferred_state="NSW"):
        """Handle state restriction message and change state if needed"""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.common.exceptions import TimeoutException, NoSuchElementException
        
        try:
            # Check if the state restriction message is present
            restriction_text = "not available in Queensland"
            try:
                state_message = driver.find_element(By.XPATH, f"//*[contains(text(), '{restriction_text}')]")
                if state_message:
                    self._take_screenshot(driver, "state_restriction_detected.png")
                    logger.info("State restriction detected. Attempting to change state...")
            except NoSuchElementException:
                logger.debug("No state restriction detected")
                return
            
            # Try to click the state selector to open the dropdown
            try:
                # Look for the state selector with class that contains 'StateSelector' or by text content
                state_selector = driver.find_element(
                    By.XPATH,
                    "//span[contains(text(), 'State:') or contains(@class, 'CurrentState')]"
                )
                state_selector.click()
                logger.info("Clicked state selector to open dropdown")
                
                # Wait for dropdown to appear
                time.sleep(1)
                
                # Find the preferred state in the dropdown list
                state_items = driver.find_elements(By.CSS_SELECTOR, "li a[role='button']")
                logger.info(f"Found {len(state_items)} state options")
                
                # Find and click on the preferred state
                for item in state_items:
                    if item.text.strip() == preferred_state:
                        logger.info(f"Clicking on state: {preferred_state}")
                        item.click()
                        time.sleep(3)  # Wait for page to update
                        self._take_screenshot(driver, "after_state_change.png")
                        return
                        
                logger.warning(f"Could not find {preferred_state} in dropdown options")
                
                # If preferred state not found, try clicking NSW, VIC, or OTHER as fallbacks
                fallback_states = ["NSW", "VIC", "OTHER"]
                for fallback in fallback_states:
                    for item in state_items:
                        if item.text.strip() == fallback:
                            logger.info(f"Trying fallback state: {fallback}")
                            item.click()
                            time.sleep(3)  # Wait for page to update
                            self._take_screenshot(driver, "after_state_change.png")
                            return
                            
            except Exception as e:
                # Try a more direct approach with the specific CSS classes from the HTML
                try:
                    logger.info("Trying alternative method for state selection")
                    
                    # Click on the state dropdown directly by its class
                    current_state = driver.find_element(By.CSS_SELECTOR, ".css-12fwdzf-CurrentState")
                    current_state.click()
                    logger.info("Clicked on state dropdown")
                    time.sleep(1)
                    
                    # Find all state options by their CSS class
                    state_options = driver.find_elements(By.CSS_SELECTOR, ".css-7xvb4k-State")
                    logger.info(f"Found {len(state_options)} state options with CSS selector")
                    
                    for option in state_options:
                        if option.text.strip() == preferred_state:
                            logger.info(f"Clicking on state: {preferred_state}")
                            option.click()
                            time.sleep(3)  # Wait for page to update
                            self._take_screenshot(driver, "after_state_change.png")
                            return
                            
                    # Try fallbacks
                    for fallback in ["NSW", "VIC", "OTHER"]:
                        for option in state_options:
                            if option.text.strip() == fallback:
                                logger.info(f"Using fallback state: {fallback}")
                                option.click()
                                time.sleep(3)
                                self._take_screenshot(driver, "after_state_change.png")
                                return
                                
                except Exception as nested_e:
                    logger.error(f"Error in alternative state selection: {str(nested_e)}")
                
            # Try JavaScript click as a last resort
            try:
                logger.info(f"Attempting JavaScript click for state: {preferred_state}")
                script = f"""
                var stateItems = document.querySelectorAll('.css-7xvb4k-State');
                for(var i=0; i < stateItems.length; i++) {{
                    if(stateItems[i].textContent.trim() === '{preferred_state}') {{
                        stateItems[i].click();
                        return true;
                    }}
                }}
                return false;
                """
                success = driver.execute_script(script)
                if success:
                    logger.info(f"JavaScript click on {preferred_state} succeeded")
                    time.sleep(3)
                else:
                    logger.warning(f"JavaScript could not find {preferred_state}")
            except Exception as js_error:
                logger.error(f"Error with JavaScript click: {str(js_error)}")
                
        except Exception as e:
            logger.error(f"Error handling state restriction: {str(e)}")
    
    def _process_draw_header_selenium(self, header):
        """Extract and validate draw information from header using Selenium"""
        from selenium.webdriver.common.by import By
        
        try:
            # Extract date and draw number from header text
            header_text = header.text.strip()
            
            # Skip phone numbers, empty text, etc.
            if not header_text or "1300" in header_text or len(header_text) < 10:
                return
            
            # Log the header text for debugging
            logger.debug(f"Processing header text: {header_text}")
            
            # Try more flexible pattern matching
            date_match = re.search(self.config.result_pattern, header_text, re.IGNORECASE)
            if not date_match:
                # Try alternative pattern with just date and draw
                alt_pattern = r"(\d{1,2}\s+\w+\s+\d{4}).*?(\d{4})"
                date_match = re.search(alt_pattern, header_text, re.IGNORECASE)
                if not date_match:
                    logger.debug(f"No date/draw match in: {header_text}")
                    return
            
            date_str, draw_number = date_match.groups()
            logger.debug(f"Found date: {date_str}, draw: {draw_number}")
            
            # Find the associated draw container with multiple fallbacks
            container = None
            try:
                container = header.find_element(By.XPATH, "./following-sibling::div[contains(@class, 'DrawResult')][1]")
            except:
                try:
                    container = header.find_element(By.XPATH, "./following-sibling::div[1]")
                except:
                    try:
                        # Try parent's next sibling
                        container = header.find_element(By.XPATH, "../following-sibling::div[1]")
                    except:
                        logger.warning(f"Could not find container for {date_str}, draw {draw_number}")
                        return
            
            if not container:
                logger.warning("Container element not found")
                return
            
            # Try multiple selectors for numbers
            number_elements = []
            selectors = [
                "div[class*='number']", 
                "[class*='ball']", 
                "span[class*='number']",
                "div.number",
                "span.number"
            ]
            
            for selector in selectors:
                try:
                    elements = container.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        number_elements = elements
                        logger.debug(f"Found {len(elements)} numbers with selector: {selector}")
                        break
                except:
                    continue
            
            numbers = []
            for elem in number_elements:
                try:
                    num_text = elem.text.strip()
                    if num_text.isdigit():
                        numbers.append(int(num_text))
                except:
                    continue
                
            logger.debug(f"Extracted numbers: {numbers}")
            
            # Check if we have enough numbers
            total_needed = self.config.main_numbers + self.config.supplementary
            if len(numbers) < total_needed:
                logger.warning(f"Not enough numbers found: {len(numbers)}/{total_needed}")
                return
            
            # Split into main numbers and supplementary
            main_numbers = numbers[:self.config.main_numbers]
            supplementary = numbers[self.config.main_numbers:self.config.main_numbers + self.config.supplementary]
            
            result = {
                'date': date_str,  # Keep original format for now, validate later
                'draw': draw_number,
                'main_numbers': main_numbers,
                'supplementary': supplementary
            }
            
            # Try to convert date to standard format
            try:
                date_obj = datetime.strptime(date_str, self.config.date_format)
                result['date'] = date_obj.strftime("%Y-%m-%d")
            except ValueError:
                # Try alternate formats
                for fmt in ["%d %B %Y", "%d %b %Y", "%A, %d %B %Y", "%A %d %B %Y"]:
                    try:
                        date_obj = datetime.strptime(date_str, fmt)
                        result['date'] = date_obj.strftime("%Y-%m-%d")
                        break
                    except ValueError:
                        continue
            
            # Add with less strict validation
            try:
                if len(main_numbers) == self.config.main_numbers and len(supplementary) == self.config.supplementary:
                    self.results.append(result)
                    logger.info(f"Successfully parsed draw {draw_number}: {main_numbers} + {supplementary}")
                else:
                    logger.warning(f"Wrong number count - main: {len(main_numbers)}/{self.config.main_numbers}, supp: {len(supplementary)}/{self.config.supplementary}")
            except Exception as e:
                logger.warning(f"Validation error: {str(e)}")
            
        except Exception as e:
            logger.warning(f"Error processing header: {str(e)}")
    
    def _clean_results(self):
        """Clean and deduplicate results"""
        # Deduplicate by draw number
        unique_results = {}
        for result in self.results:
            draw_key = result['draw']
            unique_results[draw_key] = result
        
        # Convert to list and sort by draw number (descending)
        cleaned = list(unique_results.values())
        cleaned.sort(key=lambda x: int(x['draw']) if x['draw'].isdigit() else 0, reverse=True)
        
        return cleaned
    
    def save_results(self, format="json"):
        """Save results to file"""
        try:
            # Create results directory if it doesn't exist
            output_dir = Path("results")
            output_dir.mkdir(exist_ok=True)
            
            filename = output_dir / f"{self.lottery_type}_results.{format}"
            
            if format.lower() == "json":
                with open(filename, 'w') as f:
                    json.dump(self.results, f, indent=2)
            elif format.lower() == "csv":
                df = pd.DataFrame(self.results)
                
                # Format lists as space-separated strings for CSV
                if len(self.results) > 0:
                    df["main_numbers"] = df["main_numbers"].apply(
                        lambda x: " ".join(map(str, x)) + " "
                    )
                    df["supplementary"] = df["supplementary"].apply(
                        lambda x: " ".join(map(str, x)) + " "
                    )
                
                df.to_csv(filename, index=False)
            
            logger.info(f"Results saved to {filename}")
            return str(filename)
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

    def scrape(self, years_back: int = 5):
        """Main scraping method - delegates to scrape_with_selenium for backward compatibility"""
        logger.info(f"Starting scrape for {self.config.name} from {years_back} years back")
        
        # Calculate date range based on years_back
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*years_back)
        
        # Delegate to the selenium implementation
        return self.scrape_with_selenium(
            start_date=start_date,
            end_date=end_date,
            years_back=years_back
        )

    def _take_screenshot(self, driver, filename=None):
        """Take a screenshot to document page state"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.png"
            
            screenshots_dir = Path("screenshots")
            screenshots_dir.mkdir(exist_ok=True)
            
            screenshot_path = screenshots_dir / filename
            driver.save_screenshot(str(screenshot_path))
            logger.info(f"Screenshot saved to {screenshot_path}")
        except Exception as e:
            logger.error(f"Error taking screenshot: {str(e)}")

def run_scraper(start_date=None, end_date=None, lottery_type="saturday_lotto", years_back=3):
    """Main function to run the scraper with optional parameters"""
    try:
        scraper = UniversalLottoScraper(lottery_type)
        
        # Use Selenium with date range filtering
        scraper.scrape_with_selenium(start_date=start_date, end_date=end_date, years_back=years_back)
        
        # Save results in both formats
        scraper.save_results("json")
        scraper.save_results("csv")
        
        print(f"✅ Scraping completed for {lottery_type}. Found {len(scraper.results)} draws.")
        return scraper.results
    except Exception as e:
        logger.error(f"Error in scraper: {str(e)}")
        print(f"❌ Scraping failed: {str(e)}")
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scrape Lottery results')
    parser.add_argument('--lottery', choices=list(LOTTERY_TYPES.keys()), default='saturday_lotto',
                       help='Lottery type to scrape')
    parser.add_argument('--start', type=str, help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end', type=str, help='End date in YYYY-MM-DD format')
    parser.add_argument('--years', type=int, default=3, help='Years of history to scrape')
    args = parser.parse_args()

    # Convert string dates to datetime objects
    start_date = datetime.strptime(args.start, "%Y-%m-%d") if args.start else None
    end_date = datetime.strptime(args.end, "%Y-%m-%d") if args.end else None

    run_scraper(
        start_date=start_date, 
        end_date=end_date, 
        lottery_type=args.lottery,
        years_back=args.years
    )