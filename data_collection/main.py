from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from utils.medicaments import get_medicament_links_from_table
from utils.scraper import scrape_medicament_details
from database.models import create_tables, Medicament, Session
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
# This line is the magic part—it downloads the version 145 driver automatically
service = Service(ChromeDriverManager().install())

driver = webdriver.Chrome(service=service)

Letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
driver.get("https://www.pharmnet-dz.com/alphabet.aspx?char=A")

# Scroll down the page
driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
# Wait for content to load
create_tables()
session = Session()

for letter in Letters:
    driver.get(f"https://www.pharmnet-dz.com/alphabet.aspx?char={letter}")
    # Scroll down the page
    driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
    # Wait for content to load
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, 'MainContent_DataTable'))
    )

    TABLE = driver.find_elements(By.ID, 'MainContent_DataTable')
    pagination = driver.find_elements(By.CLASS_NAME,"btn-group-justified")
    numbers  = ""
    for el in pagination:
        numbers = el.text

    if numbers.strip():
        numbers =  list(map(str, numbers.split()))
    else:
        numbers = ['1']


    for i in numbers:
        if i != '1':
            driver.get(f"https://www.pharmnet-dz.com/alphabet.aspx?char={letter}&p={i}")
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, 'MainContent_DataTable'))
            )
            
        TABLE = driver.find_elements(By.ID, 'MainContent_DataTable')
        page_medicaments = get_medicament_links_from_table(TABLE)
        
        for element in page_medicaments:
            if not element['link']:
                continue
                
            driver.get(element['link'])
            
            # Scrape details
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "col-lg-8"))
                )
                
                details = scrape_medicament_details(driver)
                
                if details:
                    # Check if exists
                    exists = session.query(Medicament).filter_by(name=details['name']).first()
                    if not exists:
                        print(f"Adding to DB: {details.get('name')}")
                        med = Medicament(**details)
                        session.add(med)
                        session.commit()
                    else:
                        print(f"Skipping {details.get('name')} - Already exists")

            except Exception as e:
                print(f"Error processing {element['link']}: {e}")
                session.rollback()
    
print("Scraping Completed")


