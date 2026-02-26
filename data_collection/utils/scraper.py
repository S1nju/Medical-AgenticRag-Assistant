from selenium.webdriver.common.by import By
import re

def scrape_medicament_details(driver):
    """
    Extracts details from the medicament page.
    Assumes the driver is already on the medicament page.
    """
    details = {}
    
    try:
        # Check if login modal blocks content (optional handling if needed)
        
        # Main info on the left/top
        # The structure is specific:
        # col-lg-7 ... contains several links and strong tags
        
        # We can locate elements by text content or structure
        
        # Header/Title is usually outside extraction area provided in snippet, 
        # but let's assume we want to extract fields shown in the snippet.
        
        container = driver.find_element(By.CLASS_NAME, 'col-lg-8') 
        info_block = container.find_element(By.CLASS_NAME, 'col-lg-7')
        
        # Extract Links (Laboratory, Pharmaco Class, Therapeutic Class, DCI)
        links = info_block.find_elements(By.TAG_NAME, 'a')
        
        details['laboratory'] = links[0].text if len(links) > 0 else ""
        details['pharmaco_class'] = links[1].text if len(links) > 1 else ""
        details['theraputic_class'] = links[2].text if len(links) > 2 else ""
        details['dci'] = links[3].text if len(links) > 3 else ""
        
        # Parse text content for Strong tags
        text_content = info_block.get_attribute("innerText")
        
        def extract_field(label, text):
            match = re.search(f"{label}: (.*)", text)
            return match.group(1).strip() if match else ""

        details['commercial_name'] = extract_field("Nom Commercial", text_content)
        details['dci_code'] = extract_field("Code DCI", text_content)
        details['form'] = extract_field("Forme", text_content)
        details['dosage'] = extract_field("Dosage", text_content)
        details['conditioning'] = extract_field("Conditionnement", text_content)
        
        # Right side info block
        status_block = container.find_element(By.CLASS_NAME, 'col-lg-5')
        status_text = status_block.get_attribute("innerText")
        
        details['type'] = extract_field("Type", status_text)
        details['list'] = extract_field("Liste", status_text)
        details['country'] = extract_field("Pays", status_text)
        
        # Check icons for boolean values
        # Commercialisation
        try:
            comm_icon = status_block.find_element(By.XPATH, ".//strong[contains(text(),'Commercialisation')]/following-sibling::i")
            details['marketed'] = "check" in comm_icon.get_attribute("class")
        except:
            details['marketed'] = False
            
        # Remboursable
        try:
            remb_icon = status_block.find_element(By.XPATH, ".//strong[contains(text(),'Remboursable')]/following-sibling::i")
            details['reimbursable'] = "check" in remb_icon.get_attribute("class")
        except:
            details['reimbursable'] = False

        ref_price_str = extract_field("Tarif de référence", status_text).replace(" DA", "")
        try:
            details['reference_price'] = float(ref_price_str)
        except:
            details['reference_price'] = None
            
        details['ppa_indicative'] = extract_field("PPA \(indicatif\)", status_text)
        details['registration_num'] = extract_field("Num Enregistrement", status_text)

        # Image and Notice
        image_block = driver.find_element(By.CLASS_NAME, 'col-lg-4')
        try:
            img_element = image_block.find_element(By.TAG_NAME, 'img')
            details['img_link'] = img_element.get_attribute('src')
        except:
            details['img_link'] = ""
            
        try:
            notice_link = image_block.find_element(By.PARTIAL_LINK_TEXT, "Notice")
            details['notice_link'] = notice_link.get_attribute('href')
        except:
            details['notice_link'] = ""

        # Name is usually the h3 title from the very top
        try:
            title_element = driver.find_element(By.XPATH, "//h3[contains(@style, 'text-shadow')]")
            details['name'] = title_element.text.strip()
        except:
            details['name'] = details['commercial_name'] # Fallback

    except Exception as e:
        print(f"Error scraping details: {e}")
        return None
        
    return details
