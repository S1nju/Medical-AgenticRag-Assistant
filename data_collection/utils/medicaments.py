from selenium.webdriver.common.by import By


def get_medicament_links_from_table(TABLE):
    if not TABLE:
        return []
    medicaments=[]
    for item in TABLE:
     for i in item.find_elements(By.TAG_NAME, 'tr'):
        medicament_name = i.find_element(By.TAG_NAME,'strong').text
        med_link=""
        Laboratoire=""
        CTherapeutique=""
        for links  in i.find_elements(By.TAG_NAME,'a'):
            link = links.get_attribute('href')
            if link and 'medic.aspx?id=' in link:
                med_link = link
        medicaments.append({
                'name':medicament_name,
                'link':med_link
            })
    return medicaments
    
        
    