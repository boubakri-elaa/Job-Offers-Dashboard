from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

def scrape_hellowork():
    print("Chargement de la page...")

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    url = "https://www.hellowork.com/fr-fr/emploi/recherche.html?k=&l=&st=relevance"
    driver.get(url)
    time.sleep(5)

    jobs = driver.find_elements(By.CSS_SELECTOR, "article")

    if not jobs:
        print("❌ Aucune offre trouvée !")
        driver.quit()
        return

    all_offers = []
    for job in jobs:
        try:
            titre = job.find_element(By.CSS_SELECTOR, "h3").text
            entreprise = job.find_element(By.CSS_SELECTOR, ".company").text
            all_offers.append((titre, entreprise))
        except:
            continue

    driver.quit()

    print(f"Nombre total d'offres récupérées : {len(all_offers)}")

scrape_hellowork()
