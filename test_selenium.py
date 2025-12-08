from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import csv

# Initialisation du driver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# URL de recherche sans filtre pour toutes les offres
base_url = "https://www.hellowork.com/fr-fr/emploi/recherche.html?k=&l=&d=all&page="

all_offers = []
page = 1

while True:
    url = base_url + str(page)
    driver.get(url)
    print(f"⏳ Chargement de la page {page}...")
    time.sleep(5)  # attendre le chargement complet

    # Récupérer toutes les cartes visibles
    cards = driver.find_elements(By.CSS_SELECTOR, "div[data-cy='serpCard']")
    if not cards:  # plus d'offres -> sortie de la boucle
        print("🚫 Plus d'offres disponibles.")
        break

    print(f"Page {page} : {len(cards)} offres trouvées.")

    # Extraire les infos de chaque offre
    for card in cards:
        try:
            titre = card.find_element(By.CSS_SELECTOR, "h2[data-cy='jobTitle']").text
        except:
            titre = ""
        try:
            entreprise = card.find_element(By.CSS_SELECTOR, "span[data-cy='companyName']").text
        except:
            entreprise = ""
        try:
            ville = card.find_element(By.CSS_SELECTOR, "span[data-cy='jobLocation']").text
        except:
            ville = ""
        try:
            date = card.find_element(By.CSS_SELECTOR, "span[data-cy='jobDate']").text
        except:
            date = ""

        all_offers.append([titre, entreprise, ville, date])

    page += 1
    time.sleep(2)  # petit délai pour ne pas surcharger le site

# Sauvegarde dans un CSV
with open("offres_hellowork.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Titre", "Entreprise", "Ville", "Date"])  # en-tête
    writer.writerows(all_offers)

print(f"✅ Scraping terminé : {len(all_offers)} offres enregistrées dans 'offres_hellowork.csv'.")

driver.quit()
