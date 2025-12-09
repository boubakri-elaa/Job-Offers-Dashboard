from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import time
import csv
import os


# Définir le chemin du CSV (seule modification)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)
CSV_PATH = os.path.join(RAW_DIR, "offres_hellowork.csv")


# Initialisation
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
wait = WebDriverWait(driver, 10)

base_url = "https://www.hellowork.com/fr-fr/emploi/recherche.html?k=&l=&d=all&page="

all_offers = []
page = 1
max_pages = 10  # ajuste selon ton besoin

while page <= max_pages:
    url = base_url + str(page)
    driver.get(url)
    print(f"⏳ Chargement de la page {page}...")

    cards = wait.until(
        EC.presence_of_all_elements_located(
            (By.CSS_SELECTOR, "div[data-cy='serpCard']")
        )
    )
    print("Nb cartes:", len(cards))

    for i, card in enumerate(cards):
        # 1) Titre : si pas trouvé, on ignore la carte
        try:
            titre = card.find_element(
                By.CSS_SELECTOR, "[data-cy='offerTitle']"
            ).text.strip()
        except NoSuchElementException:
            continue

        # 2) Entreprise
        try:
            entreprise = card.find_element(
                By.CSS_SELECTOR, "[data-cy='offerTitle'] p.tw-typo-s"
            ).text.strip()
        except NoSuchElementException:
            entreprise = ""

        # 3) Ville
        try:
            ville = card.find_element(
                By.CSS_SELECTOR, "[data-cy='localisationCard']"
            ).text.strip()
        except NoSuchElementException:
            ville = ""

        # 4) Contrat
        try:
            contrat = card.find_element(
                By.CSS_SELECTOR, "[data-cy='contractCard']"
            ).text.strip()
        except NoSuchElementException:
            contrat = ""

        # 5) Date
        try:
            date = card.find_element(
                By.CSS_SELECTOR, "div.tw-typo-s.tw-text-grey-500"
            ).text.strip()
        except NoSuchElementException:
            date = ""

        # si tout est vide, on n’enregistre pas
        if not any([titre, entreprise, ville, contrat, date]):
            continue

        print(f"OFFRE {i}:", titre, "|", entreprise, "|", ville, "|", contrat, "|", date)
        all_offers.append([titre, entreprise, ville, contrat, date])

    page += 1
    time.sleep(2)

# Sauvegarde CSV (même contenu, nouvel emplacement)
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Titre", "Entreprise", "Ville", "Contrat", "Date"])
    writer.writerows(all_offers)

print(f"✅ Scraping terminé : {len(all_offers)} offres enregistrées dans '{CSV_PATH}'.")
driver.quit()
