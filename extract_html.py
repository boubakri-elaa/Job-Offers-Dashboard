from selenium import webdriver
from selenium.webdriver.common.by import By
import time

driver = webdriver.Chrome()

url = "https://www.hellowork.com/fr-fr/emploi/recherche.html?k=python"
driver.get(url)

print("⏳ Chargement...")
time.sleep(5)

html = driver.page_source

with open("page_source.txt", "w", encoding="utf-8") as f:
    f.write(html)

print("✅ Fichier enregistré sous : page_source.txt")

driver.quit()
