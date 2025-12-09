import pandas as pd

pd.set_option("display.max_colwidth", None)  # ne pas couper les colonnes

df = pd.read_csv("offres_hellowork.csv", encoding="utf-8")
print(df.shape)
print(df.head(5))      # ou df.sample(10)
