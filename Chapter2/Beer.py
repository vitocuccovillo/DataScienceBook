import pandas as pd
import os

if __name__ == '__main__':
    script_dir = os.path.dirname(__file__)
    drinks = pd.read_csv(script_dir.replace("Chapter2","") + "data/drinks.csv")
    print(drinks.head())
    print(drinks.shape)

    for col in list(drinks):  # oppure list(drinks.columns.values)
        print("----------------------")
        print(drinks[col].describe())
