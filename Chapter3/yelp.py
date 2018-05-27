import pandas as pd
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    script_dir = os.path.dirname(__file__)
    yelp_raw = pd.read_csv(script_dir.replace("Chapter3","") + "data/yelp.csv")
    print(yelp_raw.head())

    for col in list(yelp_raw):
        print("-------------")
        print(yelp_raw[col].describe())

    ratings = yelp_raw['stars'].value_counts()
    ratings.sort_index(ascending=True, inplace=True)
    print(ratings)
    ratings.plot(kind='bar')
    plt.show()