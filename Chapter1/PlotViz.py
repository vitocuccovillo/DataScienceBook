import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print("START MAIN")

    script_dir = os.path.dirname(__file__)
    data = pd.read_csv(os.path.join(script_dir,"data//Advertising.csv"), index_col=0)
    print(data.head())
    sns.pairplot(data, x_vars=['TV','radio','newspaper'], y_vars='sales')
    sns.pairplot(data, x_vars=['TV', 'radio', 'newspaper'], y_vars='sales', size=4.5, aspect=0.7)
    plt.show()