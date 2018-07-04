import statsmodels.api as sm
import seaborn as sb
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score

affairs_df = sm.datasets.fair.load_pandas().data
print(affairs_df.head())

# creo un nuovo campo nel dataframe, variabile categorica
affairs_df['affair_binary'] = (affairs_df['affairs'] > 0)

sb.heatmap(affairs_df.corr())
plt.show()

affairs_X = affairs_df.drop(['affairs', 'affair_binary'], axis= 1)
affairs_y = affairs_df['affair_binary']

model = DecisionTreeClassifier()
scores = cross_val_score(model, affairs_X, affairs_y, cv= 10)

print(scores.mean())

