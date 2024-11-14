import sklearn 
from sklearn.datasets import load_iris


print(load_iris())
print(load_iris(return_X_y=True))
x,y = load_iris(return_X_y=True)
print(x,y)


from sklearn.linear_model import LinearRegression
Model = LinearRegression()
print(Model.fit(x,y))
print(Model.predict(x))


from sklearn.neighbors import KNeighborsRegressor
mod = KNeighborsRegressor()
print(mod.fit(x,y))
print(mod.predict(x))


import matplotlib.pyplot as plt
pred = mod.predict(x)
plt.scatter(pred,y)
plt.show()


import pandas as pd
from sklearn.datasets import fetch_openml
df = fetch_openml('titanic',version=1,as_frame=True)['data']
print(df.info())
print(df.isnull())
print(df.isnull().sum())


import seaborn as sns
sns.set()
miss_val_per = pd.DataFrame((df.isnull().sum() / len(df)) * 100)
print(miss_val_per.plot(kind='bar',title="Missing values in percentage",ylabel="percentage"))
plt.show()

print(f'size of the dataset:{df.shape}')

df.drop(['body'],axis=1,inplace=True)
print(f"size of the dataset after droping a feature:{df.shape}")

from sklearn.impute import SimpleImputer
print(f'Number of null values before imputing:{df.age.isnull().sum()}')

imp = SimpleImputer(strategy='mean')
df['age'] = imp.fit_transform(df[['age']])
print(f'Number')