# %%
# Matrix and plots
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
RANDOM_STATE = 2022

# %%
df = pd.read_csv("training_data.csv", header=None)
df=df.iloc[(df.iloc[:,0].str.len() ==30).values,:]

# %%
df = df[(df[1] == 0) | (df[1] ==1)]
X_train = df[0]
print(X_train.head())
y_train = df[1]
print(X_train.head())

# %%
X_train = pd.DataFrame(list(df.iloc[:,0].str)).T

# %%
X_train = X_train.applymap(ord)
X_train.columns = [str(i) for i in range(len(X_train.columns))]

# %%
df_test = pd.read_csv("simple_test_data.csv")
df_test = pd.DataFrame(list(df_test.iloc[:,0].str)).T
df_test = df_test.applymap(ord)
df_test.columns = [str(i) for i in range(len(df_test.columns))]
print(df_test.head())

# %%
xgb_params = {"n_estimators": np.arange(10, 210, step=10),
              "eta": np.arange(0.01, 0.3, step=0.01),
              "subsample": np.arange(0.5, 1, step=0.05),
              "colsample_bytree": np.arange(0.5, 1, step=0.05),
              "max_depth": np.arange(3, 10, step=1),
              "min_child_weight": np.arange(1, 5, step=0.05),
              "random_state": [RANDOM_STATE]}

xgb_models = RandomizedSearchCV(estimator=XGBClassifier(), n_jobs=2, param_distributions=xgb_params, n_iter=5,  verbose=1, cv=4,
                                scoring='accuracy', random_state=RANDOM_STATE)
xgb_models.fit(X_train, y_train)
print(xgb_models.best_score_)

# %%
result  =pd.DataFrame(xgb_models.best_estimator_.predict(df_test))
result.to_csv("p3.txt", index=False)


