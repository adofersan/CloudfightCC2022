# %%
# Matrix and plots
import pandas as pd
import numpy as np

# %%
df = pd.read_csv("training_data.csv", header=None)
df2=df.iloc[:,1].iloc[(df.iloc[:,0].str.len() ==30).values]

# %%
df2.value_counts()


