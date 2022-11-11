# %%
# Matrix and plots
import pandas as pd
import numpy as np

# %%
df = pd.read_csv("training_data.csv", header=None)

# %%
df.iloc[:,1].value_counts()


