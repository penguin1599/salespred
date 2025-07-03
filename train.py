import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Load dataset
DATA_PATH = "data/data.csv"  # make sure the CSV lives here

data = pd.read_csv(DATA_PATH)
X = data[["TV", "Radio", "Newspaper"]]
y = data["Sales"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Persist model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved to model.pkl")