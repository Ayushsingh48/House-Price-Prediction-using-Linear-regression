import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv("housing.csv")
y = df["median_house_value"]
X = df.drop(columns="median_house_value")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

num_cols = [c for c in X.columns if c != "ocean_proximity"]
pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", OneHotEncoder(drop="first"), ["ocean_proximity"])])
model = Pipeline([("pre", pre), ("lr", LinearRegression())])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("R2  :", r2_score(y_test, y_pred))
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
joblib.dump(model, "house_price_lr_model.pkl")
