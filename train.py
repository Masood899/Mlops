
import joblib
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

data = load_diabetes()
model = LinearRegression()
model.fit(data.data, data.target)

joblib.dump(model, 'diabetes_model.joblib')
print("Model trained and saved successfully!")
