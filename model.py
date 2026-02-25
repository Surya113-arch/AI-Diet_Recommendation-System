import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

# 1. Load dataset
data = pd.read_csv(r"C:\Users\SURYA TEJA\OneDrive\Desktop\diet_recommendation_system\dataset.csv")

# 2. Select input features
X = data[['age','gender','height','weight','father_weight','mother_weight','activity','diabetes_family']]

# 3. Target output
y = data['bmi_category']

# 4. Convert text to numbers (Male/Female)
X = pd.get_dummies(X)

# 5. Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 6. Train ML model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

# 7. Save trained model as pickle file
pickle.dump(model, open("diet_model.pkl", "wb"))

print("Model trained and pickle file created successfully!")

print("Model Accuracy:", accuracy)
