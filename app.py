from flask import Flask, render_template, request
import pickle
import pandas as pd
from diet_logic import diet_plan

# create flask application
app = Flask(__name__)

# load trained machine learning model
model = pickle.load(open('diet_model.pkl', 'rb'))


# ---------------- HOME PAGE ----------------
@app.route('/')
def home():
    return render_template('index.html')


# ---------------- PREDICTION PAGE ----------------
@app.route('/predict', methods=['POST'])
def predict():

    age = int(request.form['age'])
    gender = request.form['gender']
    height = float(request.form['height'])
    weight = float(request.form['weight'])
    father = float(request.form['father'])
    mother = float(request.form['mother'])
    activity = int(request.form['activity'])
    diabetes = int(request.form['diabetes'])

    input_dict = {
        'age': age,
        'gender': gender,
        'height': height,
        'weight': weight,
        'father_weight': father,
        'mother_weight': mother,
        'activity': activity,
        'diabetes_family': diabetes
    }

    input_data = pd.DataFrame([input_dict])

    # Apply same encoding as training
    input_data = pd.get_dummies(input_data)

    # Load model columns
    model_columns = model.feature_names_in_

    # Add missing columns
    for col in model_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    
    # Ensure correct column order
    input_data = input_data[model_columns]

    # Calculate BMI
    height_m = height / 100
    bmi = weight / (height_m * height_m)
    bmi = round(bmi, 2)
    
    # Calorie calculation (BMR formula)
    if gender == "M":
        bmr = 10*weight + 6.25*height - 5*age + 5
    else:
        bmr = 10*weight + 6.25*height - 5*age - 161

    # activity multiplier
    if activity == 0:
        calories = bmr * 1.2
    elif activity == 1:
        calories = bmr * 1.55
    else:
        calories = bmr * 1.9

    calories = round(calories,2)

    prediction = model.predict(input_data)[0]

    diet = diet_plan(prediction)

    return render_template('result.html', result=prediction, diet=diet, bmi=bmi, calories=calories)

# run the program
if __name__== '__main__':
    app.run(debug=True)