import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
import os
import json

app = Flask(__name__, static_folder='static')

def categorize_data(file_path, health_condition):
    dataset = pd.read_csv(file_path, skiprows=3, header=None)
    dataset = dataset.rename(columns={0: 'heart_sensor', 1: 'sweat_sensor'})

    if health_condition == 'Normal':
        categories = {
            'GOOD': {'heart_sensor': (60, 85), 'sweat_sensor': (-10, 2)},
            'AVERAGE': {'heart_sensor': (85, 100), 'sweat_sensor': (2, 4)},
            'BAD': {'heart_sensor': (100, 150), 'sweat_sensor': (4, 20)}
        }
    elif health_condition == 'High BP':
        categories = {
            'GOOD': {'heart_sensor': (85, 100), 'sweat_sensor': (-10, 3)},
            'AVERAGE': {'heart_sensor': (100, 112), 'sweat_sensor': (3, 6)},
            'LESS': {'heart_sensor': (112, 150), 'sweat_sensor': (6, 20)}
        }
    elif health_condition == 'Low BP':
        categories = {
            'GOOD': {'heart_sensor': (55, 78), 'sweat_sensor': (-10, 1)},
            'AVERAGE': {'heart_sensor': (78, 90), 'sweat_sensor': (1, 3)},
            'LESS': {'heart_sensor': (90, 150), 'sweat_sensor': (3, 20)}
        }
    else:
        raise ValueError("Invalid health condition")

    unknown_count = 0  # Counter to keep track of the 'UNKNOWN' category count

    for index, row in dataset.iterrows():
        heart_value = row['heart_sensor']
        sweat_value = row['sweat_sensor']

        for category, ranges in categories.items():
            heart_range = ranges['heart_sensor']
            sweat_range = ranges['sweat_sensor']

            if heart_range[0] <= heart_value <= heart_range[1] and sweat_range[0] <= sweat_value <= sweat_range[1]:
                dataset.loc[index, 'confidence_category'] = category
                break
        else:
            dataset.loc[index, 'confidence_category'] = 'UNKNOWN'
            unknown_count += 1

    category_counts = dataset['confidence_category'].value_counts()
    total_rows = dataset.shape[0]

    category_percentages = {}
    for category, count in category_counts.items():
        if category == 'UNKNOWN':
            continue  # Skip processing 'UNKNOWN' category as we'll handle it separately
        percentage = (count / total_rows) * 100
        category_percentages[category] = round(percentage, 2)

    # Calculate the percentage of 'Above Average' and 'Below Average' categories from 'UNKNOWN'
    unknown_percentage = (unknown_count / total_rows) * 100
    above_average_percentage = (unknown_percentage * 0.45) 
    below_average_percentage = (unknown_percentage * 0.20) 
    unpredictable = (unknown_percentage * 0.35)

    # Add 'Above Average' and 'Below Average' categories to the result
    category_percentages['Above Average'] = round(above_average_percentage, 2)
    category_percentages['Below Average'] = round(below_average_percentage, 2)
    category_percentages['unpredictable'] = round(unpredictable, 2)

    return category_percentages

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    health_condition = request.form.get('health_condition')

    if file.filename.endswith('.csv'):
        file.save('temp.csv')
        category_percentages = categorize_data('temp.csv', health_condition)
        os.remove('temp.csv')

        confidence_score = category_percentages.get('GOOD', 0) + category_percentages.get('Above Average', 0) + category_percentages.get('AVERAGE', 0)
        stress_score = (category_percentages.get('Below Average', 0) + category_percentages.get('BAD', 0) + category_percentages.get('unpredictable', 0))

        return render_template('result.html', confidence_score=confidence_score, stress_score=stress_score)
    else:
        return "Unsupported file format"

if __name__ == '__main__':
    app.run(debug=True)














