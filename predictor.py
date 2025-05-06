import math
import pandas as pd
from model_predictor import predict_grade, expand_grades, downsample

INTERPOLATE_GRADES=True

# Define five test cases with varied grade patterns
test_cases = [
    [6.5, 7.0, 7.5, 8.0, 8.5, 9.0],  # Increasing trend
    [9.0, 8.5, 8.0, 7.5, 7.0, 6.5],  # Decreasing trend
    [7.5, 8.0, 6.5, 7.0, 8.5, 7.8],  # Fluctuating
    [9.5, 9.8, 9.7, 9.9, 9.6, 9.8],  # High performer
    [1.5, 2.0, 2.5, 3.0, 2.8, 3.2],  # Low performer
    [7.8, 8.2],
    [8],
    [7.8, 7.85, 7.75, 8.2, 8.3, 8.25, 8.45, 8.5, 8.8, 8.7, 8.75, 9, 9.1, 8.95, 9.15, 9.25, 9.4, 9.45, 9.35]
]

df = pd.read_csv("input.csv", delimiter=",", header=None)
num_columns_per_row = df.apply(lambda row: len(row), axis=1) - 1
mean_grades = num_columns_per_row.mean()
mean_grades = math.ceil(mean_grades)

# Run predictions for all test cases
for i, grades in enumerate(test_cases, 1):
    if len(grades) < 2:
        print(f"Test {i} - Input Grades: {grades} is too small. Need at least 2 grades for prediction")
        continue
        
    if len(grades) < mean_grades:
        new_grades = expand_grades(grades, mean_grades, interpolate_grades=INTERPOLATE_GRADES)
    elif len(grades) > mean_grades:
        new_grades = downsample(grades, mean_grades)
    else:
        new_grades = grades

    predicted_grade = predict_grade(new_grades)
    
    # Format to two decimal places using f-strings
    formatted_grades = [f"{grade:.2f}" for grade in new_grades]
    print(f"Test {i} - Input Grades: {grades}, transformed to {formatted_grades}, Predicted Final Grade: {predicted_grade:.2f}")