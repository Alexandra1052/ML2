from model_predictor import predict_grade

# Define five test cases with varied grade patterns
test_cases = [
    [6.5, 7.0, 7.5, 8.0, 8.5, 9.0],  # Increasing trend
    [9.0, 8.5, 8.0, 7.5, 7.0, 6.5],  # Decreasing trend
    [7.5, 8.0, 6.5, 7.0, 8.5, 7.8],  # Fluctuating
    [9.5, 9.8, 9.7, 9.9, 9.6, 9.8],  # High performer
    [1.5, 2.0, 2.5, 3.0, 2.8, 3.2],  # Low performer
]

# Run predictions for all test cases
for i, grades in enumerate(test_cases, 1):
    predicted_grade = predict_grade(grades)
    print(f"Test {i} - Input Grades: {grades}, Predicted Final Grade: {predicted_grade:.2f}")
