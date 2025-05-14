import csv, math
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU, LSTM
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

INPUT_TXT="input.txt"
MANUAL_INPUT_TXT="manual_input.txt"
INPUT_CSV=INPUT_TXT.replace(".txt", ".csv")
INTERPOLATE_GRADES=True
GENERATE_INPUT=False

def generate_grades_dataset(num_rows, num_grades_per_row=5, mean_grade=8.5, std_dev=0.4):
    dataset = []
    for _ in range(num_rows):
        num_grades_per_row = np.random.randint(3, 11)
        grades = np.random.normal(mean_grade, std_dev, num_grades_per_row)
        grades = np.clip(grades, 1.0, 10.0)  # Ensure grades stay between 1 and 10
        dataset.append(list(grades))

    # Writing to a text file
    with open(INPUT_TXT, "w") as file:
        for row in dataset:
            file.write(", ".join(f"{grade:.1f}" for grade in row) + "\n")

    print("Dataset generated and saved as ${INPUT_TXT}.")

def calculate_grade_evolution(row):
    percentage_changes = []
    
    for i in range(1, len(row)):  # Iterăm de la al doilea element
        change = (row[i] - row[i-1]) / row[i-1] * 100  # Calcul procentual
        percentage_changes.append(change)

    average_evolution = sum(percentage_changes) / len(percentage_changes)  # Media aritmetică
    return round(average_evolution / 100, 2)  # Rotunjim la 2 zecimale

def downsample(input_list, max_len):
    # Determine the interval size
    n = len(input_list)
    if n <= max_len:
        return input_list  # Return as is if no downsampling is needed

    # Calculate step size to select the elements for downsampling
    step = n / max_len

    # Create a new list for the downsampled values
    downsampled = []

    for i in range(max_len):
        start_idx = int(i * step)
        end_idx = int((i + 1) * step) if i + 1 < max_len else n  # Make sure the last segment goes to the end
        downsampled.append(np.mean(input_list[start_idx:end_idx]))  # Take mean of the interval

    return downsampled

# Funcția îmbunătățită pentru extinderea datelor conform metodei ratio
def expand_grades(row, max_length, interpolate_grades=True):
    ratio = max_length / len(row)  # Factor de replicare
    expanded_row = []
    
    mean_acceleration = calculate_grade_evolution(row)

    for grade in row:
        # Rotunjim în sus pentru consistență
        repeat_count = math.ceil(ratio)

        if interpolate_grades:
            for _ in range(repeat_count):
                # Apply exponential decay based on grade proximity to 1 or 10
                modifier = math.exp(-abs(grade - 5.5))  # Stronger change near midpoint
                adjusted_acceleration = mean_acceleration * modifier  # Decay applied
                
                new_grade = grade * (1 + adjusted_acceleration)  # Accelerated progression
                
                # Ensure the new grade stays within [1, 10]
                new_grade = max(1, min(10, round(new_grade, 2)))

                expanded_row.append(new_grade)  # Store the adjusted value
                grade = new_grade  # Update grade progressively
        else:
            expanded_row.extend([grade] * repeat_count)   

    # Ajustăm exact la max_length pentru a evita depășirea
    return expanded_row[:max_length]

def read_from_txt(manual_input):
    valid_data = []
    invalid_rows = []

    input_file = MANUAL_INPUT_TXT if manual_input else INPUT_TXT
    # Citim fișierul linie cu linie
    with open(input_file, "r") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split(",")
            if len(parts) >= 3:
                valid_data.append(list(map(float, parts)))
            else:
                invalid_rows.append(stripped)

    # Afișăm statistici
    invalid_count = len(invalid_rows)
    total_rows = len(valid_data) + invalid_count
    print(f"Total rows: {total_rows}")
    print(f"Invalid rows: {invalid_count}")
    print("List of invalid rows:")
    for row in invalid_rows:
        print(row)

    # Separăm input-urile de etichete (nota finală este ultima valoare)
    features = [row[:-1] for row in valid_data]
    labels = [row[-1] for row in valid_data]

    # Calculăm lungimea maximă a unui rând
    max_len = max(len(row) for row in features)

    # Aplicăm transformarea pe fiecare rând
    features_expanded = [expand_grades(row, max_len, interpolate_grades=INTERPOLATE_GRADES) for row in features]

    # Combinăm features cu etichetele
    data_expanded = [row + [label] for row, label in zip(features_expanded, labels)]

    # Salvăm datele extinse în fișierul CSV
    with open(INPUT_CSV, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data_expanded)

def read_from_csv():
    # Citim datele din fișier
    df = pd.read_csv("input.csv", delimiter=",", header=None)
    data = df.to_numpy()

    X = np.array(data)  # Input features (notele anterioare)
    y = np.mean(X, axis=1)  # Etiqueta: nota finală estimată (medie simplă ca baseline)

    return X, y

def evaluate_model(model, X_test, y_test):
    # Get predictions for test set
    y_pred = model.predict(X_test)

    # Compute metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    
    print("📊 Model Evaluation Metrics:")
    print(f"✅ MAE (Mean Absolute Error): {mae:.4f} (Ideal: <0.5) → Lower MAE means more accurate predictions.")
    print(f"✅ MSE (Mean Squared Error): {mse:.4f} (Ideal: Close to 0) → Penalizes larger errors more heavily.")
    print(f"✅ RMSE (Root Mean Squared Error): {rmse:.4f} (Ideal: <1) → Measures error magnitude; lower is better.")
    print(f"✅ R² Score: {r2:.4f} (Ideal: Close to 1) → Explains how well predictions match real values.")


def predict_grade(input_grades):
    model = keras.models.load_model("grade_predictor.keras")  # Load model

    # Convert input to NumPy array
    input_array = np.array([input_grades]).astype(np.float32)

    predicted_grade = model.predict(input_array)[0][0]
    
    # Ensure predictions stay within [1, 10]
    return max(1, min(10, round(predicted_grade, 2)))

# Wrap Keras model for scikit-learn compatibility
def build_model(input_shape, learning_rate=0.001, units=64, dropout_rate=0.3, l2_rate=0.01):
    model = keras.Sequential([
        # First LSTM layer with L2 regularization
        LSTM(units, return_sequences=True, input_shape=(input_shape, 1), 
             kernel_regularizer=l2(l2_rate)),
        Dropout(dropout_rate),

        # Second LSTM layer with reduced units and L2 regularization
        LSTM(units // 2, return_sequences=True, kernel_regularizer=l2(l2_rate)),
        Dropout(dropout_rate),

        # Third LSTM layer with further reduced units and L2 regularization
        LSTM(units // 4, kernel_regularizer=l2(l2_rate)),
        Dropout(dropout_rate),

        # Dense layers
        Dense(units, kernel_regularizer=l2(l2_rate)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),

        Dense(units // 2, kernel_regularizer=l2(l2_rate)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),

        Dense(1, activation='linear')  # Output predicted grade
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    # Learning rate scheduler to reduce learning rate when the validation loss plateaus
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    
    return model, lr_scheduler

def train_model(model, X_train, y_train, X_test, y_test, lr_scheduler):
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=16, verbose=1, callbacks=[lr_scheduler])
    model.save("grade_predictor.keras")

def build_and_evaluate_model(manual_input):
    read_from_txt(manual_input)
    X, y = read_from_csv()

    # Împărțim dataset-ul în seturi de antrenare și testare
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    param_grid = {
        "batch_size": [15, 10, 15, 25, 35],  # Different mini-batch sizes
        "epochs": [100, 150, 200, 250, 300],  # Number of training iterations
        "learning_rate": [0.0005, 0.001, 0.002, 0.005, 0.01],  # Fine-tuning step size
        "units": [8, 16, 32, 64, 128],  # Number of LSTM neurons per layer
        "dropout_rate": [0.1, 0.2, 0.25, 0.3, 0.4]  # Dropout percentage to prevent overfitting
    }

    model, lr_scheduler = build_model(X_train.shape[1])

    # Apelăm funcția de antrenare
    train_model(model, X_train, y_train, X_test, y_test, lr_scheduler)
    # Call evaluation after training
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    if (GENERATE_INPUT):
        generate_grades_dataset(100000)
    build_and_evaluate_model(not GENERATE_INPUT)