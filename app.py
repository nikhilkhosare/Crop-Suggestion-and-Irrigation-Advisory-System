from flask import Flask, render_template, jsonify
import serial
import time
import pandas as pd
# Import modules for ML model and evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize # For multi-class ROC
import numpy as np
import matplotlib.pyplot as plt # For generating the curve image
import threading
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

#SERIAL COMMUNICATION
# Adjust COM port for your system (Windows: COM3, Linux/Mac: /dev/ttyUSB0)
try:
    # Attempt connection (will proceed even if it fails, as shown in your logs)
    arduino = serial.Serial('COM6', 9600, timeout=1)
    time.sleep(2)
except serial.SerialException as e:
    print(f"Error connecting to Arduino on COM6: {e}. Check if the port is correct and available.")
    arduino = None


# ML MODEL TRAINING, EVALUATION, and ROC 
# Variables to hold models, initialized to None
crop_model = None
irrigation_model = None

try:
    data = pd.read_csv('realistic_crop_irrigation_dataset.csv')

    # CRITICAL FIX: Clean column names
    data.columns = data.columns.str.strip()

    # Define features (X) and targets (y)
    X = data[['Temperature', 'Humidity', 'SoilMoisture']]
    y_crop = data['Crop']
    y_irrigation = data['Irrigation_Needed'] 

    # --- 1. Split Data into Training and Testing Sets (80% Train, 20% Test) ---
    # Split for Crop Model
    X_train, X_test, y_crop_train, y_crop_test = train_test_split(
        X, y_crop, test_size=0.2, random_state=42
    )
    # Split for Irrigation Model
    _, _, y_irrigation_train, y_irrigation_test = train_test_split(
        X, y_irrigation, test_size=0.2, random_state=42
    )

    # --- 2. Train Models on Training Data ---
    crop_model = RandomForestClassifier(random_state=42)
    crop_model.fit(X_train, y_crop_train)

    irrigation_model = RandomForestClassifier(random_state=42)
    irrigation_model.fit(X_train, y_irrigation_train)

    # --- 3. Evaluate Models and Print Results ---
    
    print("\n" + "="*50)
    print("--- 🌾 Crop Prediction Model Evaluation ---")
    
    # Crop Model Accuracy & Report
    y_crop_pred = crop_model.predict(X_test)
    crop_accuracy = accuracy_score(y_crop_test, y_crop_pred)
    print(f"Accuracy on Test Set: {crop_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_crop_test, y_crop_pred, zero_division=0))

    # Crop Model AUC/ROC Calculation and Plotting (Multi-class via Micro-average)
    try:
        y_crop_score = crop_model.predict_proba(X_test)
        
        # Binarize the target labels for multi-class ROC (One-vs-Rest)
        classes_crop = np.unique(y_crop)
        y_crop_test_bin = label_binarize(y_crop_test, classes=classes_crop)
        
        # Calculate Micro-Averaged AUC Score
        crop_auc_micro = roc_auc_score(y_crop_test_bin, y_crop_score, average="micro")
        print(f"\nMicro-Averaged AUC-ROC Score: {crop_auc_micro:.4f}")

        # Plot ROC Curve (Micro-average)
        fpr, tpr, _ = roc_curve(y_crop_test_bin.ravel(), y_crop_score.ravel())
        plt.figure()
        plt.plot(fpr, tpr, label=f'Micro-average ROC (AUC = {crop_auc_micro:.4f})')
        plt.plot([0, 1], [0, 1], 'k--') # Diagonal 45 degree line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Crop Model ROC Curve (Micro-Average)')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve_crop.png')
        plt.close() # Close the figure to free memory
        print("ROC Curve image saved as: roc_curve_crop.png")

    except ValueError as e:
        print(f"\nCould not calculate AUC/ROC for Crop Model: {e}. Check data classes.")
    
    print("="*50)

    # --- 💧 Irrigation Prediction Model Evaluation ---
    print("\n" + "="*50)
    print("--- 💧 Irrigation Prediction Model Evaluation ---")
    
    # Irrigation Model Accuracy & Report
    y_irrigation_pred = irrigation_model.predict(X_test)
    irrigation_accuracy = accuracy_score(y_irrigation_test, y_irrigation_pred)
    print(f"Accuracy on Test Set: {irrigation_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_irrigation_test, y_irrigation_pred, zero_division=0))

    # Irrigation Model AUC/ROC Calculation and Plotting (Assuming Binary Classification)
    try:
        if len(np.unique(y_irrigation)) == 2:
            # 1. Map string labels to numeric (1 for 'Yes', 0 for 'No')
            y_test_numeric = y_irrigation_test.map({'Yes': 1, 'No': 0})
            
            # For binary classification, use probabilities of the positive class (index 1, which is 'Yes')
            y_irrigation_score = irrigation_model.predict_proba(X_test)[:, 1]
            
            # Use the numeric labels for roc_auc_score
            irrigation_auc = roc_auc_score(y_test_numeric, y_irrigation_score)
            print(f"\nAUC-ROC Score: {irrigation_auc:.4f}")

            # Use the numeric labels for the roc_curve function
            fpr, tpr, _ = roc_curve(y_test_numeric, y_irrigation_score)
            
            plt.figure()
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {irrigation_auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Irrigation Model ROC Curve')
            plt.legend(loc="lower right")
            plt.savefig('roc_curve_irrigation.png')
            plt.close()
            print("ROC Curve image saved as: roc_curve_irrigation.png")

        else:
             print("\nIrrigation model is multi-class, skipping detailed ROC plotting for brevity.")

    except Exception as e:
        print(f"\nCould not calculate AUC/ROC for Irrigation Model: {e}. Check data classes.")
    print("="*50)
    
except FileNotFoundError:
    print("Error: 'realistic_crop_irrigation_dataset.csv' not found. Please ensure it is in the same directory.")
    exit()
except KeyError as e:
    print(f"KeyError: {e}. Column name mismatch in CSV. Check column names.")
    exit()
#END MODIFIED SECTION 


#SENSOR DATA STORAGE 
sensor_data = {'temp': 0, 'hum': 0, 'soil': 0, 'crop': 'N/A', 'advice': 'N/A'}

#  READ ARDUINO DATA IN BACKGROUND 
def read_arduino():
    if arduino is None:
        print("Arduino connection failed, background thread not running.")
        return

    while True:
        try:
            line = arduino.readline().decode('utf-8').strip()
            if line:
                # Ensure the line has the expected number of values (3: temp, hum, soil)
                values = line.split(',')
                if len(values) == 3:
                    temp, hum, soil = map(float, values)
                    
                    sensor_data['temp'] = temp
                    sensor_data['hum'] = hum
                    sensor_data['soil'] = soil

                    # IMPROVED PREDICTION LOGIC: Using a DataFrame for robustness 
                    if crop_model and irrigation_model:
                        input_data_df = pd.DataFrame([[temp, hum, soil]], 
                                                     columns=['Temperature', 'Humidity', 'SoilMoisture'])
                        
                        # Predict the crop and irrigation advice using the trained models
                        sensor_data['crop'] = crop_model.predict(input_data_df)[0]
                        sensor_data['advice'] = irrigation_model.predict(input_data_df)[0]
                    # -END IMPROVEMENT 
                
        except ValueError:
            print(f"Non-numeric data received: {line}")
            continue
        except Exception as e:
            print(f"Error reading from Arduino: {e}")
            time.sleep(1)
            continue

# Start the Arduino reading thread only if connected
if arduino:
    thread = threading.Thread(target=read_arduino)
    thread.daemon = True
    thread.start()

#FLASK ROUTES 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def data():
    return jsonify(sensor_data)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
