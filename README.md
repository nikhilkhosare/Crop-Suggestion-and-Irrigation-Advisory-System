# 🌾 Crop Suggestion and Irigation Advisory

A Flask-based IoT web app that uses **Arduino sensor data** and **Machine Learning (Random Forest)** to recommend suitable crops and irrigation schedules in real time.
The system integrates live temperature, humidity, and soil moisture readings for smart-farming insights.



## 🚀 Features

- 🌤️ Real-time data from Arduino sensors (Temperature, Humidity, Soil Moisture)
- 🌱 Crop recommendation based on live environmental data
- 💧 Irrigation advisory — when and how much to water
- 📊 Dashboard for monitoring sensor data and predictions
- 🤖 Random Forest model for accurate classification
- 📁 Training script to generate and update ML models
- 🔌 Serial communication between Arduino and Flask app
- 💻 Easy to deploy and extend for real farms or smart agriculture projects

---

## 🧠 Tech Stack

| Component | Technology Used |
|------------|-----------------|
| **Backend** | Flask (Python) |
| **Frontend** | HTML, CSS, JavaScript |
| **Machine Learning** | scikit-learn (Random Forest Classifier) |
| **Data Handling** | pandas |
| **IoT / Hardware** | Arduino (Serial Communication via pySerial) |

---

## 🗂️ Project Structure

Crop-Suggestion-And-Irrigation-Advisory/
│
├── app.py # Flask main backend file

├── templates/ # HTML templates (Flask frontend)
│ ├── index.html/
        CSS, JS, and image assets
├── data/ # Training and testing CSV files
├── requirements.txt # Dependencies list
├── README.md # Project documentation
└── .gitignore



## ⚙️ Installation and Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/Crop-Suggestion-And-Irrigation-Advisory.git
cd Crop-Suggestion-And-Irrigation-Advisory
2️⃣ Create Virtual Environment and Install Dependencies
Windows

bash
Copy code
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
macOS / Linux

bash
Copy code
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
3️⃣ Configure Serial Port
Edit app.py and set your correct Arduino serial port:

python
Copy code
arduino = serial.Serial('COM6', 9600, timeout=1)  # Windows
# or
arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)  # Linux/Mac
4️⃣ Run the Application
bash
Copy code
flask run
Now open your browser and visit:

cpp
Copy code
http://127.0.0.1:5000/
🧩 Model Training
Use train_model.py to train your Random Forest model:

bash
Copy code
python train_model.py
Make sure your dataset is located in the data/ folder with columns like:

Copy code
temperature, humidity, soil_moisture, soil_type, crop_label
The trained model will be saved as models/model.pkl.

📊 Example Workflow
Arduino collects real-time sensor readings.

Flask receives data through serial communication.

Data is passed to the Random Forest model.

Model predicts the best crop and irrigation requirement.

Results are displayed live on the web dashboard.

🧾 Example requirements.txt
ini
Copy code
Flask==2.3.3
pandas==2.1.1
scikit-learn==1.3.2
pyserial==3.5
joblib==1.3.2
Install all with:

bash
Copy code
pip install -r requirements.txt
🧰 Tools and Hardware Used
Arduino UNO / Nano with DHT11 (Temperature & Humidity)

Soil Moisture Sensor

USB Serial Connection

Python 3.10+

Flask Web Framework

🧑‍💻 Developer
👤 Name: Abhinay Somnath Sonavane
         Nikhil Khosare
         Sandip Randive
         Abhishek Bhoite
         
         
