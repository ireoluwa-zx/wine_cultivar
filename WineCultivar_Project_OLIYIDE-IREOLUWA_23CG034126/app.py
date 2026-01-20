from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the pipeline (includes scaler + model)
model = joblib.load('model/wine_cultivar_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get inputs (Matching the 6 features we trained on)
            features = [
                float(request.form['alcohol']),
                float(request.form['malic_acid']),
                float(request.form['ash']),
                float(request.form['alcalinity_of_ash']),
                float(request.form['magnesium']),
                float(request.form['color_intensity'])
            ]
            
            # Convert to numpy array
            final_features = np.array([features])
            
            # Predict (The pipeline handles scaling automatically!)
            prediction = model.predict(final_features)
            
            # Map result (0, 1, 2) to Cultivar Names
            cultivar_map = {0: "Cultivar 1", 1: "Cultivar 2", 2: "Cultivar 3"}
            result = cultivar_map.get(prediction[0], "Unknown")

            return render_template('index.html', prediction_text=f'Predicted Origin: {result}')
        except Exception as e:
            return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)