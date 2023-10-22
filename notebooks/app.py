from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Carregar o modelo
model = joblib.load('elasticnet_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([data['features']])
    return jsonify(prediction=prediction[0])

if __name__ == '__main__':
    app.run(port=5000, debug=True)
