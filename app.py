from flask import Flask, request, jsonify
from joblib import load
import os

app = Flask(__name__)

# Carregar o modelo treinado
model = load('elasticnet_model.joblib')

@app.route('/')
def index():
    return "Hello World!"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return "Send data as JSON in POST request to get a prediction."
    try:
        # Obter dados do JSON recebido
        data = request.get_json(force=True)
        
        # Converter os dados em um dataframe (ajuste isso conforme sua necessidade)
        # Aqui estou supondo que os dados vêm como um dicionário onde as chaves são os nomes das características e os valores são listas.
        import pandas as pd
        df = pd.DataFrame.from_dict(data)
        
        # Fazer previsão usando o modelo
        predictions = model.predict(df)
        
        # Retornar as previsões em formato JSON
        return jsonify(predictions.tolist())
    
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
