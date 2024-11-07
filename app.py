from flask import Flask, request, jsonify, render_template
import torch
from utils import preprocess_transaction, load_model

app = Flask(__name__)

# Load the trained machine learning model
model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get transaction data from the form
    data = request.form.to_dict()
    transaction_data = preprocess_transaction(data)
    
    # Predict fraud or valid transaction
    result = predict_fraud(transaction_data)
    
    return jsonify(result)

def predict_fraud(transaction_data):
    # Convert the transaction data into a PyTorch tensor
    inputs = torch.tensor(transaction_data).float()
    
    # Set the model to evaluation mode
    model.eval()
    
    # Perform prediction
    with torch.no_grad():
        output = model(inputs)
    
    # Decision threshold (example: output > 0.5 indicates fraud)
    prediction = 'Fraud' if output.item() > 0.5 else 'Valid'
    
    return {"prediction": prediction, "confidence": output.item()}

if __name__ == '__main__':
    app.run(debug=True)
