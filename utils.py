def preprocess_transaction(data):
    # Example preprocessing: convert input data into a numeric array
    # (Assuming data is a dictionary with numeric values as strings)
    transaction_data = [
        float(data.get("amount", 0)),  # Amount of the transaction
        int(data.get("is_credit", 0)),  # 1 if it's a credit transaction, 0 if debit
        float(data.get("location", 0)),  # Location score or indicator (e.g., distance from usual location)
        float(data.get("merchant", 0)),  # Merchant score (e.g., known fraud risk)
        # Add other features as needed...
    ]
    
    # Padding the data to match model's input size
    while len(transaction_data) < 10:
        transaction_data.append(0.0)  # Padding with zeros if needed
    return transaction_data
