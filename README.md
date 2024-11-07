# Wallet Shield - Transaction Fraud Detection and Transaction Safety with Machine Learning

## Project Description

This is a Python-based fintech application that integrates machine learning algorithms to detect fraudulent transactions. The application ensures data security by implementing secure hashing techniques such as SHA-256 for maintaining data integrity.

- **Machine Learning Integration**: The project leverages PyTorch to develop a machine learning model for detecting fraud in real-time transaction data.
- **Secure Data Handling**: Sensitive financial information is processed and encrypted using SHA-256 hashing techniques, ensuring the integrity and confidentiality of transaction data.
- **Problem-Solving Focus**: The application employs problem-solving skills to design a robust transaction verification system, accurately identifying fraudulent activities and maintaining financial safety.
- **Real-Time Transaction Analysis**: The system allows for real-time analysis of transaction data, providing predictions on whether a transaction is legitimate or fraudulent.

## Features
- **Machine Learning Model**: Utilizes PyTorch for detecting fraudulent transactions based on transaction attributes such as amount, location, merchant, and transaction type (credit/debit).
- **Data Integrity**: Uses SHA-256 for hashing transaction details to maintain data integrity and prevent tampering. Data can be entered either through Web Interface or through integrating PANDAS directly to application.
- **Web Interface**: The application provides a simple web form (no css design) where users can submit transaction data to check if the transaction is fraudulent or legitimate.
- **Flask Backend**: Flask serves as the backend framework to handle requests and run the machine learning model.
- **Pandas**: Used to optionally deal with large sums of user data directly with model.py

## Technologies Used
- **Python**: Programming language used for the application.
- **Flask**: Web framework used for serving the frontend and backend.
- **PyTorch**: Deep learning framework used for building the machine learning model.
- **SHA-256**: Cryptographic hashing technique used for data integrity.
- **HTML/CSS/JavaScript**: Frontend technologies used to build the user interface.

## Clone and Run the Project Locally

Follow the steps below to clone and run this project on your local machine:

### 1. Clone the Repository

Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/your-username/transaction-fraud-detection.git
Sure! Below is the entire `README.md` template with all the instructions and project details included. You can copy and paste this directly into your `README.md` file.

```markdown
# Transaction Fraud Detection with Machine Learning

## Project Description

This is a Python-based fintech application that integrates machine learning algorithms to detect fraudulent transactions. The application ensures data security by implementing secure hashing techniques such as SHA-256 for maintaining data integrity.

- **Machine Learning Integration**: The project leverages PyTorch to develop a machine learning model for detecting fraud in real-time transaction data.
- **Secure Data Handling**: Sensitive financial information is processed and encrypted using SHA-256 hashing techniques, ensuring the integrity and confidentiality of transaction data.
- **Problem-Solving Focus**: The application employs problem-solving skills to design a robust transaction verification system, accurately identifying fraudulent activities and maintaining financial safety.
- **Real-Time Transaction Analysis**: The system allows for real-time analysis of transaction data, providing predictions on whether a transaction is legitimate or fraudulent.

## Features
- **Machine Learning Model**: Utilizes PyTorch for detecting fraudulent transactions based on transaction attributes such as amount, location, merchant, and transaction type (credit/debit).
- **Data Integrity**: Uses SHA-256 for hashing transaction details to maintain data integrity and prevent tampering.
- **Web Interface**: The application provides a simple web form where users can submit transaction data to check if the transaction is fraudulent or legitimate.
- **Flask Backend**: Flask serves as the backend framework to handle requests and run the machine learning model.

## Technologies Used
- **Python**: Programming language used for the application.
- **Flask**: Web framework used for serving the frontend and backend.
- **PyTorch**: Deep learning framework used for building the machine learning model.
- **SHA-256**: Cryptographic hashing technique used for data integrity.
- **HTML/CSS/JavaScript**: Frontend technologies used to build the user interface.

## Clone and Run the Project Locally

Follow the steps below to clone and run this project on your local machine:

### 1. Clone the Repository

Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/your-username/transaction-fraud-detection.git
```

Replace `your-username` with your actual GitHub username.

### 2. Navigate to the Project Directory

Change into the project directory:

```bash
cd transaction-fraud-detection
```

### 3. Set Up a Virtual Environment (Optional but Recommended)

It's a good practice to use a virtual environment to manage dependencies. You can create a virtual environment with:

```bash
python3 -m venv venv
```

Activate the virtual environment:

- On Windows:

    ```bash
    venv\Scripts\activate
    ```

- On macOS/Linux:

    ```bash
    source venv/bin/activate
    ```

### 4. Install Dependencies

Install the required dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

If `requirements.txt` does not exist, install the following manually:

```bash
pip install Flask torch
```

You may also need to install other dependencies for your specific system, such as PyTorch-specific versions for CUDA.

### 5. Run the Application

Once the dependencies are installed, you can run the Flask application with the following command:

```bash
python app.py
```

This will start a development server at `http://localhost:5000`.

### 6. Access the Application

Open your browser and navigate to `http://localhost:5000` to access the transaction fraud detection form.

You can now submit transaction data (amount, type, merchant score, location score, etc.) to check if the transaction is detected as fraudulent or legitimate.

### File Descriptions:

- **`app.py`**: The main Flask backend file that handles the routes and requests.
- **`model.py`**: Contains the PyTorch model definition for fraud detection.
- **`/templates/index.html`**: HTML file that provides a user interface for entering transaction details.
- **`/static/js/script.js`**: JavaScript file for form validation and handling asynchronous form submissions via fetch API.

## Model Description

The machine learning model in `model.py` is built using PyTorch and trained on a dataset of transactions with labeled fraud information. The model takes various features such as transaction amount, type (credit/debit), location score, and merchant score as input, and predicts whether the transaction is fraudulent.

The model is loaded into memory when the application starts and used to predict the likelihood of fraud for incoming transaction data.