// Validate input fields
function validateForm() {
    const amount = document.getElementById("amount").value;
    const isCredit = document.getElementById("is_credit").value;
    const location = document.getElementById("location").value;
    const merchant = document.getElementById("merchant").value;

    if (!amount || isNaN(amount) || amount <= 0) {
        alert("Please enter a valid transaction amount.");
        return false;
    }
    if (!isCredit || (isCredit !== '0' && isCredit !== '1')) {
        alert("Please enter 1 for Credit or 0 for Debit.");
        return false;
    }
    if (!location || isNaN(location)) {
        alert("Please enter a valid location score.");
        return false;
    }
    if (!merchant || isNaN(merchant)) {
        alert("Please enter a valid merchant score.");
        return false;
    }
    return true;
}

// Submit form using Fetch API
async function submitForm() {
    // Validate the form inputs
    if (!validateForm()) {
        return;
    }

    // Show loading message
    document.getElementById("loading").style.display = "block";

    // Collecting form data
    const formData = new FormData(document.getElementById('fraudDetectionForm'));
    const data = {};
    formData.forEach((value, key) => {
        data[key] = value;
    });

    try {
        // Sending data to Flask backend using fetch
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        // Hide loading message
        document.getElementById("loading").style.display = "none";

        // Handling the response from Flask
        if (!response.ok) {
            throw new Error("Network response was not ok.");
        }

        const result = await response.json();

        const resultDiv = document.getElementById('result');
        if (result.is_fraud === 1) {
            resultDiv.innerHTML = '<p>Fraudulent Transaction Detected!</p>';
        } else {
            resultDiv.innerHTML = '<p>Transaction is Legitimate.</p>';
        }
    } catch (error) {
        // Hide loading message
        document.getElementById("loading").style.display = "none";
        // Display error message
        document.getElementById('result').innerHTML = '<p>Error processing the transaction. Please try again later.</p>';
        console.error("There was an error with the request:", error);
    }
}
