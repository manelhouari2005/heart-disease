document.getElementById('prediction-form').addEventListener('submit', function(e) {
    e.preventDefault();

    const formData = new FormData(e.target);
    const data = {};
    for (let [key, value] of formData.entries()) {
        data[key] = value;
    }

    // Convert to numbers
    data.age = parseInt(data.age);
    data.sex = parseInt(data.sex);
    data.cp = parseInt(data.cp);
    data.trestbps = parseInt(data.trestbps);
    data.chol = parseInt(data.chol);
    data.fbs = parseInt(data.fbs);
    data.restecg = parseInt(data.restecg);
    data.thalach = parseInt(data.thalach);
    data.exang = parseInt(data.exang);
    data.oldpeak = parseFloat(data.oldpeak);
    data.slope = parseInt(data.slope);
    data.ca = parseInt(data.ca);
    data.thal = parseInt(data.thal);

    fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        })
        .then(response => response.json())
        .then(result => {
            const resultDiv = document.getElementById('result');
            const predictionText = document.getElementById('prediction-text');
            const modelUsed = document.getElementById('model-used');

            if (result.prediction === 1) {
                resultDiv.className = 'result danger';
                predictionText.textContent = `⚠️ The model predicts Heart Disease (Probability: ${result.probability})`;
            } else {
                resultDiv.className = 'result success';
                predictionText.textContent = `✅ The model predicts No Heart Disease (Probability: ${result.probability})`;
            }

            modelUsed.textContent = `Model used: ${data.model}`;
            resultDiv.classList.remove('hidden');
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred. Please try again.');
        });
});