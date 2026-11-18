// Standardization parameters for the Calorie Regression dataset
// (Update these values with your exact `mean` and `std` arrays from Colab for 100% precision)
const CALORIES_MEAN = [0.50, 42.78, 174.47, 74.96, 15.53, 95.51, 40.02];
const CALORIES_STD  = [0.50, 16.98, 14.25, 15.03, 8.31, 9.58, 0.78];

// Toggle between the Calories and Heart Disease UI tabs
function switchModel(modelType) {
    document.getElementById('btn-calories').classList.remove('active');
    document.getElementById('btn-heart').classList.remove('active');
    document.getElementById('form-calories').classList.remove('active');
    document.getElementById('form-heart').classList.remove('active');
    document.getElementById('result-box').style.display = 'none';

    if (modelType === 'calories') {
        document.getElementById('btn-calories').classList.add('active');
        document.getElementById('form-calories').classList.add('active');
    } else {
        document.getElementById('btn-heart').classList.add('active');
        document.getElementById('form-heart').classList.add('active');
    }
}

// 1. Run Calorie Expenditure Regression ONNX Model
async function runCaloriesONNX(event) {
    event.preventDefault();
    const btn = document.getElementById('btn-submit-calories');
    btn.disabled = true;
    btn.innerText = "Running ONNX Inference...";

    try {
        // Gather the 7 raw inputs in exact training column order
        const rawInputs = [
            parseFloat(document.getElementById('c_gender').value),
            parseFloat(document.getElementById('c_age').value),
            parseFloat(document.getElementById('c_height').value),
            parseFloat(document.getElementById('c_weight').value),
            parseFloat(document.getElementById('c_duration').value),
            parseFloat(document.getElementById('c_hr').value),
            parseFloat(document.getElementById('c_temp').value)
        ];

        // Apply Z-score standardization matching Python preprocessing
        const scaledInputs = new Float32Array(7);
        for (let i = 0; i < 7; i++) {
            scaledInputs[i] = (rawInputs[i] - CALORIES_MEAN[i]) / CALORIES_STD[i];
        }

        const tensorX = new ort.Tensor('float32', scaledInputs, [1, 7]);

        // Load ONNX model along with its external .data weights file
        const session = await ort.InferenceSession.create("./calories_dl_model.onnx", {
            externalData: [
                {
                    data: "./calories_dl_model.onnx.data",
                    path: "calories_dl_model.onnx.data"
                }
            ]
        });

        // Dynamically grab input/output node names
        const inputName = session.inputNames[0];
        const outputName = session.outputNames[0];

        const results = await session.run({ [inputName]: tensorX });
        const predictedCalories = results[outputName].data[0];

        // Display result
        const resultBox = document.getElementById('result-box');
        document.getElementById('result-title').innerText = "Estimated Calorie Expenditure (ONNX Output)";
        document.getElementById('result-text').innerHTML = `<span style="color: #2563eb;">${Math.max(0, predictedCalories).toFixed(2)} kcal</span>`;
        document.getElementById('result-subtext').innerText = `Model: calories_dl_model.onnx | Session Duration: ${rawInputs[4]} mins`;
        resultBox.style.display = 'block';

    } catch (e) {
        console.error("Calories ONNX Error:", e);
        alert("Failed to run Calories ONNX model: " + e.message);
    } finally {
        btn.disabled = false;
        btn.innerText = "Run Calorie ONNX Model";
    }
}

// 2. Run Cleveland Heart Disease Classification ONNX Model
async function runHeartONNX(event) {
    event.preventDefault();
    const btn = document.getElementById('btn-submit-heart');
    btn.disabled = true;
    btn.innerText = "Running ONNX Inference...";

    try {
        // Gather the 13 raw clinical inputs (Heart_MLP standardizes internally!)
        const rawInputs = new Float32Array([
            parseFloat(document.getElementById('h_age').value),
            parseFloat(document.getElementById('h_sex').value),
            parseFloat(document.getElementById('h_cp').value),
            parseFloat(document.getElementById('h_trestbps').value),
            parseFloat(document.getElementById('h_chol').value),
            parseFloat(document.getElementById('h_fbs').value),
            parseFloat(document.getElementById('h_restecg').value),
            parseFloat(document.getElementById('h_thalach').value),
            parseFloat(document.getElementById('h_exang').value),
            parseFloat(document.getElementById('h_oldpeak').value),
            parseFloat(document.getElementById('h_slope').value),
            parseFloat(document.getElementById('h_ca').value),
            parseFloat(document.getElementById('h_thal').value)
        ]);

        const tensorX = new ort.Tensor('float32', rawInputs, [1, 13]);

        // Load ONNX model along with its external .data weights file
        const session = await ort.InferenceSession.create("./heart_disease_mlp_model.onnx", {
            externalData: [
                {
                    data: "./heart_disease_mlp_model.onnx.data",
                    path: "heart_disease_mlp_model.onnx.data"
                }
            ]
        });

        const inputName = session.inputNames[0];
        const outputName = session.outputNames[0];

        const results = await session.run({ [inputName]: tensorX });
        const probabilities = results[outputName].data; // Softmax probabilities: [prob_healthy, prob_disease]

        const probHealthy = (probabilities[0] * 100).toFixed(2);
        const probDisease = (probabilities[1] * 100).toFixed(2);

        const resultBox = document.getElementById('result-box');
        document.getElementById('result-title').innerText = "Cleveland Heart Disease Classification (ONNX Output)";

        if (probabilities[1] > probabilities[0]) {
            document.getElementById('result-text').innerHTML = `<span style="color: #dc2626;">Positive: Heart Disease Risk Detected</span>`;
        } else {
            document.getElementById('result-text').innerHTML = `<span style="color: #16a34a;">Negative: Low Risk / Healthy Profile</span>`;
        }

        document.getElementById('result-subtext').innerText = `Model Confidence — Healthy: ${probHealthy}% | Heart Disease Risk: ${probDisease}%`;
        resultBox.style.display = 'block';

    } catch (e) {
        console.error("Heart Disease ONNX Error:", e);
        alert("Failed to run Heart Disease ONNX model: " + e.message);
    } finally {
        btn.disabled = false;
        btn.innerText = "Run Heart Disease ONNX Model";
    }
}