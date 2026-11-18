# 🧠 AI Health & Fitness Predictor (ONNX Web Deployment)

> **🌐 Live Application:** **[https://<Shraddha12-sys>.github.io/<ml-onnx-web-app>/](https://shraddha12-sys.github.io/ml-onnx-web-app/)**

An interactive, serverless machine learning web application powered by **PyTorch**, **ONNX (`opset_version=18`)**, and **ONNX Runtime Web**. Both neural networks execute directly inside the user's browser with zero backend server requirements.

---

## 🚀 Deployed Models

### 1. 🔥 Calorie Expenditure Regression (`DLNet`)
* **File:** `calories_dl_model.onnx` (+ `calories_dl_model.onnx.data`)
* **Task:** Continuous regression estimating total kilocalories burned during a workout session.
* **Input Features (7):** Gender (`M/F`), Age, Height (`cm`), Weight (`kg`), Exercise Duration (`minutes`), Average Heart Rate (`bpm`), and Body Temperature (`°C`).
* **Architecture:** 4-layer feed-forward deep neural network (`7 → 64 → 32 → 16 → 1`) with ReLU activations and training-set Z-score standardization.
* **Performance:** Achieved **$R^2 > 0.999$** on the held-out test set, outperforming linear regression, shallow MLP, wide-and-deep residual networks, and XGBoost baselines.

### 2. ❤️ Cleveland Heart Disease Classification (`Heart_MLP`)
* **File:** `heart_disease_mlp_model.onnx` (+ `heart_disease_mlp_model.onnx.data`)
* **Task:** Binary classification screening for cardiovascular disease risk (`Healthy / Low Risk` vs. `Heart Disease Risk Detected`).
* **Input Features (13):** Age, Sex, Chest Pain Type (`cp`), Resting Blood Pressure (`trestbps`), Serum Cholesterol (`chol`), Fasting Blood Sugar (`fbs`), Resting ECG (`restecg`), Max Heart Rate (`thalach`), Exercise-Induced Angina (`exang`), ST Depression (`oldpeak`), Peak Exercise ST Slope (`slope`), Major Vessels Colored (`ca`), and Thalassemia (`thal`).
* **Architecture:** Regularized Multi-Layer Perceptron (`13 → 16 → 2`) with built-in Z-score normalization inside the forward pass, ReLU activation, Dropout (`p = 0.25`), and Softmax probability output.
* **Performance:** Achieved **~90.16% test accuracy** with high recall and minimal false negatives on the cleaned Cleveland Heart Disease dataset.

---

## 📁 Repository Structure

```text
├── index.html                          # Responsive frontend UI with model switcher & input guides
├── index.js                            # Client-side ONNX Runtime Web inference & scaling logic
├── calories_dl_model.onnx              # Exported Calorie Regression ONNX graph (opset 18)
├── calories_dl_model.onnx.data         # External tensor weights for Calorie model
├── heart_disease_mlp_model.onnx        # Exported Heart Disease Classification ONNX graph (opset 18)
├── heart_disease_mlp_model.onnx.data   # External tensor weights for Heart Disease model
└── README.md                           # Project documentation & live demo link