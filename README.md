\# AI-Powered Fraud Detection \& Risk Analytics Platform



\## Project Overview



The AI-Powered Fraud Detection \& Risk Analytics Platform is a machine learning application designed to identify potentially fraudulent financial transactions and assess their risk levels.



The platform uses machine learning models to analyze transaction data and provides a web interface for predictions, risk analysis, model performance, and transaction history.



\## Key Features



\- Fraud prediction using machine learning.

\- Transaction risk scores from 0 to 100.

\- Risk classification into Low, Medium, and High.

\- Comparison of Logistic Regression and Random Forest models.

\- Interactive web interface for transaction analysis.

\- Transaction history stored in an SQLite database.

\- Search and filter functionality for transaction history.

\- CSV export of transaction history.

\- PDF project report.

\- REST API built with FastAPI.

\- Automated API tests using pytest.



\## Technologies Used



\- \*\*Programming language:\*\* Python

\- \*\*Machine learning:\*\* Scikit-learn

\- \*\*Data processing:\*\* Pandas, NumPy

\- \*\*Data visualization:\*\* Matplotlib, Seaborn, Chart.js

\- \*\*Backend:\*\* FastAPI

\- \*\*Frontend:\*\* HTML, CSS, JavaScript

\- \*\*Database:\*\* SQLite

\- \*\*Testing:\*\* Pytest, HTTPX

\- \*\*API documentation:\*\* Swagger UI



\## Project Structure



```text

AI-Fraud-Detection/

├── api/

│   └── main.py

├── data/

│   └── processed/

├── database/

│   ├── db.py

│   └── fraud\_detection.db

├── frontend/

│   ├── index.html

│   ├── analysis.html

│   ├── analytics.html

│   ├── model\_performance.html

│   └── history.html

├── models/

├── reports/

├── src/

├── tests/

│   └── test\_api.py

├── README.md

└── requirements.txt

```



\## Machine Learning Models



The project compares two machine learning algorithms:



1\. Logistic Regression

2\. Random Forest Classifier



The models are evaluated using accuracy, precision, recall, and F1-score.



\## Risk Assessment



The application calculates a risk score between 0 and 100 and assigns a risk category.



\- \*\*Low Risk:\*\* Score below 30

\- \*\*Medium Risk:\*\* Score from 30 to below 70

\- \*\*High Risk:\*\* Score of 70 or above



The prediction threshold and risk bands are demonstration settings and should be calibrated and validated before real financial use.



\## How to Run the Project



\### 1. Activate the virtual environment



```powershell

.\\venv\\Scripts\\Activate.ps1

```



\### 2. Start the backend



From the project root, run:



```powershell

python -m uvicorn api.main:app --reload --port 8000

```



\### 3. Start the frontend



Open a second PowerShell terminal in the project root and run:



```powershell

python -m http.server 5500 --directory frontend

```



\### 4. Open the application



Visit:



\- Application: http://127.0.0.1:5500/

\- API documentation: http://127.0.0.1:8000/docs

\- API health check: http://127.0.0.1:8000/health



\## Run Automated Tests



From the project root, with the virtual environment activated, run:



```powershell

python -m pytest tests/test\_api.py -v

```



\## Dataset



The project uses a credit card transaction dataset containing transaction features and a class label indicating whether a transaction is fraudulent.



The dataset is used for experimentation and model evaluation. The application is an academic demonstration, not a production banking system.



\## Future Improvements



\- Add user authentication and role-based access.

\- Improve risk threshold calibration.

\- Add explainable AI to show why a transaction was flagged.

\- Monitor model performance and data drift.

\- Deploy the application to a cloud platform.

\- Add live alerts for high-risk transactions.



\## Disclaimer



This project is intended for educational and research purposes. Predictions are not guaranteed to be accurate and should not be used as the sole basis for real financial decisions.

