
## ⚠️ **This project was created with the help of GitHub Copilot.**

## Running with Docker

You can run this application using Docker:

1. Build the Docker image:

	```bash
	cd <into the git project>
	```

	```bash
	docker build -t guardrailbenchmark .
	```


2. Run the Docker container with persistent folders:

	```bash
	docker run -d \
		--name guardrailbenchmark \
		-p 8501:8501 \
		-v "$(pwd)/config":/app/config \
		-v "$(pwd)/data":/app/data \
		-v "$(pwd)/results":/app/results \
		guardrailbenchmark
	```

The app will be available at http://localhost:8501
# AI Guardrail Benchmarking Tool


A Streamlit-based tool to benchmark and compare AI guardrail providers.

## Setup Instructions

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd LLM-Guardrail-Benchmarking-tool
```

### 2. Create and activate a Python virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
streamlit run app.py
```

---

## How to Use the Application

### 1. Overview
- Read about the tool and its modules.

### 2. Dataset Management
- Upload CSV/JSON datasets.
- Preview, rename, label, and manage columns.
- Mark datasets as 'benign' or 'malicious'.
- Select which column to send to the guardrail.

### 3. Guardrail Integration
- Configure API endpoints, keys, and payloads for your guardrail/model.
- Test the connection manually.
- Save and manage guardrail configurations.

### 4. Test Suite Management
- Select datasets and a guardrail config to run batch tests.
- Monitor progress, ETA, and logs.
- Download prompt/reply logs after evaluation.

### 5. Results & Visualization
- Select a result file to view.
- See run metadata, confusion matrix, and full results log.

### 6. Comparison
- Select two result files to compare side-by-side.
- View metadata, metrics, confusion matrices, bar charts, and row-level differences.

### 7. Other Tabs
- Additional modules for automation and user management (if enabled).

---

## Tips
- Restart Streamlit after changing code in modules to see updates.
- All results are saved in the `results/` directory for reproducibility and comparison.
- For help, check the code comments or reach out to the project maintainer.

---

## Features
- Dataset management
- Guardrail/model integration
- Test suite management
- Metrics & evaluation
- Customizable criteria
- Result visualization
- Reproducibility
- Comparison
- Automation
- (Optional) User management
