# Development Task Checklist for AI Guardrail Benchmarking Tool


## Project Structure

```
AITEMOGuardRailComparisonTool/
│
├── app.py                      # Main Streamlit app entry point
├── requirements.txt            # Python dependencies
├── README.md
│
├── data/
│   └── (uploaded datasets)
│
├── modules/
│   ├── dataset_manager.py      # Dataset upload, selection, management
│   ├── guardrail_connector.py  # Model/API integration logic
│   ├── test_suite.py           # Test case definitions and execution
│   ├── metrics.py              # Metrics calculation
│   ├── criteria.py             # Custom pass/fail logic
│   ├── results.py              # Result storage, export, comparison
│   └── user_management.py      # (Optional) User login, roles, etc.
│
├── utils/
│   └── helpers.py              # Utility functions
│
├── reports/
│   └── (generated reports)
│
└── config/
		└── (saved configs)
```

## High-Level Streamlit App Layout

- **Sidebar:**
	- User login (optional)
	- Dataset upload/selection
	- Guardrail API/model configuration
	- Test suite selection/creation
	- Run configuration (batch/schedule)
	- Export/download options

- **Main Area:**
	- Dataset preview
	- Guardrail/model status
	- Test suite editor
	- Metrics and pass/fail criteria editor
	- Run/execute button
	- Results visualization (tables, charts)
	- Comparison dashboard
	- Download/export reports

## Core Modules & Functions

### Dataset Management
- Upload CSV/JSON
- List/select datasets
- Preview data
- Add new column to any dataset with a preset value
- Rename columns in any dataset
- Remove datasets
- Select which column to send to the guardrail (per dataset)
- Persist and remember user column selection

### Guardrail Integration
- Configure API endpoints/keys
- Select model/provider
- Test connection

### Test Suite Management
- Define test types (prompt injection, toxicity, jailbreak, etc.)
- Custom test creation UI
- Save/load test suites

### Metrics & Evaluation
- Accuracy, latency, robustness, etc.
- Custom metrics support

### Criteria Definition
- UI for pass/fail logic per test
- Save/load criteria

### Results & Visualization
- Store results (with config for reproducibility)
- Table/chart visualizations (Streamlit, Plotly, etc.)
- Export as CSV/JSON/PDF

### Comparison
- Select multiple runs for side-by-side comparison
- Highlight differences

### Automation
- Schedule runs (integrate with cron or Streamlit schedule)
- CI/CD integration hooks

### User Management (Optional)
- Login/signup
- Role-based access

## Development Task Checklist

- [x] Scaffold project directories and files
- [x] Implement basic Streamlit app entry point
- [x] Implement dataset management module
- [x] Implement guardrail/model integration module
- [x] Implement test suite management module
- [ ] Implement metrics & evaluation module
- [ ] Implement customizable criteria module
- [ ] Implement results storage/export/comparison module
- [ ] Implement user management module (optional)
- [x] Implement utility/helper functions
- [x] Add requirements.txt and README.md

Tick each task as it is completed.
