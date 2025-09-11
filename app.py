# Streamlit entry point for AI Guardrail Benchmarking Tool
import streamlit as st
import os
import json

st.set_page_config(page_title="AI Guardrail Benchmarking Tool", layout="wide")
st.title("AI Guardrail Benchmarking Tool")

# Tabs for main modules
tab_names = [
	"Overview",
	"Dataset Management",
	"Guardrail Integration",
	"Test Suite Management",
	"Metrics & Evaluation",
	"Customizable Criteria",
	"Results & Visualization",
	"Comparison",
	"Automation",
	"User Management (Optional)"
]
tabs = st.tabs(tab_names)


# --- Overview Tab ---
with tabs[0]:
	st.markdown("""
	Welcome! This tool helps you benchmark and compare AI guardrail providers. Use the tabs above to navigate between modules.
	""")
	modules = [
		{
			"name": "Dataset Management",
			"description": "Upload, select, and manage datasets for testing.",
			"features": [
				"Upload CSV/JSON",
				"List/select datasets",
				"Preview data"
			]
		},
		{
			"name": "Guardrail Integration",
			"description": "Connect to models or APIs (your 'guardrail').",
			"features": [
				"Configure API endpoints/keys",
				"Select model/provider",
				"Test connection"
			]
		},
		{
			"name": "Test Suite Management",
			"description": "Define and run various test cases or scenarios.",
			"features": [
				"Define test types (prompt injection, toxicity, jailbreak, etc.)",
				"Custom test creation UI",
				"Save/load test suites"
			]
		},
		{
			"name": "Metrics & Evaluation",
			"description": "Measure accuracy, latency, robustness, and other relevant metrics.",
			"features": [
				"Accuracy, latency, robustness, etc.",
				"Custom metrics support"
			]
		},
		{
			"name": "Customizable Criteria",
			"description": "Let users define what 'success' or 'failure' means for their use case.",
			"features": [
				"UI for pass/fail logic per test",
				"Save/load criteria"
			]
		},
		{
			"name": "Results & Visualization",
			"description": "Show results with tables, charts, and downloadable reports.",
			"features": [
				"Store results (with config for reproducibility)",
				"Table/chart visualizations",
				"Export as CSV/JSON/PDF"
			]
		},
		{
			"name": "Comparison",
			"description": "Compare results across different models, datasets, or guardrail configurations.",
			"features": [
				"Select multiple runs for side-by-side comparison",
				"Highlight differences"
			]
		},
		{
			"name": "Automation",
			"description": "Schedule or batch runs, and possibly integrate with CI/CD.",
			"features": [
				"Schedule runs (integrate with cron or Streamlit schedule)",
				"CI/CD integration hooks"
			]
		},
		{
			"name": "User Management (Optional)",
			"description": "For multi-user environments.",
			"features": [
				"Login/signup",
				"Role-based access"
			]
		}
	]
	cols = st.columns(3)
	for idx, module in enumerate(modules):
		with cols[idx % 3]:
			st.subheader(module["name"])
			st.caption(module["description"])
			st.markdown("\n".join([f"- {f}" for f in module["features"]]))

# --- Dataset Management Tab ---
from modules import dataset_manager
with tabs[1]:
	st.subheader("Dataset Management")
	st.write("Manage your datasets for benchmarking. Use the left column for actions and the right for details/preview.")
	left_col, right_col = st.columns([1, 2])
	with left_col:
		st.subheader("Actions")
		uploaded_file = st.file_uploader("Upload CSV or JSON", type=["csv", "json"])
		if uploaded_file is not None:
			file_path = os.path.join(dataset_manager.DATA_DIR, uploaded_file.name)
			with open(file_path, "wb") as f:
				f.write(uploaded_file.getbuffer())
			st.success(f"Uploaded {uploaded_file.name}")
		dataset_files = dataset_manager.list_datasets()
		selected_dataset = st.selectbox("Select dataset", dataset_files if dataset_files else ["No datasets found"])
		if st.button("Refresh List"):
			st.rerun()
		# Remove dataset functionality
		if dataset_files and selected_dataset and selected_dataset != "No datasets found":
			if st.button(f"Remove '{selected_dataset}'"):
				dataset_manager.remove_dataset(selected_dataset)
				st.success(f"Removed {selected_dataset} and its config.")
				st.rerun()
	with right_col:
		st.subheader("Dataset Preview")
		if dataset_files and selected_dataset and selected_dataset != "No datasets found":
			try:
				df = dataset_manager.load_dataset(selected_dataset)
				st.write(f"Preview of `{selected_dataset}`:")
				st.dataframe(df.head(20), width='stretch')
				st.caption(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

				# Error check for label and column
				config_file = os.path.join(dataset_manager.CONFIG_DIR, f"{selected_dataset}.config.json")
				missing_fields = []
				if os.path.exists(config_file):
					import json
					with open(config_file, "r") as f:
						config = json.load(f)
					if not config.get("label"):
						missing_fields.append("label")
					if not config.get("column"):
						missing_fields.append("column")
				else:
					missing_fields = ["label", "column"]
				if missing_fields:
					st.error(f"Missing required field(s) in dataset config: {', '.join(missing_fields)}. Please set them below.")

				# Show dataset label (benign/malicious) as a badge
				current_label = dataset_manager.get_dataset_label(selected_dataset)
				if current_label:
					color = "green" if current_label == "benign" else "red"
					st.markdown(f"<span style='color:white;background-color:{color};padding:4px 12px;border-radius:8px;font-weight:bold;'>{current_label.capitalize()} dataset</span>", unsafe_allow_html=True)

				# --- Condensed grid for dataset actions ---
				grid1, grid2 = st.columns(2)
				with grid1:
					st.subheader("Rename Dataset & Description")
					current_desc = dataset_manager.get_dataset_description(selected_dataset)
					st.write(f"Current description: {current_desc if current_desc else 'No description'}")
					new_desc = st.text_input("Set new description", value=current_desc or "", key="desc")
					if st.button("Save Description"):
						dataset_manager.set_dataset_description(selected_dataset, new_desc)
						st.success(f"Description saved for {selected_dataset}.")
						st.rerun()

					new_dataset_name = st.text_input("Rename dataset (include .csv/.json)", value=selected_dataset, key="renamedataset")
					if st.button("Rename Dataset"):
						if new_dataset_name and new_dataset_name != selected_dataset and new_dataset_name.endswith(('.csv','.json')):
							dataset_manager.rename_dataset(selected_dataset, new_dataset_name)
							st.success(f"Dataset renamed to {new_dataset_name}.")
							st.rerun()
						else:
							st.warning("Enter a valid new name (must end with .csv or .json and be different from current name).")
					st.subheader("Mark Dataset Type")
					current_label = dataset_manager.get_dataset_label(selected_dataset)
					st.write(f"Current label: **{current_label if current_label else 'Not set'}**")
					label_option = st.radio("Set dataset label:", ["benign", "malicious"], index=0 if current_label == "benign" else 1 if current_label == "malicious" else 0)
					if st.button("Save Label"):
						dataset_manager.set_dataset_label(selected_dataset, label_option)
						st.success(f"Dataset '{selected_dataset}' marked as {label_option}.")
						st.rerun()

					st.subheader("Download Dataset")
					import io
					if selected_dataset.endswith('csv'):
						csv_bytes = df.to_csv(index=False).encode('utf-8')
						st.download_button(
							label="Download as CSV",
							data=csv_bytes,
							file_name=selected_dataset,
							mime="text/csv"
						)
					else:
						json_bytes = df.to_json(orient='records').encode('utf-8')
						st.download_button(
							label="Download as JSON",
							data=json_bytes,
							file_name=selected_dataset,
							mime="application/json"
						)

				with grid2:
					st.subheader("Add New Column")
					new_col_name = st.text_input("New column name", key="addcol")
					preset_value = st.text_input("Preset value for all rows", key="presetval")
					if st.button("Add Column"):
						if new_col_name and new_col_name not in df.columns:
							dataset_manager.add_column(selected_dataset, new_col_name, preset_value)
							st.success(f"Column '{new_col_name}' added to {selected_dataset}.")
							st.cache_data.clear()
							st.rerun()
						else:
							st.warning("Column name must be unique and not empty.")

					st.subheader("Rename Column")
					col_to_rename = st.selectbox("Select column to rename", df.columns, key="renamecol")
					new_name = st.text_input("New name for selected column", key="newcolname")
					if st.button("Rename Column"):
						if new_name and new_name not in df.columns:
							dataset_manager.rename_column(selected_dataset, col_to_rename, new_name)
							st.success(f"Column '{col_to_rename}' renamed to '{new_name}' in {selected_dataset}.")
							st.cache_data.clear()
							st.rerun()
						else:
							st.warning("New column name must be unique and not empty.")

					st.subheader("Delete Column")
					col_to_delete = st.selectbox("Select column to delete", df.columns, key="deletecol")
					if st.button("Delete Column"):
						if col_to_delete:
							dataset_manager.delete_column(selected_dataset, col_to_delete)
							st.success(f"Column '{col_to_delete}' deleted from {selected_dataset}.")
							st.cache_data.clear()
							st.rerun()

				# Column selection for guardrail input (below grid)
				saved_column = dataset_manager.get_column_selection(selected_dataset)
				col_to_send = st.selectbox(
					"Select the column to send to the guardrail:",
					df.columns,
					index=df.columns.get_loc(saved_column) if saved_column in df.columns else 0,
					key="sendcol"
				)
				if st.button("Save column selection"):
					dataset_manager.set_column_selection(selected_dataset, col_to_send)
					st.success(f"Column '{col_to_send}' saved for {selected_dataset}.")
			except Exception as e:
				st.error(f"Failed to load dataset: {e}")
		else:
			st.info("No dataset selected or available.")


# --- Guardrail Integration Tab ---
from modules import guardrail_connector
with tabs[2]:
	# ...existing code...
	st.subheader("Guardrail Integration")
	st.write("Configure and test any AI guardrail provider endpoint. All fields below are customizable.")

	configs = guardrail_connector.list_guardrail_configs()
	selected_config = st.selectbox("Select guardrail config", configs if configs else ["No configs found"])
	# Button to remove selected guardrail config
	if configs and selected_config != "No configs found":
		remove_col, _ = st.columns([1, 5])
		with remove_col:
			if st.button(f"Remove '{selected_config}'", key="remove_guardrail_config"):
				guardrail_connector.remove_guardrail_config(selected_config.replace(".guardrail.json", ""))
				st.success(f"Removed guardrail config '{selected_config}'.")
				st.rerun()


	st.markdown("---")
	st.subheader("Manual Guardrail API Test (Fully Customizable)")
	# Allow user to select HTTP method, endpoint, payload, headers, and guardrail type
	http_method = st.selectbox("HTTP Method", ["POST", "GET", "PUT", "DELETE"], index=0)
	config_obj = None
	if selected_config == "No configs found":
		config_obj = None
	else:
		config_obj = guardrail_connector.load_guardrail_config(selected_config.replace(".guardrail.json", ""))
	url = st.text_input("Request URL", value="" if not config_obj else config_obj.get('endpoint', ''), key="manual_url")
	default_payload = '{"prompt": "hi how are you?"}'
	if config_obj and "payload_template" in config_obj:
		default_payload = config_obj["payload_template"]
	json_payload_str = st.text_area("JSON Payload", value=default_payload, key="manual_json")
	if config_obj and "headers" in config_obj:
		default_headers = json.dumps(config_obj["headers"], indent=2)
	else:
		default_headers = '{"Content-Type": "application/json"}'
	headers_str = st.text_area("Headers (JSON format)", value=default_headers, key="manual_headers")
	guardrail_type = st.selectbox("Guardrail Type", ["Generic (Async)", "Promptfoo (Sync)", "Other"], index=0)

	# --- Input Path, Input Key, and Pass/Fail Rule Input ---
	input_key = st.text_input(
		"Payload input key (flat, e.g. 'input', 'prompt', 'text')",
		value=config_obj.get("input_key", "input") if config_obj else "input",
		key="input_key"
	)
	pass_fail_rule = st.text_input(
		"Pass/Fail Rule (Python expression, e.g. response['flagged']==False) should evaluate to True given benign input",
		value=config_obj.get("pass_fail_rule", "") if config_obj else "",
		key="pass_fail_rule"
	)

	import json
	import asyncio
	col_test, col_save = st.columns([2, 2])
	with col_test:
		send_clicked = st.button("Send Test Request")
	with col_save:
		config_name_save = st.text_input("Config name to save", key="manual_save_name")
		save_clicked = st.button("Save Guardrail Config", key="manual_save_btn")

	# Parse payload and headers once for both actions
	json_payload = None
	headers = {}
	try:
		json_payload = json.loads(json_payload_str) if json_payload_str.strip() else None
		headers = json.loads(headers_str) if headers_str.strip() else {}
	except Exception as e:
		st.error(f"Error parsing JSON: {e}")
		headers = {}  # Ensure headers is always defined

	passfail_result_placeholder = st.empty()
	if send_clicked:
		try:
			# Route to correct connector function
			if guardrail_type == "Promptfoo (Sync)":
				response = guardrail_connector.request_promptfoo_api(url, json_payload, headers, method=http_method)
			else:
				loop = asyncio.new_event_loop()
				asyncio.set_event_loop(loop)
				response = loop.run_until_complete(guardrail_connector.request_guardrail_api(url, json_payload, headers, method=http_method))
			st.success("Request sent successfully!")
			st.json(response)

			# --- Evaluate Pass/Fail Rule ---
			if pass_fail_rule.strip():
				try:
					# Safe eval: only allow 'response' in locals
					result = eval(pass_fail_rule, {"__builtins__": {}}, {"response": response})
					if result:
						passfail_result_placeholder.success(f"Pass/Fail Rule Result: PASS (Expression evaluated to True)")
					else:
						passfail_result_placeholder.error(f"Pass/Fail Rule Result: FAIL (Expression evaluated to False)")
				except Exception as e:
					passfail_result_placeholder.error(f"Error evaluating pass/fail rule: {e}")
			else:
				passfail_result_placeholder.info("No pass/fail rule provided.")
		except Exception as e:
			st.error(f"Error: {e}")

	if save_clicked:
		config_to_save = {
			"endpoint": url,
			"api_key": headers.get("Authorization", "") if headers else "",
			"provider": guardrail_type,
			"http_method": http_method,
			"headers": headers if headers else {},
			"payload_template": json_payload_str,
			"input_key": input_key,
			"pass_fail_rule": pass_fail_rule
		}
		guardrail_connector.save_guardrail_config(config_name_save, config_to_save)
		st.success(f"Guardrail config '{config_name_save}' saved with input key, input path, and pass/fail rule.")

# --- Test Suite Management Tab ---
with tabs[3]:
	from modules import test_suite
	st.subheader("Test Suite Management & Automated Evaluation")
	st.write("Select multiple datasets and a guardrail config to run batch tests. Progress, ETA, and logs will be shown.")


	import pandas as pd
	dataset_files = dataset_manager.list_datasets()
	guardrail_configs = guardrail_connector.list_guardrail_configs()

	# Estimate row counts for each dataset (efficient, no full load)
	def estimate_row_count(file):
		path = os.path.join(dataset_manager.DATA_DIR, file)
		if file.endswith('.csv'):
			try:
				with open(path, 'r', encoding='utf-8') as f:
					return sum(1 for _ in f) - 1  # subtract header
			except Exception:
				return 0
		elif file.endswith('.json'):
			try:
				import json
				with open(path, 'r', encoding='utf-8') as f:
					data = json.load(f)
					if isinstance(data, list):
						return len(data)
					else:
						return 0
			except Exception:
				return 0
		else:
			return 0

	dataset_options = [f"{f} ({estimate_row_count(f)} rows)" for f in dataset_files]
	file_map = {f"{f} ({estimate_row_count(f)} rows)": f for f in dataset_files}

	selected_options = st.multiselect("Select datasets for evaluation", dataset_options, key="test_suite_datasets")
	selected_datasets = [file_map[o] for o in selected_options]
	selected_guardrail = st.selectbox("Select guardrail config", guardrail_configs, key="test_suite_guardrail")

	run_eval = st.button("Run Evaluation", key="run_eval_btn_suite")
	stop_eval = st.button("Stop Evaluation", key="stop_eval_btn_suite")
	progress_placeholder = st.empty()
	eta_placeholder = st.empty()
	row_progress_placeholder = st.empty()
	results_placeholder = st.empty()
	log_download_placeholder = st.empty()
	error_placeholder = st.empty()

	# Use session state to allow stopping
	if "suite_eval_running" not in st.session_state:
		st.session_state["suite_eval_running"] = False

	def progress_cb(rows_done, total_rows):
		progress_placeholder.progress(rows_done / total_rows, text=f"Progress: {rows_done}/{total_rows} rows")
	def eta_cb(eta):
		eta_placeholder.info(f"Estimated time left: {eta}s")
	def row_cb(ds, idx, total):
		if idx is not None:
			row_progress_placeholder.info(f"{ds}: Row {idx}/{total}")
		else:
			row_progress_placeholder.empty()

	if run_eval and selected_datasets and selected_guardrail:
		st.session_state["suite_eval_running"] = True
		all_logs, result_file = test_suite.run_evaluation(
			dataset_manager,
			guardrail_connector,
			selected_datasets,
			selected_guardrail,
			st.session_state,
			progress_cb=progress_cb,
			eta_cb=eta_cb,
			row_cb=row_cb
		)
		progress_placeholder.empty()
		eta_placeholder.empty()
		# Show logs, download, and save to results directory
		if all_logs:
			results_placeholder.write("Evaluation complete. Download logs below.")
			import json as _json
			log_json = _json.dumps(all_logs, indent=2, ensure_ascii=False)
			log_download_placeholder.download_button(
				label="Download prompt/reply logs as JSON",
				data=log_json.encode('utf-8'),
				file_name="guardrail_test_logs.json",
				mime="application/json"
			)
			# Show logs in a table (flattened for display)
			import pandas as pd
			log_df = pd.json_normalize(all_logs)
			results_placeholder.dataframe(log_df)
			results_placeholder.success(f"Logs saved to {result_file}")
		st.session_state["suite_eval_running"] = False
	elif stop_eval:
		st.session_state["suite_eval_running"] = False
		error_placeholder.error("Evaluation stopped by user.")
		progress_placeholder.empty()
		eta_placeholder.empty()
		row_progress_placeholder.empty()
		results_placeholder.empty()
		log_download_placeholder.empty()

# --- Placeholders for other tabs ---
for i, tab in enumerate(tabs[4:], start=4):
    if tab_names[i] == "Results & Visualization":
        from modules import results_viz
        results_dir = os.path.join(os.path.dirname(__file__), "results")
        with tab:
            results_viz.show_results_viz_tab(tab, results_dir)
    elif tab_names[i] == "Comparison":
        # Comparison tab logic is already implemented below
        pass
    else:
        with tab:
            st.info(f"{tab_names[i]} module coming soon.")

# --- Comparison Tab UI scaffold ---
if "Comparison" in tab_names:
    comparison_tab_idx = tab_names.index("Comparison")
    with tabs[comparison_tab_idx]:
        import modules.results as results
        import matplotlib.pyplot as plt
        st.subheader("Compare Two Results Files")
        results_dir = os.path.join(os.path.dirname(__file__), "results")
        result_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
        if len(result_files) < 2:
            st.warning("Need at least two result files to compare.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                file1 = st.selectbox("Select first result file", result_files, key="compare_file1")
            with col2:
                file2 = st.selectbox("Select second result file", result_files, key="compare_file2")
            if file1 and file2 and file1 != file2:
                data1 = results.load_result(os.path.join(results_dir, file1))
                data2 = results.load_result(os.path.join(results_dir, file2))
                logs1 = data1.get("results", [])
                logs2 = data2.get("results", [])
                meta1 = data1.get("guardrail_config", "N/A")
                meta2 = data2.get("guardrail_config", "N/A")
                ds_info1 = results.get_dataset_info(data1)
                ds_info2 = results.get_dataset_info(data2)
                st.markdown("### Metadata")
                st.write({"File 1": file1, "Config 1": meta1, "Datasets": ds_info1["datasets"], "Row counts": ds_info1["row_counts"]})
                st.write({"File 2": file2, "Config 2": meta2, "Datasets": ds_info2["datasets"], "Row counts": ds_info2["row_counts"]})
                # Metrics
                metrics1 = results.get_metrics(logs1)
                metrics2 = results.get_metrics(logs2)
                st.markdown("### Metrics Comparison")
                metrics_df = pd.DataFrame({
                    file1: {k: v for k, v in metrics1.items() if k != "confusion_matrix"},
                    file2: {k: v for k, v in metrics2.items() if k != "confusion_matrix"}
                })
                st.table(metrics_df)
                # Confusion matrices
                st.markdown("### Confusion Matrices")
                cm1 = pd.DataFrame(metrics1["confusion_matrix"], index=["Actual: malicious", "Actual: benign"], columns=["Pred: False", "Pred: True"])
                cm2 = pd.DataFrame(metrics2["confusion_matrix"], index=["Actual: malicious", "Actual: benign"], columns=["Pred: False", "Pred: True"])
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"Confusion Matrix for {file1}")
                    st.table(cm1)
                with c2:
                    st.write(f"Confusion Matrix for {file2}")
                    st.table(cm2)
                # Bar chart for metrics
                st.markdown("### Metrics Bar Chart")
                fig, ax = plt.subplots(figsize=(7, 4))
                metric_names = ["accuracy", "precision", "recall", "f1"]
                vals1 = [metrics1[m] for m in metric_names]
                vals2 = [metrics2[m] for m in metric_names]
                ax.bar([x + " (1)" for x in metric_names], vals1, alpha=0.7, label=file1)
                ax.bar([x + " (2)" for x in metric_names], vals2, alpha=0.7, label=file2)
                ax.set_ylabel("Score")
                ax.set_ylim(0, 1)
                ax.legend()
                st.pyplot(fig)
                # Row-level diff
                st.markdown("### Row-Level Differences (where predictions differ)")
                diff_df = results.diff_logs(logs1, logs2)
                if not diff_df.empty:
                    st.dataframe(diff_df)
                else:
                    st.info("No row-level differences found (based on 'input' and prediction).")
