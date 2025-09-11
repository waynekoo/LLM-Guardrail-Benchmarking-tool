

import time
import copy
import json
import os
from datetime import datetime
import pandas as pd

def run_evaluation(dataset_manager, guardrail_connector, selected_datasets, selected_guardrail, session_state, progress_cb=None, eta_cb=None, row_cb=None):

	# --- New: Save guardrail config name and metadata in results ---
	all_logs = []
	total_rows = 0
	for ds in selected_datasets:
		df = dataset_manager.load_dataset(ds)
		total_rows += len(df)
	rows_done = 0
	start_time = time.time()
	for ds in selected_datasets:
		if not session_state.get("suite_eval_running", False):
			break
		df = dataset_manager.load_dataset(ds)
		col_to_send = dataset_manager.get_column_selection(ds)
		label = dataset_manager.get_dataset_label(ds)
		config = guardrail_connector.load_guardrail_config(selected_guardrail.replace(".guardrail.json", ""))
		payload_template = config.get("payload_template")
		input_key = config.get("input_key", "input")
		input_path = config.get("input_path", None)
		headers = config.get("headers", {})
		method = config.get("http_method", "POST")
		url = config.get("endpoint", "")
		provider = config.get("provider", "")
		guardrail_name = selected_guardrail.replace(".guardrail.json", "")
		try:
			payload = json.loads(payload_template) if payload_template else {input_key: ""}
		except Exception:
			payload = {input_key: ""}
		def set_nested_value(obj, path, value):
			import re
			parts = re.findall(r'\w+|\[\d+\]', path)
			cur = obj
			for i, part in enumerate(parts):
				if part.startswith('[') and part.endswith(']'):
					idx = int(part[1:-1])
					if i == len(parts) - 1:
						cur[idx] = value
					else:
						cur = cur[idx]
				else:
					if i == len(parts) - 1:
						cur[part] = value
					else:
						if part not in cur:
							cur[part] = {}
						cur = cur[part]
		for idx, row in df.iterrows():
			if not session_state.get("suite_eval_running", False):
				break
			prompt_val = row[col_to_send] if col_to_send in row else ""
			payload_copy = copy.deepcopy(payload)
			if input_path:
				set_nested_value(payload_copy, input_path, prompt_val)
			else:
				payload_copy[input_key] = prompt_val
			t0 = time.time()
			try:
				if provider == "Promptfoo (Sync)":
					reply = guardrail_connector.request_promptfoo_api(url, payload_copy, headers, method=method)
				else:
					import asyncio
					loop = asyncio.new_event_loop()
					asyncio.set_event_loop(loop)
					reply = loop.run_until_complete(guardrail_connector.request_guardrail_api(url, payload_copy, headers, method=method))
			except Exception as e:
				reply = {"error": str(e)}
			t1 = time.time()

			# --- Evaluate pass/fail rule from config ---
			pass_fail_rule = config.get("pass_fail_rule", "")
			eval_result = None
			eval_error = None
			if pass_fail_rule.strip():
				try:
					eval_result = eval(pass_fail_rule, {"__builtins__": {}}, {"response": reply})
				except Exception as e:
					eval_result = False
					eval_error = str(e)

			log_entry = {
				"dataset": ds,
				"row": idx,
				"label": label,
				"payload": payload_copy,
				"reply": reply,
				"elapsed": round(t1-t0, 3),
				"evaluation": {
					"pass_fail_rule": pass_fail_rule,
					"isSafeResult": eval_result,
					"error": eval_error
				}
			}
			all_logs.append(log_entry)
			rows_done += 1
			avg_time = (t1 - start_time) / rows_done if rows_done else 0
			rows_left = total_rows - rows_done
			eta = rows_left * avg_time
			if progress_cb:
				progress_cb(rows_done, total_rows)
			if eta_cb:
				eta_cb(int(eta))
			if row_cb:
				row_cb(ds, idx+1, len(df))
		if row_cb:
			row_cb(ds, None, None)  # clear
	# Save logs to results dir
	results_dir = "results"
	os.makedirs(results_dir, exist_ok=True)
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	# Use guardrail_name for file naming, replace spaces and special chars with underscores
	safe_guardrail_name = ''.join(c if c.isalnum() else '_' for c in guardrail_name)
	result_file = os.path.join(results_dir, f"{safe_guardrail_name}_result_{timestamp}.json")
	result_json = json.dumps({
		"guardrail_config": guardrail_name,
		"guardrail_metadata": {
			"provider": provider,
			"endpoint": url,
			"http_method": method,
			"headers": headers
		},
		"results": all_logs
	}, indent=2, ensure_ascii=False)
	with open(result_file, "w", encoding="utf-8") as f:
		f.write(result_json)
	return all_logs, result_file
