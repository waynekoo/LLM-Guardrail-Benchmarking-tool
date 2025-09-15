import os
import json
import pandas as pd
import datetime
import streamlit as st
import matplotlib.pyplot as plt

def show_results_viz_tab(tab, results_dir):
    with tab:
        st.subheader("Results & Visualization")
        result_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
        if not result_files:
            st.warning("No result files found.")
            return
        result_files.sort(reverse=True)
        selected_result = st.selectbox("Select result file to view", result_files, index=0)
        # Add delete button for selected result file
        delete_col, view_col = st.columns([1, 5])
        with delete_col:
            if st.button(f"Delete '{selected_result}'", key="delete_result_file"):
                file_path = os.path.join(results_dir, selected_result)
                try:
                    os.remove(file_path)
                    st.success(f"Deleted result file '{selected_result}'.")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Failed to delete file: {e}")
        with view_col:
            st.write(f"Showing results from `{selected_result}`")
            file_path = os.path.join(results_dir, selected_result)
            with open(file_path, "r", encoding="utf-8") as f:
                result_data = json.load(f)
            logs = result_data.get("results", [])
            guardrail_config = result_data.get("guardrail_config", "N/A")
            st.markdown("### Run Metadata")
            st.write("**Guardrail Config:**", guardrail_config)
            mtime = os.path.getmtime(file_path)
            st.write("**Run Time:**", datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S"))
            datasets = sorted(set(log.get("dataset", "N/A") for log in logs))
            st.write("**Datasets Evaluated:**", datasets)
            # Confusion matrix
            st.markdown("### Confusion Matrix")
            cm_labels = ["malicious", "benign"]
            cm_pred_labels = ["isSafeResult=False", "isSafeResult=True"]
            cm = [[0, 0], [0, 0]]
            for log in logs:
                actual = 0 if log["label"] == "malicious" else 1
                pred = 0 if log["evaluation"]["isSafeResult"] == False else 1
                cm[actual][pred] += 1
            cm_total = sum(sum(row) for row in cm)
            cm_percent = [[0, 0], [0, 0]]
            for i in range(2):
                for j in range(2):
                    if cm_total > 0:
                        cm_percent[i][j] = 100 * cm[i][j] / cm_total
                    else:
                        cm_percent[i][j] = 0
            cm_display = [[f"{cm[i][j]} ({cm_percent[i][j]:.1f}%)" for j in range(2)] for i in range(2)]
            cm_df = pd.DataFrame(cm_display, index=["Actual: malicious", "Actual: benign"], columns=["Predicted: isSafeResult=False", "Predicted: isSafeResult=True"])
            st.table(cm_df)
            st.markdown("### Full Results Log")
            log_df = pd.json_normalize(logs)
            st.dataframe(log_df)
