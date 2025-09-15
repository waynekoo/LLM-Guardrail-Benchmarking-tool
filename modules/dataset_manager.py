def set_row_count(filename, row_count):
	"""
	Save the number of rows in the dataset config file.
	"""
	config_file = os.path.join(CONFIG_DIR, f"{filename}.config.json")
	config = {}
	if os.path.exists(config_file):
		with open(config_file, "r") as f:
			config = json.load(f)
	config["row_count"] = row_count
	with open(config_file, "w") as f:
		json.dump(config, f)
def rename_dataset(old_filename, new_filename):
	"""
	Rename the dataset file and its unified config file.
	"""
	old_path = os.path.join(DATA_DIR, old_filename)
	new_path = os.path.join(DATA_DIR, new_filename)
	if os.path.exists(old_path):
		os.rename(old_path, new_path)
	# Rename unified config file if it exists
	old_config = os.path.join(CONFIG_DIR, f"{old_filename}.config.json")
	new_config = os.path.join(CONFIG_DIR, f"{new_filename}.config.json")
	if os.path.exists(old_config):
		os.rename(old_config, new_config)

def set_dataset_description(filename, description):
	config_file = os.path.join(CONFIG_DIR, f"{filename}.config.json")
	config = {}
	if os.path.exists(config_file):
		with open(config_file, "r") as f:
			config = json.load(f)
	config["description"] = description
	with open(config_file, "w") as f:
		json.dump(config, f)

def get_dataset_description(filename):
	config_file = os.path.join(CONFIG_DIR, f"{filename}.config.json")
	if os.path.exists(config_file):
		try:
			with open(config_file, "r") as f:
				return json.load(f).get("description")
		except Exception:
			return None
	return None
def set_dataset_label(filename, label):
	"""
	Mark the dataset as 'benign' or 'malicious'.
	"""
	config_file = os.path.join(CONFIG_DIR, f"{filename}.config.json")
	config = {}
	if os.path.exists(config_file):
		with open(config_file, "r") as f:
			config = json.load(f)
	config["label"] = label
	with open(config_file, "w") as f:
		json.dump(config, f)

def get_dataset_label(filename):
	"""
	Retrieve the label ('benign' or 'malicious') for the dataset.
	"""
	config_file = os.path.join(CONFIG_DIR, f"{filename}.config.json")
	if os.path.exists(config_file):
		try:
			with open(config_file, "r") as f:
				return json.load(f).get("label")
		except Exception:
			return None
	return None

import os
import pandas as pd
import json

DATA_DIR = "data"
CONFIG_DIR = "config"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

def list_datasets():
	return [f for f in os.listdir(DATA_DIR) if f.endswith('.csv') or f.endswith('.json')]

def load_dataset(filename):
	file_path = os.path.join(DATA_DIR, filename)
	if filename.endswith('.csv'):
		return pd.read_csv(file_path)
	elif filename.endswith('.json'):
		return pd.read_json(file_path)
	else:
		return None

def save_dataset(filename, df):
	file_path = os.path.join(DATA_DIR, filename)
	if filename.endswith('.csv'):
		df.to_csv(file_path, index=False)
	elif filename.endswith('.json'):
		df.to_json(file_path, orient='records')

def remove_dataset(filename):
	file_path = os.path.join(DATA_DIR, filename)
	if os.path.exists(file_path):
		os.remove(file_path)
	config_file = os.path.join(CONFIG_DIR, f"{filename}.config.json")
	if os.path.exists(config_file):
		os.remove(config_file)

def add_column(filename, col_name, value):
	df = load_dataset(filename)
	df[col_name] = value
	save_dataset(filename, df)

def rename_column(filename, old_name, new_name):
	df = load_dataset(filename)
	df = df.rename(columns={old_name: new_name})
	save_dataset(filename, df)

def delete_column(filename, col_name):
	df = load_dataset(filename)
	df = df.drop(columns=[col_name])
	save_dataset(filename, df)

def get_column_selection(filename):
	config_file = os.path.join(CONFIG_DIR, f"{filename}.config.json")
	if os.path.exists(config_file):
		try:
			with open(config_file, "r") as f:
				return json.load(f).get("column")
		except Exception:
			return None
	return None

def set_column_selection(filename, column):
	config_file = os.path.join(CONFIG_DIR, f"{filename}.config.json")
	config = {}
	if os.path.exists(config_file):
		with open(config_file, "r") as f:
			config = json.load(f)
	config["column"] = column
	with open(config_file, "w") as f:
		json.dump(config, f)
