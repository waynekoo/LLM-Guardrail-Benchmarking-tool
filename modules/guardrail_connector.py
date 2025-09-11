def remove_guardrail_config(name):
	"""
	Remove a saved guardrail config by name.
	"""
	file_path = os.path.join(CONFIG_DIR, f"{name}.guardrail.json")
	if os.path.exists(file_path):
		os.remove(file_path)
		return True
	return False
import requests

def request_promptfoo_api(url: str, json_payload: dict = None, headers: dict = None, method: str = "POST"):
	"""
	Send a synchronous request to the Promptfoo API (or any sync endpoint).
	"""
	try:
		headers = headers or {}
		if method.upper() == "POST":
			response = requests.post(url, json=json_payload, headers=headers)
		elif method.upper() == "GET":
			response = requests.get(url, params=json_payload, headers=headers)
		elif method.upper() == "PUT":
			response = requests.put(url, json=json_payload, headers=headers)
		elif method.upper() == "DELETE":
			response = requests.delete(url, json=json_payload, headers=headers)
		else:
			raise ValueError(f"Unsupported HTTP method: {method}")
		response.raise_for_status()
		return response.json()
	except Exception as e:
		return {"error": str(e)}
import aiohttp
import asyncio

class GuardrailRequestException(Exception):
	pass

async def request_guardrail_api(url: str, json_payload: dict, headers: dict, method: str = "POST"):
	async with aiohttp.ClientSession() as session:
		try:
			method = method.upper()
			if method == "POST":
				response = await session.post(
					url=url,
					json=json_payload,
					headers=headers,
					ssl=False,
					raise_for_status=True,
				)
			elif method == "GET":
				response = await session.get(
					url=url,
					params=json_payload,
					headers=headers,
					ssl=False,
					raise_for_status=True,
				)
			elif method == "PUT":
				response = await session.put(
					url=url,
					json=json_payload,
					headers=headers,
					ssl=False,
					raise_for_status=True,
				)
			elif method == "DELETE":
				response = await session.delete(
					url=url,
					json=json_payload,
					headers=headers,
					ssl=False,
					raise_for_status=True,
				)
			else:
				raise ValueError(f"Unsupported HTTP method: {method}")
			response_json = await response.json()
			return response_json
		except Exception as e:
			raise GuardrailRequestException(e)

import os
import json

CONFIG_DIR = "config"
os.makedirs(CONFIG_DIR, exist_ok=True)

def list_guardrail_configs():
	"""
	List all saved guardrail configurations.
	"""
	return [f for f in os.listdir(CONFIG_DIR) if f.endswith(".guardrail.json")]

def save_guardrail_config(name, config):
	"""
	Save a guardrail config (API endpoint, key, provider, etc.)
	"""
	file_path = os.path.join(CONFIG_DIR, f"{name}.guardrail.json")
	with open(file_path, "w") as f:
		json.dump(config, f)

def load_guardrail_config(name):
	file_path = os.path.join(CONFIG_DIR, f"{name}.guardrail.json")
	if os.path.exists(file_path):
		with open(file_path, "r") as f:
			return json.load(f)
	return None

def test_guardrail_connection(config):
	"""
	Dummy test for now. Replace with actual API call later.
	"""
	# Example: try to send a request to config['endpoint'] with config['api_key']
	# For now, just return True if endpoint and api_key exist
	return bool(config.get("endpoint")) and bool(config.get("api_key"))
