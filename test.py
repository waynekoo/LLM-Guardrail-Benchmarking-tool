import requests

def request_llm_guard_prompt(prompt: str):
    try:
        response = requests.post(
            url="http://52.76.109.235:8000/analyze/prompt",
            json={"prompt": prompt},
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer my-token"
            }
        )
        print("after post")
        response.raise_for_status()  # raises exception for bad status codes
        print(response)
        response_json = response.json()
        return response_json
    except Exception as e:
        print("----error----")
        print(e)
        return None

def main():
    test = request_llm_guard_prompt("hi how are you?")
    print(test)

if __name__ == "__main__":
    main()
