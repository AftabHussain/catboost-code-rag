import requests

try:
    r = requests.get("https://huggingface.co", timeout=5)
    print("Status code:", r.status_code)
except Exception as e:
    print("Error:", e)

