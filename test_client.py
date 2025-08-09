# test_client.py
import requests

url = "http://localhost:8000/hackrx/run"
data = {
    "query": "Does this policy cover knee surgery, and what are the conditions?",
    "pdf_url": "your_pdf_blob_url_here"
}

response = requests.post(url, json=data)
print(response.json())
