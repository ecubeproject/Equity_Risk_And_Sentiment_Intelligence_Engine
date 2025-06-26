import os
import requests

endpoint = os.getenv("AZURE_LANG_ENDPOINT") or "https://pyspark-azurelangner.cognitiveservices.azure.com"
key = os.getenv("AZURE_LANG_KEY")

url = f"{endpoint}/language/analyze-text?api-version=2023-04-01-preview"
headers = {
    "Ocp-Apim-Subscription-Key": key,
    "Content-Type": "application/json"
}
body = {
    "kind": "EntityRecognition",
    "parameters": { "modelVersion": "latest" },
    "analysisInput": {
        "documents": [
            {
                "id": "1",
                "language": "en",
                "text": "Microsoft was founded in Redmond"
            }
        ]
    }
}

resp = requests.post(url, headers=headers, json=body)
print(resp.status_code)
print(resp.json())
