import os
import json
import requests
from tabulate import tabulate

API_URL = "http://127.0.0.1:5000/upload"
TEST_DIR = "images"   # <--- FIXED

results = []

def classify_image(path):
    try:
        files = {'file': open(path, "rb")}
        response = requests.post(API_URL, files=files)
        return response.status_code, response.text
    except Exception as e:
        return 500, str(e)

print("\n==============================")
print("    AUTOMATED TEST RUNNER")
print("==============================\n")

for root, dirs, files in os.walk(TEST_DIR):
    for file in files:
        file_path = os.path.join(root, file)
        category = os.path.basename(root)

        status, response = classify_image(file_path)

        try:
            data = json.loads(response)
            result = data.get("result", "Error")
            class_name = data.get("class_name", "-")
        except:
            result = "Error"
            class_name = "-"

        results.append([file, category, result, class_name, status])

table = tabulate(results, headers=["File", "Actual", "Predicted", "Class Name", "Status"], tablefmt="grid")

print(table)

with open("test_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\n✔ Test results saved to test_results.json")
