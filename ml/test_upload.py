import requests

url = "http://127.0.0.1:5000/upload"  # Flask server
files = {'file': open("images/oip.jpg", "rb")}  # change filename to test different images

try:
    response = requests.post(url, files=files)
    print("Status Code:", response.status_code)
    print("Server Response:", response.text)
except Exception as e:
    print("Error:", e)
