import requests


print(requests.get("http://127.0.0.1:8082").headers)
print(requests.get("http://127.0.0.1:8082").json()["data"])
