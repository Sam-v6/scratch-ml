import json, requests

with open("/home/sevani/repos/scratch-ml/log/mlruns/767172877346565471/models/m-e162cc9379824b8693d43988315d0772/artifacts/serving_input_example.json") as f:
    payload = json.load(f)

r = requests.post(
    "http://127.0.0.1:5001/invocations",
    headers={"Content-Type": "application/json"},
    data=json.dumps(payload),
    timeout=30,
)
print(r.status_code, r.text)  # expect JSON predictions