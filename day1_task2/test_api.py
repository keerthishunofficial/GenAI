import requests
import json

problem = '2x + 3 = 5'
response = requests.post(
    'http://localhost:8000/solve',
    json={'problem': problem},
    headers={'Content-Type': 'application/json'}
)

print('Status Code:', response.status_code)
print()
if response.status_code == 200:
    result = response.json()
    print('Steps found:', len(result['steps']))
    print()
    for step in result['steps']:
        print(f"Step {step['step_number']}: {step['reasoning']}")
    print()
    print(f"Final Answer: {result['final_answer']}")
else:
    print('Error:', response.text)
