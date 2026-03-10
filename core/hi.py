from google import genai
client = genai.Client(api_key="AIzaSyAgG3Ojw56gZ-H1n5cnwDfw9MAfb7Hoq6k")
for m in client.models.list():
    print(m.name)