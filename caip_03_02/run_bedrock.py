from call_bedrock import call_bedrock
import boto3
import json

flower_data = []
# Create a Bedrock runtime client
bedrock = boto3.client('bedrock-runtime')

# read from file
with open('prompt_template.txt', 'r') as file:
    prompt = file.read()

state = "Washington"

prompt = prompt.replace("{state_name}", state)

print(prompt)

# Call Bedrock and print responses
responses = call_bedrock(bedrock, prompt)

for response in responses:
    flower_data.append(response)

json.dump(flower_data, open('flower_data.json', 'w'))