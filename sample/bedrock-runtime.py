import boto3
from botocore.exceptions import ClientError
import os
import dotenv

dotenv.load_dotenv()

# Create a Bedrock Runtime client in the AWS region you want to use
client = boto3.client(
    "bedrock-runtime",
    region_name=(os.getenv("AWS_REGION1") or "us-east-1")
)

# Set the model ID
model_id = os.getenv("BEDROCK_MODEL_ID") or "amazon.nova-lite-v1:0"

# Start a conversation with the user message.
user_message = "What is an apple?"
conversation = [
    {
        "role": "user",
        "content": [{"text": user_message}]
    }
]

# conversation = [
#     {"role": "user", "content": [
#         {"text": "What’s in this picture?"},
#         {"image": {"format": "png", "source": {"bytes": base64_image_data}}}
#     ]}
# ]

# conversation = [
#     {"role": "system", "content": [{"text": "You are a Python tutor."}]},
#     {"role": "user", "content": [{"text": "Explain decorators."}]},
#     {"role": "assistant", "content": [{"text": "A decorator wraps a function..."}]},
#     {"role": "user", "content": [{"text": "Show me an example."}]}
# ]

try:
    # Send the message to the model, using a basic inference configuration.
    response = client.converse(
        modelId=model_id,
        messages=conversation,
        inferenceConfig={"maxTokens": 512, "temperature": 0.5, "topP": 0.9},
    )

    # Extract and print the response text.
    response_text = response["output"]["message"]["content"][0]["text"]
    print(response_text)

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)