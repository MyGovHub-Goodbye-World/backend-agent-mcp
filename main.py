import boto3
from botocore.exceptions import ClientError
import os
import dotenv
import json

dotenv.load_dotenv()

# Create a Bedrock Runtime client in the AWS region you want to use
client = boto3.client(
    "bedrock-runtime",
    region_name=(os.getenv("AWS_REGION1") or "us-east-1")
)

# Set the model ID (override with env var BEDROCK_MODEL_ID)
model_id = os.getenv("BEDROCK_MODEL_ID") or "amazon.nova-lite-v1:0"


def run_agent(
    prompt: str,
    max_tokens: int = int(os.getenv("BEDROCK_MAX_TOKENS", 512)),
    temperature: float = float(os.getenv("BEDROCK_TEMPERATURE", 0.5)),
    top_p: float = float(os.getenv("BEDROCK_TOP_P", 0.8))
) -> str:
    """Send `prompt` to Bedrock converse and return the text response.

    Inputs:
        - prompt: user prompt string
        - max_tokens, temperature, top_p: inference config

    Returns:
        - response text from the model
    """
    conversation = [
        {
            "role": "user",
            "content": [{"text": prompt}]
        }
    ]

    try:
        response = client.converse(
            modelId=model_id,
            messages=conversation,
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature, "topP": top_p},
        )

        # Extract the response text. The response shape follows the Bedrock Runtime converse API.
        response_text = response["output"]["message"]["content"][0]["text"]
        return response_text

    except (ClientError, Exception) as e:
        raise RuntimeError(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")


def _cli_main():
    """Allow running main.py locally with a JSON file containing {"prompt": "..."}.

    Usage: python main.py input.json
    Prints the model response to stdout.
    """
    import sys

    if len(sys.argv) < 2:
        print('Usage: python main.py <input.json>')
        sys.exit(2)

    path = sys.argv[1]
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    prompt = payload.get('prompt')
    if not prompt:
        print("Input JSON must contain a 'prompt' field")
        sys.exit(2)

    try:
        resp = run_agent(prompt)
        print(resp)
    except Exception as e:
        print(str(e))
        sys.exit(1)


if __name__ == '__main__':
    _cli_main()
