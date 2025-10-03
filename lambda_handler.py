import json
from main import run_agent


def lambda_handler(event, context):
    """AWS Lambda handler for API Gateway proxy integration.

    Expects a POST request with JSON body: {"prompt": "some text"}
    Returns: {"response": "model output"}
    """
    # Handle a simple healthcheck endpoint quickly
    request_context = event.get('requestContext', {})
    http = request_context.get('http') or {}
    path = None
    method = None
    if isinstance(http, dict):
        path = http.get('path')
        method = http.get('method')

    # Some invocations (e.g., direct from API Gateway) provide the path differently
    # fallback to raw path if present
    if not path:
        path = event.get('rawPath') or event.get('path')
    if not method:
        method = event.get('httpMethod')

    # Support health path with optional stage prefix, e.g., '/dev/health'
    if path and path.endswith('/health') and (method or '').upper() == 'GET':
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'status': 'ok'})
        }

    # event['body'] may be a JSON string when invoked via API Gateway
    body = event.get('body')
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except Exception:
            # If body isn't JSON, return 400
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Invalid JSON body'})
            }

    if not body or 'prompt' not in body:
        return {
            'statusCode': 400,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': "Missing 'prompt' in request body"})
        }

    prompt = body['prompt']

    try:
        response_text = run_agent(prompt)
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'response': response_text})
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': str(e)})
        }
