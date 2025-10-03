import json
import os
import uuid
from datetime import datetime, timezone
from main import run_agent

# Optional MongoDB support: only perform DB ops when pymongo is available.
try:
    import pymongo
    MONGO_AVAILABLE = True
except Exception:
    pymongo = None
    MONGO_AVAILABLE = False


def _connect_mongo():
    """Create a MongoDB client using ATLAS_URI from env if available.

    Returns (db, client) or (None, None) if connection cannot be established.
    """
    if not MONGO_AVAILABLE:
        return None, None
    atlas_uri = os.getenv('ATLAS_URI')
    if not atlas_uri:
        return None, None
    try:
        client = pymongo.MongoClient(atlas_uri, serverSelectionTimeoutMS=5000)
        # attempt server selection
        client.admin.command('ping')
        db = "chats"
        return db, client
    except Exception:
        return None, None


def lambda_handler(event, context):
    """Handle new request format and return MCP-style response.

    Expected input body (JSON):
    {
      "message": "__INITIATE_CONVERSATION__",
      "userId": "0402108-07-0711",
      "createdAt": "2025-10-2T01:03:00.000Z",
      "sessionId": "(new-session)",
      "attachment": [],
      "ekyc": { ... }
    }

    If sessionId == '(new-session)', treat as first_time_connection: generate a sessionId
    (uuid), create a new chats collection for this user if possible and insert the initial session doc.

    Always generate a messageId (uuid) for the response. Use `run_agent` to generate reply text.
    """
    # Early health check: detect path/method and return 200 for GET /{stage}/health
    request_context = event.get('requestContext', {})
    http = request_context.get('http') or {}
    path = None
    method = None
    if isinstance(http, dict):
        path = http.get('path')
        method = http.get('method')

    # Some invocations provide the path differently; fallback to rawPath/path
    if not path:
        path = event.get('rawPath') or event.get('path')
    if not method:
        method = event.get('httpMethod')

    if path and path.endswith('/health') and (method or '').upper() == 'GET':
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'status': 'ok'})
        }

    # Parse body for regular requests
    body = event.get('body')
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except Exception:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Invalid JSON body'})
            }

    if not isinstance(body, dict):
        return {
            'statusCode': 400,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': 'Request body must be a JSON object'})
        }

    # Validate required fields
    user_id = body.get('userId')
    message = body.get('message')
    session_id = body.get('sessionId')
    ekyc = body.get('ekyc') or {}

    if not user_id or not message or session_id is None:
        return {
            'statusCode': 400,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': "Missing required fields: 'userId', 'message', or 'sessionId'"})
        }

    # Generate a messageId for this incoming message
    message_id = str(uuid.uuid4())

    # If new session, create sessionId and initialize collection/document in MongoDB if available
    new_session_generated = None
    db, client = _connect_mongo()
    try:
        if session_id == '(new-session)':
            new_session_generated = str(uuid.uuid4())
            # Prepare the session document format
            session_doc = {
                'sessionId': new_session_generated,
                'createdAt': datetime.now(timezone.utc).isoformat(),
                'messages': [],
                'status': 'active',
                'topic': '',
                'context': {},
                'ekyc': ekyc or {}
            }
            if db is not None:
                coll_name = user_id
                coll = db[coll_name]
                # Insert the document if a session with same id does not exist
                coll.insert_one(session_doc)
        else:
            # existing session: update ekyc if provided and append message record to messages
            if db is not None and session_id:
                coll_name = user_id
                coll = db[coll_name]
                update_ops = {}
                if ekyc:
                    update_ops['ekyc'] = ekyc
                if update_ops:
                    coll.update_one({'sessionId': session_id}, {'$set': update_ops})
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    # Determine prompt for Bedrock. For first-time connection, request a welcome message.
    try:
        if session_id == '(new-session)':
            prompt = (
                "You are an assistant that composes a short friendly welcome message for a "
                "government services portal called MyGovHub. Keep it concise and helpful."
            )
        else:
            # For regular messages, pass through the user's message to the model
            prompt = message

        response_text = run_agent(prompt)

        # Prepare the MCP response payload
        resp_body = {
            'status': {'statusCode': 200, 'message': 'Success'},
            'data': {
                'messageId': message_id,
                'message': response_text,
                'sessionId': new_session_generated if new_session_generated else session_id,
                'attachment': body.get('attachment') or []
            }
        }

        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps(resp_body)
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': str(e)})
        }
