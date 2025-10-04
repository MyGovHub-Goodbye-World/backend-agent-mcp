import json
import os
import uuid
from datetime import datetime, timezone
import traceback
import base64
import requests

import boto3
from botocore.exceptions import ClientError
import logging

# optional dotenv already handled earlier in this file; ensure environment loaded
try:
    import dotenv  # type: ignore
    dotenv.load_dotenv()
except Exception:
    pass

# Create a Bedrock Runtime client
_bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name=(os.getenv("AWS_REGION1") or "us-east-1")
)

# Set the model ID (override with env var BEDROCK_MODEL_ID)
_model_id = os.getenv("BEDROCK_MODEL_ID") or "amazon.nova-lite-v1:0"


def run_agent(
    prompt: str,
    max_tokens: int = int(os.getenv("BEDROCK_MAX_TOKENS", 512)),
    temperature: float = float(os.getenv("BEDROCK_TEMPERATURE", 0.5)),
    top_p: float = float(os.getenv("BEDROCK_TOP_P", 0.8)),
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
        response = _bedrock_client.converse(
            modelId=_model_id,
            messages=conversation,
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature, "topP": top_p},
        )

        # Extract the response text. The response shape follows the Bedrock Runtime converse API.
        response_text = response["output"]["message"]["content"][0]["text"]
        return response_text

    except (ClientError, Exception) as e:
        raise RuntimeError(f"ERROR: Can't invoke '{_model_id}'. Reason: {e}")

# Optional dotenv for local development
try:
    import dotenv  # type: ignore
    dotenv.load_dotenv()
except Exception:
    dotenv = None

# pymongo is required: failing to import will cause the Lambda to fail to initialize
import pymongo  # type: ignore

# CORS defaults for browser clients (keeps it permissive for local testing/origins)
CORS_HEADERS = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type,Authorization,X-Requested-With',
    'Access-Control-Allow-Credentials': 'false',
}


def _cors_response(status_code=200, body=None, content_type='application/json'):
    """Utility to build a response that always includes CORS headers.

    body may be a dict/list (will be JSON-encoded) or a string. If body is None,
    an empty string body will be returned (useful for OPTIONS preflight 204 responses).
    """
    headers = {'Content-Type': content_type}
    headers.update(CORS_HEADERS)
    resp = {'statusCode': status_code, 'headers': headers}
    if body is None:
        resp['body'] = ''
    else:
        # If caller passed a dict/list, encode to JSON; otherwise coerce to string
        if isinstance(body, (dict, list)):
            resp['body'] = json.dumps(body)
        else:
            resp['body'] = str(body)
    # Log the response body for CloudWatch (safe to log - redact if needed)
    try:
        # If the body is a JSON string, parse it so we log an object instead of an escaped string
        raw_body = resp.get('body')
        parsed_body = None
        if isinstance(raw_body, str):
            try:
                parsed_body = json.loads(raw_body)
            except Exception:
                parsed_body = None

        if parsed_body is not None:
            # If the parsed body matches our API shape, log it exactly as returned
            try:
                # prefer to log with keys in order: status then data when present
                ordered = None
                if isinstance(parsed_body, dict) and 'status' in parsed_body and 'data' in parsed_body:
                    ordered = {'status': parsed_body.get('status'), 'data': parsed_body.get('data')}
                else:
                    ordered = parsed_body
                if _should_log():
                    logger.info('Response sent: %s', json.dumps(ordered, indent=2, default=str))
            except Exception:
                if _should_log():
                    logger.info('Response sent: %s', json.dumps(parsed_body, indent=2, default=str))
        else:
            log_resp = {'statusCode': status_code, 'body': raw_body}
            if _should_log():
                logger.info('Response sent: %s', json.dumps(log_resp))
    except Exception:
        logger.exception('Failed to log response')

    return resp


# Initialize logger for CloudWatch
logger = logging.getLogger('lambda_handler')
logger.setLevel(logging.INFO)


def _should_log():
    try:
        return os.getenv('SHOW_CLOUDWATCH_LOGS', 'false').lower() in ('1', 'true', 'yes')
    except Exception:
        return False

def _log_request(event, body_obj=None):
    try:
        request_context = event.get('requestContext', {})
        http = request_context.get('http') or {}
        path = http.get('path') if isinstance(http, dict) else event.get('rawPath') or event.get('path')
        method = http.get('method') if isinstance(http, dict) else event.get('httpMethod')
        log_obj = {
            'path': path,
            'method': method,
        }
        if body_obj is not None:
            log_obj['body'] = body_obj
        else:
            log_obj['body'] = event.get('body')
        if _should_log():
            logger.info('Request received: %s', json.dumps(log_obj))
    except Exception:
        logger.exception('Failed to log request')

def _connect_mongo():
    """Create a MongoDB client using ATLAS_URI from env.

    Raises RuntimeError if ATLAS_URI is missing or the connection cannot be established.
    Returns a pymongo.MongoClient on success.
    """
    atlas_uri = os.getenv('ATLAS_URI') + '?retryWrites=true&w=majority'
    if not atlas_uri:
        raise RuntimeError('ATLAS_URI environment variable is not set')
    try:
        client = pymongo.MongoClient(atlas_uri, serverSelectionTimeoutMS=5000)
        # attempt server selection
        client.admin.command('ping')
        return client
    except Exception as e:
        raise RuntimeError(f'Failed to connect to MongoDB: {e}')


def _process_document_attachment(attachment):
    """Process document attachment by calling OCR_ANALYZE_API_URL.
    
    Args:
        attachment: dict with 'url' and 'name' fields
        
    Returns:
        dict: OCR analysis result or None if processing fails
    """
    try:
        ocr_api_url = os.getenv('OCR_ANALYZE_API_URL')
        if not ocr_api_url:
            raise RuntimeError('OCR_ANALYZE_API_URL environment variable is not set')
        
        # Fetch image from URL
        response = requests.get(attachment['url'], timeout=30)
        response.raise_for_status()
        
        # Convert to base64
        file_content = base64.b64encode(response.content).decode('utf-8')
        
        # Prepare payload for OCR API
        payload = {
            'file_content': file_content,
            'filename': attachment['name']
        }
        
        # Call OCR API
        ocr_response = requests.post(
            ocr_api_url,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=60
        )
        ocr_response.raise_for_status()
        
        ocr_result = ocr_response.json()
        
        # Log OCR API response to CloudWatch
        if _should_log():
            logger.info('OCR API response for file %s: %s', 
                       attachment['name'], 
                       json.dumps(ocr_result, indent=2, default=str))
        
        return ocr_result
        
    except Exception as e:
        if _should_log():
            logger.error('Failed to process document attachment: %s', str(e))
        return None


def _save_document_context_to_session(user_id, session_id, ocr_result, attachment_name, processed_at_iso):
    """Save extracted document data to the session context in MongoDB.
    
    Args:
        user_id: User ID
        session_id: Session ID
        ocr_result: OCR analysis result
        attachment_name: Original attachment filename
        processed_at_iso: ISO timestamp for when the document was processed
    """
    try:
        client = _connect_mongo()
        db = client['chats']
        coll = db[user_id]
        
        # Prepare context data with extracted information
        extracted_data = ocr_result.get('extracted_data', {})
        category_detection = ocr_result.get('category_detection', {})
        
        # Sanitize filename to avoid MongoDB nested object issues (replace dots with underscores)
        sanitized_filename = attachment_name.replace('.', '_')
        
        # Update the session document's context with extracted data
        context_update = {
            f'context.document_{sanitized_filename}': {
                'extractedData': extracted_data,
                'categoryDetection': category_detection,
                'processedAt': processed_at_iso,
                'filename': attachment_name  # Keep original filename for reference
            }
        }
        
        coll.update_one(
            {'sessionId': session_id}, 
            {'$set': context_update}
        )
        
        if _should_log():
            logger.info('Saved document context to session: user=%s session=%s filename=%s', 
                       user_id, session_id, attachment_name)
        
    except Exception as e:
        if _should_log():
            logger.error('Failed to save document context to session: %s', str(e))
    finally:
        try:
            client.close()
        except Exception:
            pass


def _check_document_quality(ocr_result):
    """Check if document is blurry based on OCR analysis results.
    
    Args:
        ocr_result: OCR analysis result dict
        
    Returns:
        tuple: (is_blurry: bool, blur_message: str or None)
    """
    try:
        blur_analysis = ocr_result.get('blur_analysis', {})
        overall_assessment = blur_analysis.get('overall_assessment', {})
        is_blurry = overall_assessment.get('is_blurry', False)
        
        if is_blurry:
            return True, "The document image appears to be blurry or unclear. Please take a clearer photo and send the document again for better processing."
        
        return False, None
        
    except Exception as e:
        if _should_log():
            logger.error('Failed to check document quality: %s', str(e))
        return False, None


def _generate_document_analysis_prompt(ocr_result, user_message):
    """Generate appropriate prompt for document processing based on category detection.
    
    Args:
        ocr_result: OCR analysis result dict
        user_message: Original user message
        
    Returns:
        str: Generated prompt for the AI model
    """
    try:
        category_detection = ocr_result.get('category_detection', {})
        detected_category = category_detection.get('detected_category', 'unknown')
        confidence = category_detection.get('confidence', 0)
        
        extracted_data = ocr_result.get('extracted_data', {})
        text_content = ocr_result.get('text', [])
        
        # Extract meaningful text from OCR results
        text_parts = []
        for text_item in text_content:
            if isinstance(text_item, dict) and text_item.get('text'):
                text_parts.append(text_item['text'])
        
        extracted_text = ' '.join(text_parts) if text_parts else ''
        
        prompt_parts = [
            f"SYSTEM: You are processing a document for a government services portal (MyGovHub).",
            f"Document category detected: {detected_category} (confidence: {confidence:.2f})",
            f"Intent type: document_processing",
            ""
        ]
        
        if extracted_data:
            prompt_parts.append("Extracted structured data:")
            for key, value in extracted_data.items():
                prompt_parts.append(f"- {key}: {value}")
            prompt_parts.append("")
        
        if extracted_text:
            prompt_parts.append(f"Document text content: {extracted_text[:1000]}...")  # Limit length
            prompt_parts.append("")
        
        # Category-specific guidance
        category_guidance = {
            'receipt': "This appears to be a receipt. Help the user understand the transaction details and offer relevant government services like expense reporting or tax documentation.",
            'invoice': "This appears to be an invoice. Assist with business registration, tax filing, or payment verification services.",
            'license': "This appears to be a license document. Help with renewal processes, verification, or related permit applications.",
            'permit': "This appears to be a permit document. Assist with permit renewals, status checks, or related applications.",
            'identification': "This appears to be an identification document. Help with identity verification, document renewal, or related services.",
            'bill': "This appears to be a utility or service bill. Assist with bill payment services or account verification.",
            'form': "This appears to be a government form. Help with form completion, submission, or status tracking.",
        }
        
        guidance = category_guidance.get(detected_category, "Analyze the document and provide relevant assistance based on the content.")
        prompt_parts.append(f"Guidance: {guidance}")
        prompt_parts.append("")
        
        if user_message.strip():
            prompt_parts.append(f"User message: {user_message}")
        else:
            prompt_parts.append("User uploaded a document without additional message.")
        prompt_parts.append("")
        prompt_parts.append("Please provide a helpful response based on the document analysis and user's needs. Be specific about what MyGovHub services might be relevant.")
        prompt_parts.append("")
        prompt_parts.append("NOTE: If you include a signature, use 'MyGovHub Support Team' only. Do not use placeholders like '[Your Name]' or similar.")
        
        return '\n'.join(prompt_parts)
        
    except Exception as e:
        if _should_log():
            logger.error('Failed to generate document analysis prompt: %s', str(e))
        # Fallback prompt
        return f"SYSTEM: Document processing completed. User message: {user_message}. Please provide assistance based on the uploaded document."


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

    # Handle CORS preflight early: respond to OPTIONS with proper headers
    if method and method.upper() == 'OPTIONS':
        # 204 No Content is a lightweight preflight response
        return _cors_response(204, None)

    if path and path.endswith('/health') and (method or '').upper() == 'GET':
        return _cors_response(200, {'status': 'ok'})

    # Parse body for regular requests
    body = event.get('body')
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except Exception:
            _log_request(event)
            return _cors_response(400, {'error': 'Invalid JSON body'})

    if not isinstance(body, dict):
        _log_request(event)
        return _cors_response(400, {'error': 'Request body must be a JSON object'})

    # Log the request (include parsed body)
    _log_request(event, body)

    # Validate required fields
    user_id = body.get('userId')
    message = body.get('message', '')  # Default to empty string if not provided
    user_timestamp_z = body.get('createdAt')
    user_timestamp_iso = user_timestamp_z.replace('Z', '+00:00')
    session_id = body.get('sessionId')
    ekyc = body.get('ekyc') or {}
    attachments = body.get('attachment', [])

    # Allow empty message if there are attachments (document upload scenario)
    if not user_id or session_id is None:
        return _cors_response(400, {'error': "Missing required fields: 'userId' or 'sessionId'"})
    
    if not message and not attachments:
        return _cors_response(400, {'error': "Either 'message' or 'attachment' must be provided"})

    # Generate a messageId for this incoming message
    message_id = str(uuid.uuid4())
    # createdAt: UTC with millisecond precision and trailing Z, e.g. 2025-10-02T01:03:00.000Z
    dt = datetime.now(timezone.utc)
    created_at_iso = dt.isoformat()
    created_at_z = dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

    # If new session, create sessionId and initialize collection/document in MongoDB (required)
    new_session_generated = None
    try:
        client = _connect_mongo()
    except RuntimeError as e:
        return _cors_response(500, {'error': str(e)})

    try:
        db = client['chats']
        # Ensure the user's collection exists; create if missing
        if user_id not in db.list_collection_names():
            try:
                db.create_collection(user_id)
            except Exception:
                # If collection creation fails, it may already exist (race) or be unsupported
                pass
        coll = db[user_id]
        # Attempt to fetch existing session document so we can provide history/ekyc to the model
        session_doc = None
        if session_id and session_id != '(new-session)':
            try:
                if _should_log():
                    logger.info('Fetching session from MongoDB: user=%s sessionId=%s', user_id, session_id)
                session_doc = coll.find_one({'sessionId': session_id})
                if session_doc:
                    status_val = session_doc.get('status')
                    messages_count = len(session_doc.get('messages') or [])
                    has_ekyc = bool(session_doc.get('ekyc'))
                    if _should_log():
                        logger.info('Fetched session from MongoDB: user=%s sessionId=%s status=%s messages=%d ekyc=%s', user_id, session_id, status_val, messages_count, has_ekyc)
                    # Log the full session document from MongoDB (always)
                    try:
                        if _should_log():
                            logger.info('Full session document from MongoDB: %s', json.dumps(session_doc, default=str))
                    except Exception:
                        logger.exception('Failed to log full session document from MongoDB')
                else:
                    if _should_log():
                        logger.info('No session document found for user=%s sessionId=%s', user_id, session_id)
            except Exception:
                logger.exception('Error fetching session document for user=%s sessionId=%s', user_id, session_id)
                session_doc = None
        if session_id == '(new-session)':
            new_session_generated = str(uuid.uuid4())
            # Archive any other active sessions for this user
            try:
                coll.update_many({'status': 'active'}, {'$set': {'status': 'archived'}})
            except Exception:
                # Non-fatal: continue even if archiving fails (race or permissions)
                pass

            # Prepare the session document format
            session_doc = {
                'sessionId': new_session_generated,
                'createdAt': created_at_iso,
                'messages': [],
                'status': 'active',
                'topic': '',
                'context': {},
                'ekyc': ekyc or {}
            }
            # Insert the document
            coll.insert_one(session_doc)
        else:
            # existing session: update ekyc if provided
            update_ops = {}
            if ekyc:
                update_ops['ekyc'] = ekyc
            if update_ops:
                coll.update_one({'sessionId': session_id}, {'$set': update_ops})
            # If session_doc exists and is archived, return a restart message and instruct client to start a new session
            if session_doc and session_doc.get('status') == 'archived':
                special_msg = (
                    "It seems like you have another chat activate, please log out from the other device. "
                    "Conversation will be restarted."
                )
                resp_body = {
                    'status': {'statusCode': 200, 'message': 'Success'},
                    'data': {
                        'messageId': message_id,
                        'message': special_msg,
                        'createdAt': created_at_z,
                        'sessionId': '(new-session)',
                        'attachment': body.get('attachment') or []
                    }
                }
                return _cors_response(200, resp_body)
    finally:
        try:
            client.close()
        except Exception:
            pass

    # Check for document attachments and process them
    ocr_result = None
    intent_type = None
    
    if attachments:
        # Process the first attachment (image document)
        attachment = attachments[0]
        if attachment.get('url') and attachment.get('name'):
            if _should_log():
                logger.info('Processing document attachment: %s', attachment['name'])
            
            # Call OCR API to process the document
            ocr_result = _process_document_attachment(attachment)
            
            if ocr_result:
                # Check if document is blurry
                is_blurry, blur_message = _check_document_quality(ocr_result)
                
                if is_blurry:
                    # Return early with blur message
                    if _should_log():
                        logger.info('Document is blurry. Intent type: document_quality_issue')
                    resp_body = {
                        'status': {'statusCode': 200, 'message': 'Success'},
                        'data': {
                            'messageId': message_id,
                            'message': blur_message,
                            'createdAt': created_at_z,
                            'sessionId': session_id if session_id != '(new-session)' else new_session_generated,
                            'attachment': attachments,
                            'intent_type': 'document_quality_issue'
                        }
                    }
                    return _cors_response(200, resp_body)
                
                # Document is clear, set intent type and save context to session
                intent_type = 'document_processing'
                session_to_save = new_session_generated if new_session_generated else session_id
                _save_document_context_to_session(user_id, session_to_save, ocr_result, attachment['name'], created_at_iso)
                
                if _should_log():
                    logger.info('Document processed successfully. Category: %s, Intent type: %s', 
                               ocr_result.get('category_detection', {}).get('detected_category', 'unknown'), intent_type)

    # Determine prompt for Bedrock. For first-time connection, request a welcome message.
    try:
        if _should_log():
            logger.info('Generating prompt. Intent type: %s', intent_type or 'None')
            
        if session_id == '(new-session)':
            prompt = (
                "SYSTEM: You are a friendly assistant that composes a short welcome message "
                "for a government services portal called MyGovHub. The message MUST mention "
                "that MyGovHub provides these services: license renewal, bill payments, "
                "permit applications, checking application status, and accessing official documents. "
                "Keep it concise (max ~120 words), helpful, and end with a call-to-action such as "
                "'How can I help you today?'.\n\n"
                "IMPORTANT: Respond ONLY with the welcome message text (no JSON, no explanations, no metadata)."
            )
        elif intent_type == 'document_processing' and ocr_result:
            # Use document analysis prompt for processed documents
            prompt = _generate_document_analysis_prompt(ocr_result, message)
        else:
            # Build a contextual prompt using previous messages and ekyc (if available)
            parts = []
            # include any session-level ekyc data
            if session_doc and session_doc.get('ekyc'):
                try:
                    ekyc_str = json.dumps(session_doc.get('ekyc'))
                except Exception:
                    ekyc_str = str(session_doc.get('ekyc'))
                parts.append(f"EKYC: {ekyc_str}\n")

            # include prior messages in chronological order
            if session_doc and isinstance(session_doc.get('messages'), list):
                for m in session_doc.get('messages'):
                    role = m.get('role', 'user')
                    # messages content expected to be a list of objects with 'text'
                    content_parts = []
                    for c in m.get('content', []):
                        text = c.get('text') if isinstance(c, dict) else str(c)
                        if text:
                            content_parts.append(str(text))
                    if content_parts:
                        parts.append(f"{role.upper()}: {' '.join(content_parts)}\n")

            # finally append the current user's message
            parts.append(f"USER: {message}\n")
            # join into one prompt string
            prompt = "\n".join(parts)

        model_error = None
        response_text = None
        try:
            response_text = run_agent(prompt)
        except Exception as model_exc:
            # Record the model failure but continue — we'll persist an assistant error message
            model_error = str(model_exc)
            print('Model invocation failed:', model_error)

        # Persist the conversation: always push user message first, then assistant or error message
        session_to_update = new_session_generated if new_session_generated else session_id
        try:
            client2 = _connect_mongo()
            db2 = client2['chats']
            coll2 = db2[user_id]

            # push the user message (always)
            user_msg_doc = {
                'messageId': message_id,
                'message': str(message),
                'timestamp': user_timestamp_iso,
                'type': 'user',
                'role': 'user',
                'content': [{'text': str(message)}]
            }
            
            # Add attachment and intent if present
            if attachments:
                user_msg_doc['attachment'] = attachments
            if intent_type:
                user_msg_doc['intent'] = intent_type
                
            coll2.update_one({'sessionId': session_to_update}, {'$push': {'messages': user_msg_doc}}, upsert=True)

            # push the assistant message; if model failed, store an error message as assistant reply
            assistant_message_id = str(uuid.uuid4())
            # Generate new timestamp for assistant response (when we actually respond)
            assistant_timestamp = datetime.now(timezone.utc)
            assistant_timestamp_iso = assistant_timestamp.isoformat()
            assistant_timestamp_z = assistant_timestamp_iso.replace('+00:00', 'Z')
            
            if response_text is not None:
                assistant_msg_doc = {
                    'messageId': assistant_message_id,
                    'message': str(response_text),
                    'timestamp': assistant_timestamp_iso,
                    'type': 'assistant',
                    'role': 'assistant',
                    'content': [{'text': str(response_text)}]
                }
            else:
                assistant_msg_doc = {
                    'messageId': assistant_message_id,
                    'message': 'ERROR: assistant failed to respond. See modelError in response.',
                    'timestamp': assistant_timestamp_iso,
                    'type': 'assistant',
                    'role': 'assistant',
                    'content': [{'text': 'ERROR: assistant failed to respond. See modelError in response.'}],
                    'meta': {'modelError': model_error}
                }

            coll2.update_one({'sessionId': session_to_update}, {'$push': {'messages': assistant_msg_doc}}, upsert=True)
        except Exception as e:
            # If persisting conversation fails, return 500 to enforce durability and include traceback for debugging
            tb = traceback.format_exc()
            print('Failed to persist conversation:', str(e))
            print(tb)
            try:
                if 'client2' in locals() and client2:
                    client2.close()
            except Exception:
                pass
            return _cors_response(500, {'error': f'Failed to persist conversation: {str(e)}', 'trace': tb})
        finally:
            try:
                if 'client2' in locals() and client2:
                    client2.close()
            except Exception:
                pass

        # Prepare the MCP response payload. If model failed, still return 200 but include modelError flag
        resp_body = {
            'status': {'statusCode': 200, 'message': 'Success'},
            'data': {
                'messageId': message_id,
                'message': response_text if response_text is not None else 'ERROR: assistant failed to respond',
                'createdAt': assistant_timestamp_z,
                'sessionId': session_to_update,
                'attachment': body.get('attachment') or []
            }
        }

        if intent_type:
            resp_body['data']['intent_type'] = intent_type
            if _should_log():
                logger.info('Final response includes intent_type: %s', intent_type)

        if model_error:
            resp_body['data']['modelError'] = model_error

        # successful response
        return _cors_response(200, resp_body)
    except Exception as e:
        # print traceback to CloudWatch and return it in the response for easier debugging
        tb = traceback.format_exc()
        print('Handler exception:', str(e))
        print(tb)
        return _cors_response(500, {'error': str(e), 'trace': tb})
