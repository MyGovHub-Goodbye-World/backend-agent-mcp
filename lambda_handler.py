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


def _normalize_ic(value: str) -> str:
    """Normalize Malaysian IC / identity numbers for comparison.

    Removes all non-alphanumeric characters, uppercases the string and returns only
    the digits/letters sequence. Safe for None or empty inputs (returns empty string).
    Example: '041223-07-0745' -> '041223070745'
    """
    if not value:
        return ""
    import re
    # Keep digits and letters only (primarily digits for IC) and uppercase.
    cleaned = re.sub(r"[^0-9A-Za-z]", "", str(value))
    return cleaned.upper()


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


def _service_requirements_met(service_name: str, session_doc: dict) -> bool:
    """Check if required verified fields exist for a given service.

    renew_license requires: a verified document containing full_name AND userId (IC number).
    pay_tnb_bill requires: a verified document containing account_number AND invoice_number.
    Returns True when requirements satisfied, False otherwise.
    """
    if not service_name or not session_doc:
        return False
    ctx = (session_doc.get('context') or {})

    # Must have at least one document_* entry in context overall
    document_keys = [k for k in ctx.keys() if k.startswith('document_')]
    if not document_keys:
        return False

    # Iterate documents to find a fully verified one with required fields & category constraints
    required_category_sets = {
        'renew_license': {'allowed': {'idcard', 'license', 'license-front'}},
        'pay_tnb_bill': {'allowed': {'tnb'}},
    }

    for key, doc_meta in ctx.items():
        if not key.startswith('document_'):
            continue
        if doc_meta.get('isVerified') != 'verified':
            continue
        extracted = doc_meta.get('extractedData') or {}

        # Category detection path can vary; attempt to read nested detection structure resiliently
        detected_category = None
        try:
            cat_obj = doc_meta.get('categoryDetection') or {}
            detected_category = cat_obj.get('detected_category')
        except Exception:
            detected_category = None

        if service_name == 'renew_license':
            # Field requirements
            has_fields = extracted.get('full_name') and extracted.get('userId')
            # Category requirement: at least one of allowed categories
            if has_fields:
                if detected_category in required_category_sets['renew_license']['allowed']:
                    return True
                # Allow pass-through if category unknown but fields exist (optional: tighten later)
        elif service_name == 'pay_tnb_bill':
            has_fields = extracted.get('account_number') and extracted.get('invoice_number')
            if has_fields:
                # Strict: must have tnb category
                if detected_category in required_category_sets['pay_tnb_bill']['allowed']:
                    return True
    return False


def _build_service_next_step_message(service_name: str, user_id: str, session_id: str, session_doc: dict) -> str:
    """Return next-step text after identity/document verification for a service.

    Enhancements:
      - For renew_license: fetch license record from MongoDB `licenses` collection using userId.
        Store (without _id) under context.database_license if retrieved.
        License status handling:
          * suspended -> instruct physical branch visit (cannot renew here)
          * active or expired -> ask user to confirm proceeding with renewal (extending validity)
      - For pay_tnb_bill: keep placeholder (future: fetch bill details).
    """
    service_name = service_name or ''

    if service_name == 'renew_license':
        db_name = os.getenv('ATLAS_DB_NAME') or ''
        if not db_name:
            logger.error("License verification complete, but database name not configured. Please set ATLAS_DB_NAME environment variable.")
            return "Identity verified, but I couldn't retrieve your license record right now. Please try again shortly or provide more details."
        license_record = None
        record_for_context = None
        try:
            client = _connect_mongo()
            try:
                # Fetch license
                lic_coll = client[db_name]['licenses']
                license_record = lic_coll.find_one({'userId': user_id})
                if _should_log():
                    logger.info('License lookup userId=%s found=%s', user_id, bool(license_record))
                if not license_record:
                    return (
                        "Identity verified, but I didn't find an existing driving license record for your IC. "
                        "Please visit the nearest JPJ Malaysia branch to apply for a new license."
                    )
                
                # Prepare record (strip _id)
                record_for_context = {k: v for k, v in license_record.items() if k != '_id'}

                # Update session context
                try:
                    chats_db = client['chats']
                    user_coll = chats_db[user_id]
                    user_coll.update_one({'sessionId': session_id}, {'$set': {'context.database_license': record_for_context}})
                    if _should_log():
                        logger.info('Stored license record in session context sessionId=%s', session_id)
                except Exception:
                    if _should_log():
                        logger.exception('Failed to persist license record into session context')
            finally:
                try:
                    client.close()
                except Exception:
                    pass
        except Exception as e:
            if _should_log():
                logger.exception('License retrieval/update failure: %s', str(e))
            return "Identity verified, but I couldn't retrieve your license record right now. Please try again shortly or provide more details."

        # Check current workflow state from session
        workflow_state = None
        try:
            client_state = _connect_mongo()
            chats_db = client_state['chats']
            user_coll = chats_db[user_id]
            current_session = user_coll.find_one({'sessionId': session_id})
            if current_session and current_session.get('context'):
                workflow_state = current_session['context'].get('renewal_workflow_state')
            client_state.close()
        except Exception:
            pass
        
        # Use record_for_context for message composition
        status = (record_for_context or {}).get('status')
        valid_from = (record_for_context or {}).get('valid_from')
        valid_to = (record_for_context or {}).get('valid_to')
        license_number = (record_for_context or {}).get('license_number')

        if status == 'suspended':
            return (
                "We located your driving license record (License No: {ln}). Current status: SUSPENDED. "
                "Suspended licenses must be handled at a physical branch for investigation or reinstatement. "
                "Please visit the nearest JPJ Malaysia branch to resolve the suspension before renewal.".format(ln=license_number or 'N/A')
            )

        # Handle different workflow states
        if workflow_state == 'license_confirmed':
            # User confirmed, now ask for renewal duration using AI
            # Set workflow state and return AI prompt instruction
            try:
                client_workflow = _connect_mongo()
                chats_db = client_workflow['chats']
                user_coll = chats_db[user_id]
                user_coll.update_one(
                    {'sessionId': session_id}, 
                    {'$set': {'context.renewal_workflow_state': 'asking_duration'}}
                )
                client_workflow.close()
            except Exception:
                pass
            
            # Return prompt for AI to generate duration options
            return (
                "SYSTEM: Generate a license renewal duration selection prompt titled 'License Renewal Duration'. "
                f"Current license expires: {valid_to or 'N/A'}. "
                "Create a friendly message asking how many years to extend (1-10 years). "
                "Include pricing: RM 30.00 per year. Show options 1-10 with calculated prices in 2 decimals. "
                "Use emojis and clear formatting. Ask user to reply with number of years."
            )
        else:
            # First time or default - show license info and ask for confirmation
            # Set workflow state to track that we've shown license info
            try:
                client_workflow = _connect_mongo()
                chats_db = client_workflow['chats']
                user_coll = chats_db[user_id]
                user_coll.update_one(
                    {'sessionId': session_id}, 
                    {'$set': {'context.renewal_workflow_state': 'license_shown'}}
                )
                client_workflow.close()
            except Exception:
                pass
            
            return (
                f"We found your driving license record:\n\n"
                f"License No: {license_number or 'N/A'}\n"
                f"Valid from: {valid_from or 'N/A'} to {valid_to or 'N/A'}\n"
                f"Status: {status.upper() if status else 'N/A'}\n\n"
                "I can help extend your license validity. Are you sure you want to proceed with renewal?"
            )

    if service_name == 'pay_tnb_bill':
        return (
            "Bill details verified (account + invoice number). "
            "@TODO: Retrieve latest bill amount from MongoDB or billing service and ask user to confirm payment amount."
        )
    return "Service data verified. @TODO: implement next workflow steps."

def _detect_service_intent(message_lower: str):
    """Detect high-level service intents from a free-form user message.

    Returns one of: 'renew_license', 'pay_tnb_bill' or None.
    Detection is conservative: requires presence of key verbs + domain terms.
    """
    if not message_lower:
        return None

    # Driving license renewal
    if any(k in message_lower for k in ['renew', 'renewal', 'renewing']) and \
       any(k in message_lower for k in ['license', 'driving license', 'lesen', 'driver license']):
        return 'renew_license'

    # TNB bill payment (Tenaga Nasional Berhad - Malaysia electric utility)
    if any(k in message_lower for k in ['pay', 'payment', 'bayar']) and \
       any(k in message_lower for k in ['tnb', 'electric', 'electricity', 'bill', 'bil elektrik']):
        return 'pay_tnb_bill'

    return None

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


def _save_document_context_to_session(user_id, session_id, ocr_result, attachment_name):
    """Save extracted document data to the session context in MongoDB.
    
    Args:
        user_id: User ID
        session_id: Session ID
        ocr_result: OCR analysis result
        attachment_name: Original attachment filename
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
                'filename': attachment_name,
                # Tri-state string: 'unverified' | 'correcting' | 'verified'
                'isVerified': 'unverified'
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
            f"NOTE: The 'userId' field represents the Identity Card (IC) number.",
            ""
        ]
        
        if extracted_data:
            prompt_parts.append("Extracted structured data (show with user-friendly labels):")
            # Field mapping for user-friendly display
            field_mapping = {
                'full_name': 'Full Name',
                'userId': 'IC Number',
                'gender': 'Gender', 
                'address': 'Address',
                'licenses_number': 'License Number',
                'account_number': 'Account Number',
                'invoice_number': 'Invoice Number'
            }
            
            for key, value in extracted_data.items():
                friendly_name = field_mapping.get(key, key.replace('_', ' ').title())
                prompt_parts.append(f"- {friendly_name}: {value}")
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
        prompt_parts.append("IMPORTANT: Keep your response concise. Show ONLY the extracted key information in a simple format.")
        prompt_parts.append("After showing the data, ask: 'Is this information correct? Please reply YES to confirm.'")
        prompt_parts.append("Do not repeat the information multiple times or add lengthy explanations.")
        prompt_parts.append("")
        prompt_parts.append("NOTE: If you include a signature, use 'MyGovHub Support Team' only. Do not use placeholders like '[Your Name]' or similar.")
        
        return '\n'.join(prompt_parts)
        
    except Exception as e:
        if _should_log():
            logger.error('Failed to generate document analysis prompt: %s', str(e))
        # Fallback prompt
        return f"SYSTEM: Document processing completed. User message: {user_message}. Please provide assistance based on the uploaded document."


def _parse_document_corrections(message: str, current_data: dict) -> dict:
    """Parse user free-form correction text into field->value mapping.

    Supports patterns like:
      - "wrong, full name is lim wen hau"
      - "full_name: LIM WEN HAU"
      - "name should be LIM WEN HAU"
      - "IC is 041223-07-0745"
      - Multiple lines each containing a correction.

    The parser is tolerant of punctuation and case. It attempts fuzzy matching
    against existing keys in current_data and known synonyms. Returns only
    fields that can be confidently matched.
    """
    import re
    if not message or not current_data:
        return {}

    original_message = message
    # Normalize spacing
    message = re.sub(r"\s+", " ", message.strip())

    # Lower copy for pattern detection while we keep original for value extraction
    lower_msg = message.lower()

    # Split into candidate segments (newline, ' and ', commas used as delimiters)
    # Keep semicolons and periods as potential delimiters when followed by space
    segments = re.split(r"[\n;,]+|\band\b", message, flags=re.IGNORECASE)

    # Known synonym lists
    synonyms = {
        'full_name': ['full name', 'name', 'nama'],
        'userId': ['ic', 'ic number', 'id number', 'id', 'userid', 'identity card'],
        'gender': ['gender', 'sex', 'jantina'],
        'address': ['address', 'alamat', 'location'],
        'licenses_number': ['license', 'license number', 'lesen'],
        'account_number': ['account', 'account number'],
        'invoice_number': ['invoice', 'invoice number']
    }

    # Reverse map for quick lookup
    synonym_to_field = {}
    for field, words in synonyms.items():
        for w in words:
            synonym_to_field[w] = field

    # Helper to resolve a raw field token to actual existing field
    def resolve_field(token: str):
        t = token.lower().strip(': ').strip()
        # Exact existing key
        for k in current_data.keys():
            if t == k.lower():
                return k
        # Direct synonym
        if t in synonym_to_field:
            mapped = synonym_to_field[t]
            # prefer existing key if present
            for k in current_data.keys():
                if k.lower() == mapped.lower():
                    return k
            return mapped
        # Partial match within existing keys
        for k in current_data.keys():
            if t in k.lower() or k.lower() in t:
                return k
        return None

    corrections = {}

    # Pattern variants to attempt per segment
    pattern_specs = [
        # field: value
        re.compile(r"^(?P<field>[A-Za-z_ ]{2,30})\s*[:=-]\s*(?P<value>.+)$"),
        # field should be value
        re.compile(r"^(?P<field>[A-Za-z_ ]{2,30})\s+should\s+be\s+(?P<value>.+)$", re.IGNORECASE),
        # field is value
        re.compile(r"^(?P<field>[A-Za-z_ ]{2,30})\s+is\s+(?P<value>.+)$", re.IGNORECASE),
        # wrong, field is value OR wrong field is value
        re.compile(r"^(?:wrong[, ]+)?(?P<field>[A-Za-z_ ]{2,30})\s+is\s+(?P<value>.+)$", re.IGNORECASE),
        # fix field to value / change field to value / update field to value
        re.compile(r"^(?:fix|change|update)\s+(?P<field>[A-Za-z_ ]{2,30})\s+(?:to|as)\s+(?P<value>.+)$", re.IGNORECASE),
    ]

    for raw_segment in segments:
        segment = raw_segment.strip()
        if not segment:
            continue
        # Remove leading qualifiers
        segment = re.sub(r"^(wrong|no|not|incorrect)[, ]+", "", segment, flags=re.IGNORECASE)
        matched = False
        for pat in pattern_specs:
            m = pat.match(segment)
            if m:
                field_token = m.group('field').strip()
                value = m.group('value').strip()
                resolved = resolve_field(field_token)
                if resolved and value:
                    corrections[resolved] = value
                matched = True
                break
        if matched:
            continue
        # Heuristic: "full name is abc" inside longer sentence
        for field_key in current_data.keys():
            # Search pattern like '<synonym> is <value>'
            for syn in [field_key] + synonyms.get(field_key, []):
                syn_lower = syn.lower()
                idx = segment.lower().find(f"{syn_lower} is ")
                if idx != -1:
                    val = segment[idx + len(syn_lower) + 4:].strip()
                    if val:
                        corrections[field_key] = val
                        break

    # Normalize whitespace of values
    for k, v in list(corrections.items()):
        corrections[k] = re.sub(r"\s+", " ", v).strip()

    return corrections


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


def _save_document_context_to_session(user_id, session_id, ocr_result, attachment_name):
    """Save extracted document data to the session context in MongoDB.
    
    Args:
        user_id: User ID
        session_id: Session ID
        ocr_result: OCR analysis result
        attachment_name: Original attachment filename
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
                'filename': attachment_name,  # Keep original filename for reference
                'isVerified': 'unverified'  # Requires user verification before proceeding
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
            f"NOTE: The 'userId' field represents the Identity Card (IC) number.",
            ""
        ]
        
        if extracted_data:
            prompt_parts.append("Extracted structured data (show with user-friendly labels):")
            # Field mapping for user-friendly display
            field_mapping = {
                'full_name': 'Full Name',
                'userId': 'IC Number',
                'gender': 'Gender', 
                'address': 'Address',
                'licenses_number': 'License Number',
                'account_number': 'Account Number',
                'invoice_number': 'Invoice Number'
            }
            
            for key, value in extracted_data.items():
                friendly_name = field_mapping.get(key, key.replace('_', ' ').title())
                prompt_parts.append(f"- {friendly_name}: {value}")
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
        prompt_parts.append("IMPORTANT: Keep your response concise. Show ONLY the extracted key information in a simple format.")
        prompt_parts.append("After showing the data, ask: 'Is this information correct? Please reply YES to confirm.'")
        prompt_parts.append("Do not repeat the information multiple times or add lengthy explanations.")
        prompt_parts.append("")
        prompt_parts.append("NOTE: If you include a signature, use 'MyGovHub Support Team' only. Do not use placeholders like '[Your Name]' or similar.")
        
        return '\n'.join(prompt_parts)
        
    except Exception as e:
        if _should_log():
            logger.error('Failed to generate document analysis prompt: %s', str(e))
        # Fallback prompt
        return f"SYSTEM: Document processing completed. User message: {user_message}. Please provide assistance based on the uploaded document."


def _parse_document_corrections(message: str, current_data: dict) -> dict:
    """Parse user free-form correction text into field->value mapping.

    Supports patterns like:
      - "wrong, full name is lim wen hau"
      - "full_name: LIM WEN HAU"
      - "name should be LIM WEN HAU"
      - "IC is 041223-07-0745"
      - Multiple lines each containing a correction.

    The parser is tolerant of punctuation and case. It attempts fuzzy matching
    against existing keys in current_data and known synonyms. Returns only
    fields that can be confidently matched.
    """
    import re
    if not message or not current_data:
        return {}

    original_message = message
    # Normalize spacing
    message = re.sub(r"\s+", " ", message.strip())

    # Lower copy for pattern detection while we keep original for value extraction
    lower_msg = message.lower()

    # Split into candidate segments (newline, ' and ', commas used as delimiters)
    # Keep semicolons and periods as potential delimiters when followed by space
    segments = re.split(r"[\n;,]+|\band\b", message, flags=re.IGNORECASE)

    # Known synonym lists
    synonyms = {
        'full_name': ['full name', 'name', 'nama'],
        'userId': ['ic', 'ic number', 'id number', 'id', 'userid', 'identity card'],
        'gender': ['gender', 'sex', 'jantina'],
        'address': ['address', 'alamat', 'location'],
        'licenses_number': ['license', 'license number', 'lesen'],
        'account_number': ['account', 'account number'],
        'invoice_number': ['invoice', 'invoice number']
    }

    # Reverse map for quick lookup
    synonym_to_field = {}
    for field, words in synonyms.items():
        for w in words:
            synonym_to_field[w] = field

    # Helper to resolve a raw field token to actual existing field
    def resolve_field(token: str):
        t = token.lower().strip(': ').strip()
        # Exact existing key
        for k in current_data.keys():
            if t == k.lower():
                return k
        # Direct synonym
        if t in synonym_to_field:
            mapped = synonym_to_field[t]
            # prefer existing key if present
            for k in current_data.keys():
                if k.lower() == mapped.lower():
                    return k
            return mapped
        # Partial match within existing keys
        for k in current_data.keys():
            if t in k.lower() or k.lower() in t:
                return k
        return None

    corrections = {}

    # Pattern variants to attempt per segment
    pattern_specs = [
        # field: value
        re.compile(r"^(?P<field>[A-Za-z_ ]{2,30})\s*[:=-]\s*(?P<value>.+)$"),
        # field should be value
        re.compile(r"^(?P<field>[A-Za-z_ ]{2,30})\s+should\s+be\s+(?P<value>.+)$", re.IGNORECASE),
        # field is value
        re.compile(r"^(?P<field>[A-Za-z_ ]{2,30})\s+is\s+(?P<value>.+)$", re.IGNORECASE),
        # wrong, field is value OR wrong field is value
        re.compile(r"^(?:wrong[, ]+)?(?P<field>[A-Za-z_ ]{2,30})\s+is\s+(?P<value>.+)$", re.IGNORECASE),
        # fix field to value / change field to value / update field to value
        re.compile(r"^(?:fix|change|update)\s+(?P<field>[A-Za-z_ ]{2,30})\s+(?:to|as)\s+(?P<value>.+)$", re.IGNORECASE),
    ]

    for raw_segment in segments:
        segment = raw_segment.strip()
        if not segment:
            continue
        # Remove leading qualifiers
        segment = re.sub(r"^(wrong|no|not|incorrect)[, ]+", "", segment, flags=re.IGNORECASE)
        matched = False
        for pat in pattern_specs:
            m = pat.match(segment)
            if m:
                field_token = m.group('field').strip()
                value = m.group('value').strip()
                resolved = resolve_field(field_token)
                if resolved and value:
                    corrections[resolved] = value
                matched = True
                break
        if matched:
            continue
        # Heuristic: "full name is abc" inside longer sentence
        for field_key in current_data.keys():
            # Search pattern like '<synonym> is <value>'
            for syn in [field_key] + synonyms.get(field_key, []):
                syn_lower = syn.lower()
                idx = segment.lower().find(f"{syn_lower} is ")
                if idx != -1:
                    val = segment[idx + len(syn_lower) + 4:].strip()
                    if val:
                        corrections[field_key] = val
                        break

    # Normalize whitespace of values
    for k, v in list(corrections.items()):
        corrections[k] = re.sub(r"\s+", " ", v).strip()

    return corrections


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
                'service': '',  # service identifier e.g. renew_license, pay_tnb_bill
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
    
    # Check document verification status and handle user responses
    verification_status = None
    unverified_doc_key = None
    unverified_doc_data = None
    
    # Find documents needing verification (isVerified tri-state string). Migrate legacy boolean if encountered.
    if session_doc and session_doc.get('context'):
        migrate_updates = {}
        for key, doc_data in session_doc['context'].items():
            if not key.startswith('document_'):
                continue
            val = doc_data.get('isVerified')
            if isinstance(val, bool):  # legacy boolean -> map
                new_val = 'verified' if val else 'unverified'
                doc_data['isVerified'] = new_val
                migrate_updates[f'context.{key}.isVerified'] = new_val
                val = new_val
            if val in ('unverified', 'correcting') and unverified_doc_key is None:
                unverified_doc_key = key
                unverified_doc_data = doc_data
        if migrate_updates:
            try:
                client_mig = _connect_mongo()
                db_mig = client_mig['chats']
                coll_mig = db_mig[user_id]
                session_to_mig = new_session_generated if new_session_generated else session_id
                coll_mig.update_one({'sessionId': session_to_mig}, {'$set': migrate_updates})
                if _should_log():
                    logger.info('Migrated legacy boolean isVerified to tri-state: %s', migrate_updates)
            except Exception as e:
                if _should_log():
                    logger.error('Migration failure: %s', str(e))
            finally:
                try:
                    client_mig.close()
                except Exception:
                    pass
    
    # Handle verification responses
    message_lower = message.lower().strip()
    
    def _has_field_pattern(msg: str) -> bool:
        field_synonyms = ['name', 'full name', 'ic', 'ic number', 'gender', 'address', 'license', 'account', 'invoice']
        return any(f" {syn} " in msg or msg.startswith(f"{syn} ") for syn in field_synonyms)

    def _is_affirmative(msg: str) -> bool:
        # Accept short pure confirmations only; reject if appears to contain field corrections
        aff_tokens = {'yes', 'ya', 'y', 'ok', 'okay', 'correct', 'accurate', 'looks good', 'betul', 'ya betul'}
        cleaned = msg.strip().lower()
        if len(cleaned) <= 15 and cleaned in aff_tokens:
            return True
        # Multi-word accept if all tokens in affirmative set
        if all(t in aff_tokens for t in cleaned.replace('!', '').split()):
            return True
        return False
    
    # Handle service workflow state transitions
    def _update_service_workflow_state(new_state: str):
        """Update the service workflow state in session context"""
        if not active_service:
            return
        try:
            client_workflow = _connect_mongo()
            chats_db = client_workflow['chats']
            user_coll = chats_db[user_id]
            session_to_update = new_session_generated if new_session_generated else session_id
            user_coll.update_one(
                {'sessionId': session_to_update}, 
                {'$set': {'context.renewal_workflow_state': new_state}}
            )
            if _should_log():
                logger.info('Updated service workflow state to: %s', new_state)
            client_workflow.close()
        except Exception as e:
            if _should_log():
                logger.error('Failed to update workflow state: %s', str(e))
    
    # Order: explicit rejection -> corrections -> affirmation
    # Rejection (needs corrections)
    if unverified_doc_key and message_lower in ['no', 'incorrect', 'wrong', 'not correct', 'not accurate']:
        intent_type = 'document_correction_needed'
        verification_status = 'rejected'
        # Set status to correcting
        try:
            client_status = _connect_mongo()
            db_status = client_status['chats']
            coll_status = db_status[user_id]
            session_to_status = new_session_generated if new_session_generated else session_id
            coll_status.update_one({'sessionId': session_to_status}, {'$set': {f'context.{unverified_doc_key}.isVerified': 'correcting'}})
        except Exception:
            pass
        finally:
            try:
                client_status.close()
            except Exception:
                pass
    # Corrections detection
    elif unverified_doc_key:
        current_data = unverified_doc_data.get('extractedData', {}) if unverified_doc_data else {}
        parsed_corrections_probe = _parse_document_corrections(message, current_data) if current_data else {}
        if parsed_corrections_probe:
            intent_type = 'document_correction_provided'
            verification_status = 'correcting'
            if _should_log():
                logger.info('Parsed corrections found pre-classification: %s', parsed_corrections_probe)
        # Affirmation only if no corrections parsed and message is simple confirm
        elif _is_affirmative(message_lower) and not _has_field_pattern(f' {message_lower} '):
            intent_type = 'document_verified'
            verification_status = 'confirmed'
    # Legacy path (affirmation first) kept for cases without document
    elif any(keyword in message_lower for keyword in ['yes']) and not any(negative in message_lower for negative in ['no', 'not', 'wrong', 'incorrect']):
        if unverified_doc_key:
            intent_type = 'document_verified'
            verification_status = 'confirmed'
    # Apply verification update if classified as verified (after corrections flow)
    if intent_type == 'document_verified' and unverified_doc_key:
        try:
            client_verify = _connect_mongo()
            db_verify = client_verify['chats']
            coll_verify = db_verify[user_id]
            # Merge any pending correctedData into extractedData atomically
            session_to_verify = new_session_generated if new_session_generated else session_id
            doc_for_merge = coll_verify.find_one({'sessionId': session_to_verify}, {f'context.{unverified_doc_key}': 1}) or {}
            doc_context_obj = (doc_for_merge.get('context') or {}).get(unverified_doc_key, {})
            pending_corr = doc_context_obj.get('correctedData') or {}
            set_ops = {f'context.{unverified_doc_key}.isVerified': 'verified'}
            # Prepare field updates only if corrections exist
            for k, v in pending_corr.items():
                set_ops[f'context.{unverified_doc_key}.extractedData.{k}'] = v
            update_doc = {'$set': set_ops, '$unset': {f'context.{unverified_doc_key}.correctedData': ""}}
            coll_verify.update_one({'sessionId': session_to_verify}, update_doc)
            if _should_log():
                logger.info('Document verified and corrections merged (status updated): %s merged_fields=%s', unverified_doc_key, list(pending_corr.keys()))
        except Exception as e:
            if _should_log():
                logger.error('Failed to update document verification status: %s', str(e))
        finally:
            try:
                client_verify.close()
            except Exception:
                pass

    # If corrections provided branch (reparsed inside branch to capture corrections precisely)
    if unverified_doc_key and intent_type == 'document_correction_provided':
        # User is providing corrections (flexible detection)
        if _should_log():
            logger.info('Applying corrections for document: %s', unverified_doc_key)
        try:
            client_correct = _connect_mongo()
            db_correct = client_correct['chats']
            coll_correct = db_correct[user_id]
            current_data = unverified_doc_data.get('extractedData', {})
            raw_corrections = _parse_document_corrections(message, current_data)
            corrections_made = {}
            for field, corrected_value in raw_corrections.items():
                # Strip trailing filler phrases like 'others correct'
                import re as _re
                cleaned_val = _re.sub(r"\b(others?|the rest)( are| is)?( all)? (correct|ok|okay|right)\b", "", corrected_value, flags=_re.IGNORECASE).strip()
                original_value = current_data.get(field, '')
                formatted_value = cleaned_val
                if original_value and original_value.isupper():
                    formatted_value = cleaned_val.upper()
                elif original_value and original_value.islower():
                    formatted_value = cleaned_val.lower()
                elif original_value and original_value.istitle():
                    formatted_value = cleaned_val.title()
                corrections_made[field] = formatted_value
                if _should_log():
                    logger.info('Correction parsed - %s: "%s" -> "%s"', field, original_value, formatted_value)
            if corrections_made:
                session_to_correct = new_session_generated if new_session_generated else session_id
                # Store corrections separately; do NOT merge into extractedData yet
                coll_correct.update_one({'sessionId': session_to_correct}, {
                    '$set': {
                        f'context.{unverified_doc_key}.correctedData': corrections_made,
                        f'context.{unverified_doc_key}.isVerified': 'correcting'
                    }
                })
                # Refresh local reference for prompt generation later (keep extractedData original)
                unverified_doc_data['correctedData'] = corrections_made
                unverified_doc_data['isVerified'] = 'correcting'
            else:
                if _should_log():
                    logger.warning('No corrections could be parsed from message (intent kept).')
        except Exception as e:
            if _should_log():
                logger.error('Error applying corrections: %s', str(e))
        finally:
            try:
                client_correct.close()
            except Exception:
                pass
    
    # --------------------------------------------------------------
    # Service intent detection (only if no document-processing intent determined)
    # --------------------------------------------------------------
    service_intent = None
    AVAILABLE_SERVICE_INTENTS = ['renew_license', 'pay_tnb_bill']

    # Determine active service (persisted) irrespective of current message intent
    active_service = None

    if not intent_type and attachments == []:  # pure text request
        service_intent = _detect_service_intent(message_lower)
        # Only set intent_type for NEW service requests, not when service is already active
        if service_intent == 'renew_license' and not active_service:
            intent_type = 'renew_license'
        elif service_intent == 'pay_tnb_bill' and not active_service:
            intent_type = 'pay_tnb_bill'

    # If we have a NEW service intent (not already active), update session 'service' field
    if service_intent in AVAILABLE_SERVICE_INTENTS and not active_service:
        try:
            client_service = _connect_mongo()
            db_service = client_service['chats']
            coll_service = db_service[user_id]
            session_to_service = new_session_generated if new_session_generated else session_id
            coll_service.update_one({'sessionId': session_to_service}, {'$set': {'service': service_intent}})
        except Exception:
            pass
        finally:
            try:
                client_service.close()
            except Exception:
                pass

    # Refresh session_doc (may have been updated earlier) only if we need service evaluation
    if (intent_type in (None, 'document_verified')) or (not intent_type and service_intent):
        try:
            client_refetch = _connect_mongo()
            db_refetch = client_refetch['chats']
            coll_refetch = db_refetch[user_id]
            session_current_id = new_session_generated if new_session_generated else session_id
            session_doc = coll_refetch.find_one({'sessionId': session_current_id}) or session_doc
        except Exception:
            pass
        finally:
            try:
                client_refetch.close()
            except Exception:
                pass

    if session_doc:
        active_service = session_doc.get('service') or None

    # Check for service-specific confirmations (when service is active and user says yes)
    if active_service == 'renew_license' and _is_affirmative(message_lower) and not unverified_doc_key:
        # Check current workflow state
        current_workflow_state = None
        try:
            client_check_state = _connect_mongo()
            chats_db = client_check_state['chats']
            user_coll = chats_db[user_id]
            session_current = new_session_generated if new_session_generated else session_id
            current_session = user_coll.find_one({'sessionId': session_current})
            if current_session and current_session.get('context'):
                current_workflow_state = current_session['context'].get('renewal_workflow_state')
            client_check_state.close()
        except Exception:
            pass
        
        if current_workflow_state == 'license_shown':
            # User confirmed license renewal, update state
            _update_service_workflow_state('license_confirmed')
            if _should_log():
                logger.info('User confirmed license renewal, updated workflow state')

    service_ready = False
    if active_service:
        service_ready = _service_requirements_met(active_service, session_doc)
        if _should_log():
            try:
                logger.info('Service readiness check: service=%s ready=%s intent_type=%s', active_service, service_ready, intent_type)
            except Exception:
                pass

    # Check if service just became ready and clear messages if so
    service_just_became_ready = False
    if active_service and service_ready:
        # Check if this is the first time service became ready by looking at message history
        try:
            client_check = _connect_mongo()
            db_check = client_check['chats']
            coll_check = db_check[user_id]
            session_current_id = new_session_generated if new_session_generated else session_id
            current_session = coll_check.find_one({'sessionId': session_current_id})
            
            # Check if messages have been cleared for this service already using a flag
            messages_already_cleared = False
            if current_session and current_session.get('context'):
                messages_already_cleared = current_session['context'].get(f'{active_service}_messages_cleared', False)
            
            # If messages haven't been cleared yet for this service, this is the first time service is ready
            if not messages_already_cleared:
                service_just_became_ready = True
                # Clear all messages when service becomes ready for the first time
                coll_check.update_one(
                    {'sessionId': session_current_id}, 
                    {'$set': {
                        'messages': [],
                        f'context.{active_service}_messages_cleared': True
                    }}
                )
                if _should_log():
                    logger.info('Cleared all messages as service %s is now ready for first time', active_service)
                    
        except Exception as e:
            if _should_log():
                logger.error('Failed to check/clear messages for service readiness: %s', str(e))
        finally:
            try:
                client_check.close()
            except Exception:
                pass

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
                # Security check: if this is an ID card / identification document, ensure
                # the extracted IC/userId matches the authenticated user_id. If mismatch,
                # do NOT persist the document context.
                try:
                    category_detection = ocr_result.get('category_detection', {}) or {}
                    detected_category = (category_detection.get('detected_category') or '').lower()
                    extracted_data = ocr_result.get('extracted_data', {}) or {}
                    extracted_ic = extracted_data.get('userId') or extracted_data.get('ic_number') or extracted_data.get('ic')
                    if detected_category in ('identification', 'idcard', 'id_card', 'identity', 'identity_card') and extracted_ic:
                        norm_uploaded = _normalize_ic(extracted_ic)
                        norm_user = _normalize_ic(user_id)
                        if norm_uploaded and norm_user and norm_uploaded != norm_user:
                            if _should_log():
                                logger.info('Identity mismatch detected: uploaded_ic=%s user_id=%s', norm_uploaded, norm_user)
                            # Craft a user-safe masked representation of uploaded IC to avoid leaking full value.
                            masked_uploaded = norm_uploaded
                            if len(masked_uploaded) >= 12:
                                masked_uploaded = masked_uploaded[:4] + '******' + masked_uploaded[-2:]
                            mismatch_message = (
                                "The identity number on the uploaded ID card (" + masked_uploaded + ") "
                                "does not match the account you are logged in with. For security reasons I cannot "
                                "process this document. Please upload the correct ID card that belongs to you, or "
                                "log in with the matching account." 
                            )
                            resp_body = {
                                'status': {'statusCode': 200, 'message': 'Success'},
                                'data': {
                                    'messageId': message_id,
                                    'message': mismatch_message,
                                    'createdAt': created_at_z,
                                    'sessionId': session_id if session_id != '(new-session)' else new_session_generated,
                                    'attachment': attachments,
                                    'intent_type': 'identity_mismatch'
                                }
                            }
                            return _cors_response(200, resp_body)
                except Exception as sec_e:
                    if _should_log():
                        logger.error('Failed during identity mismatch check: %s', str(sec_e))

                intent_type = 'document_processing'
                session_to_save = new_session_generated if new_session_generated else session_id
                _save_document_context_to_session(user_id, session_to_save, ocr_result, attachment['name'])
                
                if _should_log():
                    logger.info('Document processed successfully. Category: %s, Intent type: %s', 
                                ocr_result.get('category_detection', {}).get('detected_category', 'unknown'), intent_type)

    # Determine prompt for Bedrock.
    try:
        # If a service is active and requirements are met, bypass model with deterministic next-step prompt
        if active_service and service_ready and intent_type not in (
            'document_processing', 'document_correction_needed', 'document_correction_provided'
        ):
            # Get service message (may be direct message or AI prompt)
            service_message = _build_service_next_step_message(active_service, user_id, session_id, session_doc)
            
            # Only force intent_type for service next step if not already set or if service just became ready
            if not intent_type or intent_type == 'document_verified' or service_just_became_ready:
                intent_type = f'service_{active_service}_next'
            
            # Check if this is a direct message or AI prompt instruction
            if service_message.startswith('SYSTEM:'):
                # This is an AI prompt - use it as prompt for model
                prompt = service_message
                if _should_log():
                    logger.info('Using AI-generated service prompt. Intent type: %s, Verification status: %s, Service just ready: %s', 
                               intent_type or 'None', verification_status or 'None', service_just_became_ready)
            else:
                # This is a direct message - skip AI model
                response_text = service_message
                if _should_log():
                    logger.info('Using direct service message. Intent type: %s, Verification status: %s, Service just ready: %s', 
                               intent_type or 'None', verification_status or 'None', service_just_became_ready)
                # Skip AI model call for deterministic service messages
                model_error = None
            
        else:
            if _should_log():
                logger.info('Generating prompt. Intent type: %s, Verification status: %s', intent_type or 'None', verification_status or 'None')

            if session_id == '(new-session)':
                # For first-time connection, request a welcome message.
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
            elif intent_type == 'document_verified':
                # Post-verification: intentionally no special prompt; generic context builder will handle.
                # Keeping an explicit no-op to avoid accidental fall-through confusion.
                pass
            elif intent_type == 'document_correction_needed':
                # User said the information is wrong, ask for specifics
                extracted_data = unverified_doc_data.get('extractedData', {}) if unverified_doc_data else {}
                
                # Generate field examples based on actual OCR API fields
                field_mapping = {
                    'full_name': 'Full Name',
                    'userId': 'IC Number',
                    'gender': 'Gender', 
                    'address': 'Address',
                    'licenses_number': 'License Number',
                    'account_number': 'Account Number',
                    'invoice_number': 'Invoice Number'
                }
                
                field_examples = []
                for field_key, field_value in extracted_data.items():
                    friendly_name = field_mapping.get(field_key, field_key.replace('_', ' ').title())
                    field_examples.append(f"{friendly_name}: [correct {friendly_name.lower()}]")
                
                format_example = '\n'.join(field_examples) if field_examples else "Field Name: [correct value]"
                data_summary = '\n'.join([f'- {field_mapping.get(key, key.replace("_", " ").title())}: {value}' for key, value in extracted_data.items()])
                
                # Include full document context for AI understanding
                doc_context = json.dumps(unverified_doc_data, indent=2, default=str) if unverified_doc_data else "{}"
                
                prompt = (
                    "SYSTEM: The user said 'No' which means the extracted document information is INCORRECT. "
                    "You MUST ask them to specify which fields are wrong and provide corrections. "
                    "DO NOT proceed with the current data - the user explicitly said it's wrong. "
                    "\nFull document context (for AI reference):\n"
                    f"{doc_context}\n\n"
                    "Current extracted data (user said this is WRONG):\n"
                    f"{data_summary}\n\n"
                    "REQUIRED RESPONSE: Ask the user exactly this:\n"
                    "'I understand the information is incorrect. Which fields need to be corrected? "
                    "Please provide the correct details in this format:\n\n"
                    f"{format_example}\n\n"
                    "You can correct multiple fields at once.'\n\n"
                    f"User message: {message}\n\n"
                    "NOTE: userId represents the Identity Card (IC) number. If you include a signature, use 'MyGovHub Support Team' only."
                )
            elif intent_type == 'document_correction_provided':
                # User has provided corrections, show updated info and ask for confirmation
                # Get the updated document data after corrections
                try:
                    client_refresh = _connect_mongo()
                    db_refresh = client_refresh['chats']
                    coll_refresh = db_refresh[user_id]
                    session_to_get = new_session_generated if new_session_generated else session_id
                    updated_session = coll_refresh.find_one({'sessionId': session_to_get})
                    
                    if updated_session and unverified_doc_key in updated_session.get('context', {}):
                        updated_data = updated_session['context'][unverified_doc_key].get('extractedData', {})
                        
                        if _should_log():
                            logger.info('Retrieved updated data after corrections: %s', updated_data)
                        
                        # Use field mapping for user-friendly display
                        field_mapping = {
                            'full_name': 'Full Name',
                            'userId': 'IC Number',
                            'gender': 'Gender', 
                            'address': 'Address',
                            'licenses_number': 'License Number',
                            'account_number': 'Account Number',
                            'invoice_number': 'Invoice Number'
                        }
                        
                        # If there are pending corrections (correctedData), overlay them for display only
                        corrected_preview = updated_session['context'][unverified_doc_key].get('correctedData') or {}
                        preview_data = dict(updated_data)
                        preview_data.update(corrected_preview)  # overlay pending corrections
                        formatted_data = []
                        for key, value in preview_data.items():
                            friendly_name = field_mapping.get(key, key.replace('_', ' ').title())
                            formatted_data.append(f'- {friendly_name}: {value}')
                        
                        data_summary = '\n'.join(formatted_data)
                        
                        # Include full updated document context for AI reference
                        updated_doc_context = updated_session['context'][unverified_doc_key]
                        doc_context = json.dumps(updated_doc_context, indent=2, default=str)
                        
                        prompt = (
                            "SYSTEM: The user has provided corrections. Show ONLY the updated information with pending corrections overlaid (not yet finalized) and ask for confirmation. "
                            "DO NOT include technical details like timestamps, filenames, confidence scores, or verification status. "
                            "Show ONLY the user-visible data.\n\n"
                            "Updated information (including proposed corrections) to show user:\n"
                            f"{data_summary}\n\n"
                            "REQUIRED RESPONSE FORMAT:\n"
                            "Thank you for the corrections. Here is the updated information (pending your confirmation):\n\n"
                            f"{data_summary}\n\n"
                            "Is this corrected information now accurate? Please reply YES to confirm.\n\n"
                            "MyGovHub Support Team\n\n"
                            f"User message: {message}\n\n"
                            "NOTE: userId represents the Identity Card (IC) number. Keep response simple and user-friendly."
                        )
                    else:
                        prompt = f"SYSTEM: Error retrieving updated document data. User message: {message}"
                        
                    client_refresh.close()
                except Exception as e:
                    prompt = f"SYSTEM: Error processing corrections. User message: {message}"
                    if _should_log():
                        logger.error('Failed to retrieve updated document data: %s', str(e))
            elif intent_type == 'renew_license':
                prompt = (
                    "SYSTEM: Respond with ONLY the following guidance (no extra elaboration beyond minor natural phrasing allowed).\n\n"
                    "USER-FACING MESSAGE:\n"
                    "I can help you renew your driving license!\n\n"
                    "To proceed with the renewal, I need to verify your identity and current license details. Please upload one of the following documents:\n\n"
                    "📸 Option 1: Your current driving license (photo of the front side)\n"
                    "📸 Option 2: Your IC (Identity Card) - front side\n\n"
                    "Please take a clear photo and send it to me. I'll extract the necessary information to process your license renewal.\n"
                    "If you already uploaded a document earlier and it's verified, just reply YES to proceed with renewal steps."
                )
            elif intent_type == 'pay_tnb_bill':
                prompt = (
                    "SYSTEM: Respond ONLY with the following user guidance (no extra sentences).\n\n"
                    "USER-FACING MESSAGE:\n"
                    "I can help you pay your TNB electricity bill! ⚡\n\n"
                    "To process your bill payment, I need to verify your account details and bill information. Please upload:\n\n"
                    "📸 TNB Bill Document: Take a photo of your TNB bill (the upper portion showing your account number and amount due)\n\n"
                    "Please ensure the photo is clear and all important details are visible. I'll extract the account information to help you with the payment process."
                )
            else:
                # Generic context-building order: 1) Document context summary 2) EKYC 3) Prior messages 4) Current user message
                parts = []
                # 1. Document/context summary (only high-level; avoid dumping huge raw objects)
                if session_doc:
                    ctx = session_doc.get('context') or {}
                    if ctx:
                        # Summarize each document entry: ref + verification + key fields
                        if _should_log():
                            try:
                                logger.info('Prompt build: summarizing %d context entries', len(ctx))
                            except Exception:
                                pass
                        for key, doc_meta in list(ctx.items())[:5]:  # limit to first 5 to keep prompt small
                            if not key.startswith('document_'):
                                continue
                            ver_status = doc_meta.get('isVerified')
                            extracted = doc_meta.get('extractedData') or {}
                            # show only a few stable fields
                            field_snippets = []
                            for f in ['full_name', 'userId', 'licenses_number', 'account_number', 'invoice_number']:
                                if f in extracted:
                                    val = str(extracted.get(f))
                                    if len(val) > 40:
                                        val = val[:37] + '...'
                                    field_snippets.append(f"{f}:{val}")
                            snippet = ', '.join(field_snippets) if field_snippets else 'no key fields'
                            parts.append(f"DOC {key} status={ver_status} {snippet}\n")
                # 2. EKYC data
                if session_doc and session_doc.get('ekyc'):
                    try:
                        ekyc_str = json.dumps(session_doc.get('ekyc'))
                    except Exception:
                        ekyc_str = str(session_doc.get('ekyc'))
                    parts.append(f"EKYC: {ekyc_str}\n")
                    if _should_log():
                        try:
                            logger.info('Prompt build: included EKYC block size=%d chars', len(ekyc_str))
                        except Exception:
                            pass
                # 3. Prior messages
                if session_doc and isinstance(session_doc.get('messages'), list):
                    if _should_log():
                        try:
                            logger.info('Prompt build: iterating %d prior messages', len(session_doc.get('messages')))
                        except Exception:
                            pass
                    for m in session_doc.get('messages'):
                        role = m.get('role', 'user')
                        content_parts = []
                        for c in m.get('content', []):
                            text = c.get('text') if isinstance(c, dict) else str(c)
                            if text:
                                content_parts.append(str(text))
                        if content_parts:
                            parts.append(f"{role.upper()}: {' '.join(content_parts)}\n")
                # 4. Current user message
                parts.append(f"USER: {message}\n")
                prompt = "\n".join(parts)
                if _should_log():
                    try:
                        logger.info('Prompt build complete: length=%d chars', len(prompt))
                        logger.info('Prompt full:\n%s', json.dumps(prompt, indent=2))
                    except Exception:
                        pass

        # Only call AI model if we don't already have a direct service response
        if 'response_text' not in locals():
            model_error = None
            response_text = None
            try:
                # Log full prompt (sanitized & truncated) for debugging if enabled
                if _should_log():
                    try:
                        _prompt_log = prompt
                        # Basic masking for IC-like patterns (e.g., 6-2-4 digits or continuous 12 digits)
                        import re as _re_mask
                        _prompt_log = _re_mask.sub(r"******IC******", _re_mask.sub(r"(\d{6}-\d{2}-)\d{4}", r"\1****", _prompt_log))
                        max_log_len = 3000
                        truncated = len(_prompt_log) > max_log_len
                        if truncated:
                            _prompt_log_out = _prompt_log[:max_log_len] + '...<truncated>'
                        else:
                            _prompt_log_out = _prompt_log
                        logger.info('Prompt full%s length=%d chars:\n%s', ' (truncated)' if truncated else '', len(prompt), _prompt_log_out)
                    except Exception:
                        pass
                response_text = run_agent(prompt)
            except Exception as model_exc:
                # Record the model failure but continue — we'll persist an assistant error message
                model_error = str(model_exc)
                print('Model invocation failed:', model_error)
        else:
            # Direct service message - no AI model call needed
            model_error = None  # No model error since we didn't call the model
            if _should_log():
                logger.info('Using direct service response, skipping AI model call. Response length: %d chars', len(response_text or ''))

        # Persist the conversation: always push user message first, then assistant or error message
        session_to_update = new_session_generated if new_session_generated else session_id
        try:
            client2 = _connect_mongo()
            db2 = client2['chats']
            coll2 = db2[user_id]

            # push the user message (always)
            user_msg_doc = {
                'messageId': message_id,
                'timestamp': user_timestamp_iso,
                'role': 'user',
                'content': [{'text': str(message)}]
            }
            
            # Add attachment reference instead of full attachment with expiring URL
            if attachments:
                attachment = attachments[0]
                sanitized_filename = attachment['name'].replace('.', '_')
                user_msg_doc['attachment'] = {
                    "reference": f'document_{sanitized_filename}',
                    "type": attachment.get('type', 'unknown'),
                    "name": attachment['name']
                }
                
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
                    'timestamp': assistant_timestamp_iso,
                    'role': 'assistant',
                    'content': [{'text': str(response_text)}]
                }
            else:
                assistant_msg_doc = {
                    'messageId': assistant_message_id,
                    'timestamp': assistant_timestamp_iso,
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
