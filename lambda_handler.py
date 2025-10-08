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


def _service_requirements_met(service_name: str, session_doc: dict, ekyc_data: dict = None) -> bool:
    """Check if required verified fields exist for a given service.

    renew_license requires: a verified document containing full_name AND userId (IC number).
    pay_tnb_bill requires: either eKYC tnb_account_no OR a verified document containing account_number AND invoice_number.
    Returns True when requirements satisfied, False otherwise.
    """
    if not service_name or not session_doc:
        return False
    ctx = (session_doc.get('context') or {})

    # Special handling for pay_tnb_bill with eKYC
    if service_name == 'pay_tnb_bill' and ekyc_data:
        # Check if eKYC has TNB account numbers
        tnb_accounts = ekyc_data.get('tnb_account_no') or []
        if isinstance(tnb_accounts, list) and tnb_accounts:
            # eKYC has TNB accounts - service requirements are met
            return True
        # If no eKYC TNB accounts, fall through to document verification check

    # Must have at least one document_* entry in context overall (for non-eKYC path)
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
                workflow_state = current_session['context'].get(f'{service_name}_workflow_state')
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
            # User confirmed, now ask for renewal duration with clean, concise options
            try:
                client_workflow = _connect_mongo()
                chats_db = client_workflow['chats']
                user_coll = chats_db[user_id]
                user_coll.update_one(
                    {'sessionId': session_id}, 
                    {'$set': {f'context.{service_name}_workflow_state': 'asking_duration'}}
                )
                client_workflow.close()
            except Exception:
                pass

            renew_fee_per_year = 30.00
            
            # Return direct message with top 5 options (cleaner presentation)
            return (
                f"**License Renewal Duration 🔄**\n\n"
                f"Your current license expires on **{valid_to or 'N/A'}**. Please select how many years you'd like to renew for:\n\n"
                f"**Popular Options:**\n"
                f"• **1 year** - RM {renew_fee_per_year:.2f}\n"
                f"• **2 years** - RM {renew_fee_per_year * 2:.2f}\n"
                f"• **3 years** - RM {renew_fee_per_year * 3:.2f}\n"
                f"• **4 years** - RM {renew_fee_per_year * 4:.2f}\n"
                f"• **5 years** - RM {renew_fee_per_year * 5:.2f}\n\n"
                f"*Available: 1 to 10 years (RM 30.00 per year)*\n\n"
                f"Please reply with the **number of years** you want (e.g., \"3\" for 3 years). 😊"
            )
        elif workflow_state == 'confirming_license_payment_details':
            # User selected duration, now show payment confirmation
            try:
                client_payment = _connect_mongo()
                chats_db = client_payment['chats']
                user_coll = chats_db[user_id]
                current_session = user_coll.find_one({'sessionId': session_id})
                
                # Get stored duration and cost
                duration_years = 1
                renew_fee = 30.00
                if current_session and current_session.get('context'):
                    duration_years = current_session['context'].get(f'{service_name}_duration_years', 1)
                    renew_fee = current_session['context'].get(f'{service_name}_renew_fee', 30.00)
                
                # Calculate new expiry date
                try:
                    if valid_to:
                        current_expiry = datetime.strptime(valid_to, '%Y-%m-%d')
                        # Add years by replacing the year component
                        new_year = current_expiry.year + duration_years
                        new_expiry = current_expiry.replace(year=new_year)
                        new_expiry_str = new_expiry.strftime('%Y-%m-%d')
                    else:
                        new_expiry_str = 'N/A'
                except:
                    new_expiry_str = 'N/A'
                
                client_payment.close()
                
                return (
                    f"**Payment Confirmation 💳**\n\n"
                    f"**License Details:**\n"
                    f"• License No: {license_number or 'N/A'}\n"
                    f"• Current Expiry: {valid_to or 'N/A'}\n"
                    f"• Extension: {duration_years} year{'s' if duration_years > 1 else ''}\n"
                    f"• New Expiry: {new_expiry_str}\n\n"
                    f"**Total Amount: RM {renew_fee:.2f}**\n\n"
                    f"Please confirm to proceed with payment. Reply **YES** to continue or **NO** to cancel. 😊"
                )
            except Exception:
                return "Error retrieving payment details. Please try again."
        elif workflow_state == 'license_payment_confirmed':
            # Payment confirmed, update license record and show completion message
            try:
                client_completion = _connect_mongo()
                chats_db = client_completion['chats']
                user_coll = chats_db[user_id]
                current_session = user_coll.find_one({'sessionId': session_id})
                
                # Get stored renewal details
                duration_years = 1
                renew_fee = 30.00
                if current_session and current_session.get('context'):
                    duration_years = current_session['context'].get(f'{service_name}_duration_years', 1)
                    renew_fee = current_session['context'].get(f'{service_name}_renew_fee', 30.00)
                
                # Update the actual license record in MongoDB licenses collection
                try:
                    db_name = os.environ.get('ATLAS_DB_NAME') or ''
                    if not db_name:
                        logger.error("License verification complete, but database name not configured. Please set ATLAS_DB_NAME environment variable.")
                        return "License renewal completed, but I couldn't update your license record right now. Please contact support if you don't see the renewal reflected in your account."
                    
                    licenses_coll = client_completion[db_name]['licenses']
                    
                    # Get current license data from session context
                    license_data = current_session.get('context', {}).get('database_license', {})
                    current_valid_to = license_data.get('valid_to')
                    
                    if current_valid_to:
                        # Parse current expiry date and extend it
                        try:
                            # Parse the current valid_to date using datetime
                            if isinstance(current_valid_to, str):
                                # Try to parse common date formats
                                try:
                                    current_expiry = datetime.fromisoformat(current_valid_to.replace('Z', '+00:00'))
                                except:
                                    # Fallback for other formats
                                    current_expiry = datetime.strptime(current_valid_to[:10], '%Y-%m-%d')
                            else:
                                current_expiry = current_valid_to
                            
                            # Calculate new expiry date (extend by duration_years from current expiry)
                            # Add years by replacing the year component
                            new_year = current_expiry.year + duration_years
                            new_expiry = current_expiry.replace(year=new_year)
                            
                            # Update valid_from to today and valid_to to new expiry (use simple date format YYYY-MM-DD)
                            renewal_date = datetime.now(timezone.utc)
                            renewal_date_str = renewal_date.strftime('%Y-%m-%d')
                            new_expiry_str = new_expiry.strftime('%Y-%m-%d')
                            
                            # Update the license document
                            update_result = licenses_coll.update_one(
                                {'userId': user_id},
                                {'$set': {
                                    'valid_from': renewal_date_str,
                                    'valid_to': new_expiry_str,
                                    'status': 'active',
                                    'last_renewed': renewal_date,
                                    'renewal_duration_years': duration_years,
                                    'renewal_amount_paid': round(renew_fee, 2)
                                }}
                            )
                            
                            if update_result.modified_count > 0:
                                if _should_log():
                                    logger.info('Successfully updated license record for userId=%s: extended to %s', 
                                                user_id, new_expiry.strftime('%Y-%m-%d'))
                            else:
                                if _should_log():
                                    logger.warning('No license record updated for userId=%s', user_id)
                                    
                        except Exception as date_e:
                            if _should_log():
                                logger.error('Failed to parse/calculate license dates: %s', str(date_e))
                    else:
                        if _should_log():
                            logger.warning('No valid_to date found in license data for userId=%s', user_id)
                            
                except Exception as license_e:
                    if _should_log():
                        logger.error('Failed to update license record: %s', str(license_e))
                
                # Set intent to redirect to confirming_end_connection after completion
                try:
                    user_coll.update_one(
                        {'sessionId': session_id}, 
                        {'$set': {
                            'context.redirect_to_end_connection': True,
                            'context.end_connection_reason': 'license_renewal_completed'
                        }}
                    )
                except Exception as e:
                    if _should_log():
                        logger.error('Failed to set end connection redirect after license renewal: %s', str(e))
                
                client_completion.close()
                
                return (
                    f"**🎉 License Renewal Successful! 🎉**\n\n"
                    f"**Transaction Completed:**\n"
                    f"• License No: {license_number or 'N/A'}\n"
                    f"• Validity Period: From {renewal_date_str} to {new_expiry_str}\n"
                    f"• Extension: {duration_years} year{'s' if duration_years > 1 else ''}\n"
                    f"• Amount Paid: RM {renew_fee:.2f}\n\n"
                    f"**Important:**\n"
                    f"• Your license has been successfully renewed\n"
                    f"• You will receive a confirmation email shortly\n"
                    f"• Please keep this transaction reference for your records\n\n"
                    f"Thank you for using MyGovHub services! 😊\n\n"
                    f"Is there anything else I can help you with today? Reply **YES** if you need other services, or **NO** to end our session.\n\n"
                    f"MyGovHub Support Team"
                )
            except Exception:
                return "License renewal completed successfully! You will receive a confirmation email shortly."
        else:
            # First time or default - show license info and ask for confirmation
            # Set workflow state to track that we've shown license info
            try:
                client_workflow = _connect_mongo()
                chats_db = client_workflow['chats']
                user_coll = chats_db[user_id]
                user_coll.update_one(
                    {'sessionId': session_id}, 
                    {'$set': {f'context.{service_name}_workflow_state': 'license_shown'}}
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
        db_name = os.getenv('ATLAS_DB_NAME') or ''
        if not db_name:
            logger.error("Bill verification complete, but database name not configured. Please set ATLAS_DB_NAME environment variable.")
            return "Bill details verified, but I couldn't retrieve your bill records right now. Please try again shortly or provide more details."
        
        # Get account number - first check for eKYC selected account, then fall back to document
        account_number = None
        if session_doc and session_doc.get('context'):
            # Priority 1: Check for user-selected eKYC account
            account_number = session_doc['context'].get('selected_tnb_account')
            if _should_log():
                logger.info('TNB service message builder - session context: %s, selected_account: %s', 
                          session_doc.get('context', {}), account_number)
            
            # Priority 2: Fall back to verified document
            if not account_number:
                for key, doc_data in session_doc['context'].items():
                    if key.startswith('document_') and doc_data.get('isVerified') == 'verified':
                        extracted_data = doc_data.get('extractedData', {})
                        account_number = extracted_data.get('account_number')
                        if account_number:
                            break
        
        if not account_number:
            return "I couldn't find a verified account number. Please upload your TNB bill document first."
        
        # Check current workflow state from session
        workflow_state = None
        try:
            client_state = _connect_mongo()
            chats_db = client_state['chats']
            user_coll = chats_db[user_id]
            current_session = user_coll.find_one({'sessionId': session_id})
            if current_session and current_session.get('context'):
                workflow_state = current_session['context'].get(f'{service_name}_workflow_state')
            client_state.close()
        except Exception:
            pass
        
        # Fetch unpaid/overdue bills from MongoDB
        bills_to_pay = []
        try:
            client = _connect_mongo()
            try:
                bills_coll = client[db_name]['tnb-bills']
                # Find bills that need payment: unpaid or overdue (all bills must be paid in full)
                bills_cursor = bills_coll.find({
                    'bill.akaun.no_akaun': account_number,
                    'status': {'$in': ['unpaid', 'overdue']}
                }).sort('bill.meta.bil_semasa.tarikh_bil', -1)  # Latest bills first
                
                bills_to_pay = list(bills_cursor)
                
                if _should_log():
                    logger.info('Found %d bills to pay for account %s', len(bills_to_pay), account_number)
                
                # Store bills in session context for later use
                try:
                    chats_db = client['chats']
                    user_coll = chats_db[user_id]
                    # Remove _id from bills before storing
                    bills_for_context = [{k: v for k, v in bill.items() if k != '_id'} for bill in bills_to_pay]
                    user_coll.update_one(
                        {'sessionId': session_id}, 
                        {'$set': {'context.database_bills': bills_for_context}}
                    )
                    if _should_log():
                        logger.info('Stored %d bills in session context sessionId=%s', len(bills_for_context), session_id)
                except Exception:
                    if _should_log():
                        logger.exception('Failed to persist bills into session context')
            finally:
                try:
                    client.close()
                except Exception:
                    pass
        except Exception as e:
            if _should_log():
                logger.exception('Bills retrieval/update failure: %s', str(e))
            return "Bill details verified, but I couldn't retrieve your bill records right now. Please try again shortly or provide more details."
        
        if not bills_to_pay:
            # Set intent to redirect to confirming_end_connection
            try:
                client_redirect = _connect_mongo()
                chats_db = client_redirect['chats']
                user_coll = chats_db[user_id]
                
                # Set context flag to trigger confirming_end_connection intent
                user_coll.update_one(
                    {'sessionId': session_id}, 
                    {'$set': {
                        'context.redirect_to_end_connection': True,
                        'context.end_connection_reason': 'no_outstanding_bills'
                    }}
                )
                client_redirect.close()
            except Exception as e:
                if _should_log():
                    logger.error('Failed to set end connection redirect: %s', str(e))
                    
            return (
                f"Great news! I checked your TNB account ({account_number}) and found no outstanding bills. "
                "All your bills appear to be paid up to date. 🎉\n\n"
                "Is there anything else I can help you with today? Reply **YES** if you need other services, or **NO** to end our session."
            )
        
        # Handle different workflow states
        if workflow_state == 'tnb_bills_confirmed':
            # User confirmed payment, show completion message
            try:
                client_completion = _connect_mongo()
                chats_db = client_completion['chats']
                user_coll = chats_db[user_id]
                current_session = user_coll.find_one({'sessionId': session_id})
                
                # Get total amount paid
                total_paid = 0.0
                bill_count = 0
                if current_session and current_session.get('context'):
                    total_paid = current_session['context'].get(f'{service_name}_total_amount', 0.0)
                    bill_count = current_session['context'].get(f'{service_name}_bill_count', 0)
                
                client_completion.close()
                
                return (
                    f"**🎉 TNB Bill Payment Successful! 🎉**\n\n"
                    f"**Transaction Completed:**\n"
                    f"• Account No: {account_number}\n"
                    f"• Bills Paid: {bill_count}\n"
                    f"• Total Amount: RM {total_paid:.2f}\n\n"
                    f"**Important:**\n"
                    f"• All outstanding bills have been paid\n"
                    f"• You will receive a confirmation email shortly\n"
                    f"• Please keep this transaction reference for your records\n\n"
                    f"Thank you for using MyGovHub services! ⚡😊\n\n"
                    f"MyGovHub Support Team"
                )
            except Exception:
                return "TNB bill payment completed successfully! You will receive a confirmation email shortly."
        else:
            # First time or default - show bill info and ask for confirmation
            # Set workflow state to track that we've shown bills info
            try:
                client_workflow = _connect_mongo()
                chats_db = client_workflow['chats']
                user_coll = chats_db[user_id]
                user_coll.update_one(
                    {'sessionId': session_id}, 
                    {'$set': {f'context.{service_name}_workflow_state': 'tnb_bills_shown'}}
                )
                client_workflow.close()
            except Exception:
                pass
            
            # Calculate total amount and prepare bill summary
            total_amount = 0.0
            bill_summaries = []
            
            for bill in bills_to_pay:
                bill_data = bill.get('bill', {})
                akaun = bill_data.get('akaun', {})
                meta = bill_data.get('meta', {})
                bil_semasa = meta.get('bil_semasa', {})
                
                # All bills must be paid in full
                amount_due = bil_semasa.get('jumlah', 0.0)
                total_amount += amount_due
                
                # Format bill info
                invoice_no = akaun.get('no_invois', 'N/A')
                bill_date = bil_semasa.get('tarikh_bil', 'N/A')
                due_date = bil_semasa.get('bayar_sebelum', 'N/A')
                status_display = bill.get('status', 'unknown').upper()
                
                bill_summaries.append(
                    f"• Invoice #{invoice_no} - {status_display}\n"
                    f"  Bill Date: {bill_date} | Due: {due_date}\n"
                    f"  Amount: RM {amount_due:.2f}"
                )
            
            # Store payment details in session
            try:
                client_store = _connect_mongo()
                chats_db = client_store['chats']
                user_coll = chats_db[user_id]
                user_coll.update_one(
                    {'sessionId': session_id}, 
                    {'$set': {
                        f'context.{service_name}_total_amount': total_amount,
                        f'context.{service_name}_bill_count': len(bills_to_pay)
                    }}
                )
                client_store.close()
            except Exception:
                pass
            
            bills_text = "\n\n".join(bill_summaries)
            
            return (
                f"**TNB Bill Payment ⚡**\n\n"
                f"I found **{len(bills_to_pay)}** outstanding bill{'s' if len(bills_to_pay) > 1 else ''} for account **{account_number}**:\n\n"
                f"{bills_text}\n\n"
                f"**Total Amount Due: RM {total_amount:.2f}**\n\n"
                f"Would you like to pay {'all these bills' if len(bills_to_pay) > 1 else 'this bill'} now? "
                f"Reply **YES** to proceed with payment or **NO** to cancel."
            )

    return "Service data verified. @TODO: implement next workflow steps."

def _detect_service_intent(message_lower: str):
    """Detect high-level service intents from a free-form user message using Bedrock AI.

    Returns one of: 'renew_license', 'pay_tnb_bill' or None.
    Uses AI to intelligently detect user intent even with varied phrasing.
    """
    if not message_lower:
        return None

    # Keep original message for better context (don't just use lowercased version)
    original_message = message_lower

    try:
        # Create a focused prompt for service intent detection
        intent_prompt = (
            "SYSTEM: You are a service intent classifier for MyGovHub, a Malaysian government services portal. "
            "Analyze the user's message and determine if they want one of these specific services:\n\n"
            "AVAILABLE SERVICES:\n"
            "1. LICENSE_RENEWAL: User wants to renew their driving license\n"
            "   - Keywords: renew license, driving license renewal, lesen memandu, license extension, update license\n"
            "   - Variations: extend my license, my license expires, need to renew driving permit\n\n"
            "2. TNB_BILL_PAYMENT: User wants to pay TNB (electricity) bills\n"
            "   - Keywords: pay TNB bill, electricity bill, TNB payment, bil elektrik\n"
            "   - Variations: pay my electric bill, TNB account payment, utility bill payment\n\n"
            "3. NONE: Message does not clearly indicate either service above\n"
            "   - General inquiries, greetings, unclear requests, other services\n\n"
            "IMPORTANT RULES:\n"
            "- Only return one of these exact labels: LICENSE_RENEWAL, TNB_BILL_PAYMENT, or NONE\n"
            "- Be conservative - if unsure between two services, return NONE\n"
            "- Consider context clues and natural language variations\n"
            "- Handle both English and Bahasa Malaysia phrases\n"
            "- Do not return anything else - just the label\n\n"
            "EXAMPLES:\n"
            "- 'I need to renew my driving license' → LICENSE_RENEWAL\n"
            "- 'My license is expiring soon' → LICENSE_RENEWAL\n"
            "- 'Pay my TNB bill' → TNB_BILL_PAYMENT\n"
            "- 'Electricity bill payment' → TNB_BILL_PAYMENT\n"
            "- 'Hello, I need help' → NONE\n"
            "- 'What services do you offer?' → NONE\n\n"
            f"User message: \"{original_message}\"\n\n"
            "Classification:"
        )

        # Call Bedrock with a lower temperature for more consistent classification
        ai_response = run_agent(
            prompt=intent_prompt,
            max_tokens=50,
            temperature=0.1,  # Low temperature for consistent classification
            top_p=0.8
        ).strip().upper()

        if _should_log():
            logger.info('Service intent detection - Input: "%s", AI Response: "%s"', original_message, ai_response)

        # Map AI response to internal intent names
        if 'LICENSE_RENEWAL' in ai_response:
            return 'renew_license'
        elif 'TNB_BILL_PAYMENT' in ai_response:
            return 'pay_tnb_bill'
        elif 'NONE' in ai_response:
            return None
        else:
            # Fallback: AI returned unexpected response, log and return None
            if _should_log():
                logger.warning('Unexpected AI response for service intent detection: "%s"', ai_response)
            return None

    except Exception as e:
        # Fallback to simple keyword matching if Bedrock fails
        if _should_log():
            logger.error('Service intent detection with Bedrock failed, falling back to keywords: %s', str(e))
        
        # Original keyword-based logic as fallback
        if any(k in message_lower for k in ['renew', 'renewal', 'renewing']) and \
           any(k in message_lower for k in ['license', 'driving license', 'lesen', 'driver license']):
            return 'renew_license'

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
    
    # Debug logging for eKYC
    if _should_log():
        logger.info('Request eKYC data: %s', ekyc)

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

    # If new session or continue session, create sessionId and initialize collection/document in MongoDB (required)
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
        # Attempt to fetch existing session document so we can provide history to the model
        session_doc = None
        if session_id and session_id not in ('(new-session)', '(session-end)'):
            try:
                if _should_log():
                    logger.info('Fetching session from MongoDB: user=%s sessionId=%s', user_id, session_id)
                session_doc = coll.find_one({'sessionId': session_id})
                if session_doc:
                    status_val = session_doc.get('status')
                    messages_count = len(session_doc.get('messages') or [])
                    if _should_log():
                        logger.info('Fetched session from MongoDB: user=%s sessionId=%s status=%s messages=%d', user_id, session_id, status_val, messages_count)
                    
                    # Check session timeout (15 minutes) - skip if already awaiting timeout choice
                    if not session_doc.get('context', {}).get('timeout_awaiting_choice'):
                        session_timeout_minutes = 15 # Short timeout for testing; change to 15 for production @TODO
                        current_time = datetime.now(timezone.utc)
                        
                        # Get last message timestamp from session
                        last_message_time = None
                        messages = session_doc.get('messages', [])
                        if messages:
                            # Get the most recent message by parsing timestamp strings
                            def parse_timestamp_safe(ts_str):
                                """Safely parse timestamp string to datetime for comparison"""
                                if not ts_str or 'T' not in ts_str:
                                    return datetime.min.replace(tzinfo=timezone.utc)
                                try:
                                    # Parse MongoDB timestamp format (always uses +00:00, never Z)
                                    return datetime.fromisoformat(ts_str)
                                except Exception:
                                    return datetime.min.replace(tzinfo=timezone.utc)
                            
                            # Find message with most recent timestamp
                            last_msg = max(messages, key=lambda m: parse_timestamp_safe(m.get('timestamp', '')))
                            last_msg_timestamp = last_msg.get('timestamp', '')
                            
                            try:
                                if last_msg_timestamp and 'T' in last_msg_timestamp:
                                    # Parse the timestamp string from MongoDB (always +00:00 format)
                                    last_message_time = datetime.fromisoformat(last_msg_timestamp)
                                    # Ensure it's timezone-aware (convert to UTC if naive)
                                    if last_message_time.tzinfo is None:
                                        last_message_time = last_message_time.replace(tzinfo=timezone.utc)
                                    if _should_log():
                                        logger.info('Parsed last message timestamp: %s -> %s', last_msg_timestamp, last_message_time)
                            except Exception as e:
                                if _should_log():
                                    logger.error('Failed to parse message timestamp %s: %s', last_msg_timestamp, str(e))
                            
                            # Fallback to session createdAt if message parsing failed
                            if not last_message_time:
                                try:
                                    session_created = session_doc.get('createdAt', '')
                                    if session_created and 'T' in session_created:
                                        last_message_time = datetime.fromisoformat(session_created)
                                        # Ensure it's timezone-aware (convert to UTC if naive)
                                        if last_message_time.tzinfo is None:
                                            last_message_time = last_message_time.replace(tzinfo=timezone.utc)
                                        if _should_log():
                                            logger.info('Using session createdAt as fallback: %s -> %s', session_created, last_message_time)
                                except Exception as e:
                                    if _should_log():
                                        logger.error('Failed to parse session createdAt: %s', str(e))
                                    last_message_time = None
                        
                        # Check if session has timed out
                        try:
                            session_has_timed_out = (last_message_time and 
                                                   (current_time - last_message_time).total_seconds() > (session_timeout_minutes * 60))
                        except Exception as e:
                            if _should_log():
                                logger.error('Error calculating session timeout: %s, current_time=%s, last_message_time=%s', 
                                            str(e), current_time, last_message_time)
                            session_has_timed_out = False
                        
                        if session_has_timed_out:
                            # Session has timed out - ask user to choose
                            timeout_message = (
                                "🕐 **Session Timeout**\n\n"
                                f"Your session has been inactive for over {session_timeout_minutes} minutes.\n\n"
                                "⚠️ **Your message was not processed** due to this timeout.\n\n"
                                "Would you like to:\n\n"
                                "1. Continue your previous session (resume any ongoing services)\n"
                                "2. Start fresh with a new conversation\n\n"
                                "Please reply:\n"
                                "• **CONTINUE** - to resume your session\n"
                                "• **NEW** - to start a fresh conversation"
                            )
                            
                            # Set flag to indicate we're awaiting timeout choice
                            context_update = {
                                f'context.timeout_awaiting_choice': True
                            }
                            coll.update_one({'sessionId': session_id}, {'$set': context_update})
                            
                            resp_body = {
                                'status': {'statusCode': 200, 'message': 'Success'},
                                'data': {
                                    'messageId': message_id,
                                    'message': timeout_message,
                                    'createdAt': created_at_z,
                                    'sessionId': session_id,
                                    'attachment': attachments,
                                    'intent_type': 'session_timeout_choice'
                                }
                            }
                            return _cors_response(200, resp_body)
                    
                    # Log the full session document from MongoDB (always)
                    try:
                        if _should_log():
                            logger.info('Full session document from MongoDB: %s', json.dumps(session_doc, default=str))
                            # Also log timeout flag specifically for debugging
                            timeout_flag = session_doc.get('context', {}).get('timeout_awaiting_choice')
                            logger.info('Timeout awaiting choice flag: %s', timeout_flag)
                    except Exception:
                        logger.exception('Failed to log full session document from MongoDB')
                else:
                    if _should_log():
                        logger.info('No session document found for user=%s sessionId=%s', user_id, session_id)
            except Exception:
                logger.exception('Error fetching session document for user=%s sessionId=%s', user_id, session_id)
                session_doc = None
        if session_id in ('(new-session)', '(session-end)'):
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
                'context': {}
            }
            # Insert the document
            coll.insert_one(session_doc)

        else:
            update_ops = {}
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
    
    # Check for transcription failure from Layer 1 using Bedrock AI
    if message and message.strip():
        try:
            # Create a focused prompt for transcription failure detection
            transcription_failure_prompt = (
                "SYSTEM: You are analyzing messages from a speech-to-text transcription service. "
                "Determine if the message indicates a transcription failure or error.\n\n"
                "TRANSCRIPTION FAILURE INDICATORS:\n"
                "- Direct failure messages: 'Transcription failed', 'Speech recognition error', 'Audio processing failed'\n"
                "- Partial failures: 'Transcription completed but text retrieval failed', 'Audio unclear', 'Could not process audio'\n"
                "- Technical errors: 'Service unavailable', 'Timeout error', 'Processing error', 'Audio format not supported'\n"
                "- Quality issues: 'Audio too quiet', 'Background noise too high', 'Speech not detected'\n"
                "- Language variations: 'Transkripsi gagal', 'Error de transcripción', 'Échec de transcription'\n\n"
                "NORMAL MESSAGES (NOT failures):\n"
                "- Regular user text: 'Hello', 'I need help', 'Can you assist me'\n"
                "- Questions: 'What services do you offer?', 'How can I renew my license?'\n"
                "- Commands: 'Show me my bills', 'I want to pay'\n"
                "- Responses: 'Yes', 'No', 'Thank you'\n\n"
                "IMPORTANT RULES:\n"
                "- Only return 'TRANSCRIPTION_FAILED' if the message clearly indicates a transcription/speech processing error\n"
                "- Return 'NORMAL_MESSAGE' for regular user communication\n"
                "- Be conservative - if unsure, return 'NORMAL_MESSAGE'\n"
                "- Consider context clues and technical terminology\n"
                "- Handle multiple languages (English, Malay, etc.)\n"
                "- Do not return anything else - just the classification\n\n"
                "EXAMPLES:\n"
                "- 'Transcription failed.' → TRANSCRIPTION_FAILED\n"
                "- 'Transcription completed but text retrieval failed.' → TRANSCRIPTION_FAILED\n"
                "- 'Audio processing error' → TRANSCRIPTION_FAILED\n"
                "- 'Speech not detected' → TRANSCRIPTION_FAILED\n"
                "- 'Hello, I need help' → NORMAL_MESSAGE\n"
                "- 'Can you help me renew my license?' → NORMAL_MESSAGE\n\n"
                f"Message to analyze: \"{message.strip()}\"\n\n"
                "Classification:"
            )

            # Call Bedrock with low temperature for consistent classification
            ai_response = run_agent(
                prompt=transcription_failure_prompt,
                max_tokens=30,
                temperature=0.1,  # Very low temperature for consistent classification
                top_p=0.7
            ).strip().upper()

            if _should_log():
                logger.info('Transcription failure detection - Input: "%s", AI Response: "%s"', message.strip(), ai_response)

            # Check AI response
            if 'TRANSCRIPTION_FAILED' in ai_response:
                intent_type = 'transcription_failed'
                if _should_log():
                    logger.info('Detected transcription failure via Bedrock AI: "%s"', message.strip())
            elif 'NORMAL_MESSAGE' in ai_response:
                # Not a transcription failure, continue with normal processing
                if _should_log():
                    logger.info('Message classified as normal (not transcription failure): "%s"', message.strip())
            else:
                # Unexpected AI response, log and fallback to keyword detection
                if _should_log():
                    logger.warning('Unexpected AI response for transcription failure detection: "%s", falling back to keywords', ai_response)
                
                # Fallback to exact string matching for known failure messages
                failure_messages = [
                    'Transcription failed.',
                    'Transcription completed but text retrieval failed.',
                    'Speech recognition error',
                    'Audio processing failed',
                    'Could not process audio',
                    'Transcription service unavailable'
                ]
                
                if any(msg.lower() in message.strip().lower() for msg in failure_messages):
                    intent_type = 'transcription_failed'
                    if _should_log():
                        logger.info('Detected transcription failure via fallback keywords: "%s"', message.strip())

        except Exception as e:
            # Fallback to simple string matching if Bedrock fails
            if _should_log():
                logger.error('Transcription failure detection with Bedrock failed, falling back to exact matching: %s', str(e))
            
            # Original exact matching as ultimate fallback
            if message.strip() == 'Transcription failed.' or message.strip() == 'Transcription completed but text retrieval failed.':
                intent_type = 'transcription_failed'
                if _should_log():
                    logger.info('Detected transcription failure via exact string matching: "%s"', message.strip())
    
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
    
    if _should_log():
        logger.info('VERIFICATION DEBUG - message: "%s", message_lower: "%s", unverified_doc_key: %s', 
                   message, message_lower, unverified_doc_key)
    
    def _has_field_pattern(msg: str) -> bool:
        field_synonyms = ['name', 'full name', 'ic', 'ic number', 'gender', 'address', 'license', 'account', 'invoice']
        result = any(f" {syn} " in msg or msg.startswith(f"{syn} ") for syn in field_synonyms)
        if _should_log():
            logger.info('VERIFICATION DEBUG - _has_field_pattern("%s") = %s', msg, result)
        return result

    def _is_affirmative(msg: str) -> bool:
        # Accept short pure confirmations only; reject if appears to contain field corrections
        aff_tokens = {
            'yes', 'ya', 'y', 'ok', 'okay', 'true', 'benar', 'sure',
            'correct', 'accurate', 'looks good', 'betul', 'ya betul',
            'setuju', 'confirm'
        }
        cleaned = msg.strip().lower()
        
        # Remove common punctuation for better matching
        cleaned_no_punct = cleaned.rstrip('.,!?;:')
        
        if _should_log():
            logger.info('VERIFICATION DEBUG - _is_affirmative("%s") cleaned="%s", in_tokens=%s', 
                       msg, cleaned_no_punct, cleaned_no_punct in aff_tokens)
        
        if len(cleaned_no_punct) <= 15 and cleaned_no_punct in aff_tokens:
            return True
        # Multi-word accept if all tokens in affirmative set (after removing punctuation)
        tokens = cleaned_no_punct.replace('!', '').split()
        if all(t in aff_tokens for t in tokens):
            return True
        
        # For unclear cases, use AI as backup (only for longer messages that might be affirmative)
        if len(cleaned) > 5 and len(cleaned) < 50:
            try:
                # Create a focused prompt for affirmative detection
                affirmative_prompt = (
                    "SYSTEM: You are analyzing user messages to detect affirmative responses. "
                    "Determine if the message indicates agreement, confirmation, or acceptance.\n\n"
                    "AFFIRMATIVE INDICATORS:\n"
                    "- Direct confirmations: 'yes', 'ya', 'ok', 'okay', 'sure', 'correct'\n"
                    "- Agreement words: 'true', 'benar', 'betul', 'setuju', 'confirm'\n"
                    "- Positive phrases: 'looks good', 'that's right', 'sounds good'\n"
                    "- Language variations: 'ya betul', 'okay lah', 'yes please'\n"
                    "- With punctuation: 'yes.', 'ok!', 'sure?', 'correct.'\n\n"
                    "NON-AFFIRMATIVE (should return NEGATIVE):\n"
                    "- Field corrections: 'name is John', 'IC should be 123456', 'address is wrong'\n"
                    "- Questions: 'what about...', 'how do I...', 'can you...'\n"
                    "- Negative responses: 'no', 'not correct', 'wrong', 'incorrect'\n"
                    "- Unclear responses: 'maybe', 'I think', 'not sure'\n\n"
                    "IMPORTANT RULES:\n"
                    "- Return 'AFFIRMATIVE' only for clear agreement/confirmation responses\n"
                    "- Return 'NEGATIVE' for corrections, questions, or disagreements\n"
                    "- Be conservative - if unsure, return 'NEGATIVE'\n"
                    "- Ignore punctuation when determining intent\n"
                    "- Consider context clues and natural language patterns\n"
                    "- Do not return anything else - just 'AFFIRMATIVE' or 'NEGATIVE'\n\n"
                    "EXAMPLES:\n"
                    "- 'yes.' → AFFIRMATIVE\n"
                    "- 'ok!' → AFFIRMATIVE\n"
                    "- 'correct, proceed' → AFFIRMATIVE\n"
                    "- 'ya betul.' → AFFIRMATIVE\n"
                    "- 'looks good!' → AFFIRMATIVE\n"
                    "- 'name is John Smith' → NEGATIVE\n"
                    "- 'IC should be 123456' → NEGATIVE\n"
                    "- 'what about the address?' → NEGATIVE\n\n"
                    f"User message: \"{msg.strip()}\"\n\n"
                    "Classification:"
                )
    
                # Call Bedrock with low temperature for consistent classification
                ai_response = run_agent(
                    prompt=affirmative_prompt,
                    max_tokens=20,
                    temperature=0.1,  # Very low temperature for consistent classification
                    top_p=0.7
                ).strip().upper()
    
                if _should_log():
                    logger.info('Affirmative detection - Input: "%s", AI Response: "%s"', msg.strip(), ai_response)
    
                # Check AI response
                if 'AFFIRMATIVE' in ai_response:
                    if _should_log():
                        logger.info('AI detected affirmative intent: "%s"', msg.strip())
                    return True
                else:
                    if _should_log():
                        logger.info('AI classified as non-affirmative: "%s"', msg.strip())
                    return False
                    
            except Exception as e:
                if _should_log():
                    logger.error('Affirmative detection with Bedrock failed, falling back to keywords: %s', str(e))
                # Fallback to enhanced keyword matching
                return cleaned_no_punct in aff_tokens
                
        return False

    def _is_negative(msg: str) -> bool:
        # Accept negative responses - both English and Malay
        neg_tokens = {
            'no', 'nope', 'nah', 'not', 'cancel', 'cancelled', 'stop', 'quit', 'exit',
            'not interested', 'no thanks', 'no thank you', 'decline', 'reject',
            'tidak', 'tak', 'tak mahu', 'tak nak', 'batal', 'batalkan'
        }
        cleaned = msg.strip().lower()
        
        # Remove common punctuation for better matching
        cleaned_no_punct = cleaned.rstrip('.,!?;:')
        
        if len(cleaned_no_punct) <= 15 and cleaned_no_punct in neg_tokens:
            return True
        # Multi-word negative if all tokens in negative set (after removing punctuation)
        tokens = cleaned_no_punct.replace('!', '').split()
        if all(t in neg_tokens for t in tokens if len(t) > 1):  # Skip single letters
            return True
        
        # Check for phrases that start with negative words
        for neg_word in ['no', 'not', 'cancel', 'stop', 'tidak', 'tak', 'batal']:
            if cleaned_no_punct.startswith(f'{neg_word} ') or cleaned_no_punct == neg_word:
                return True
        
        # Multi-word negative phrases
        negative_phrases = ['not interested', 'no thanks', 'no thank you', 'tak mahu', 'tak nak']
        if any(phrase in cleaned for phrase in negative_phrases):
            return True
        
        # For unclear cases, use AI as backup (only for longer messages that might be negative)
        if len(cleaned) > 5 and len(cleaned) < 50:
            try:
                # Create a focused prompt for negative detection
                negative_prompt = (
                    "SYSTEM: You are analyzing user messages to detect negative responses. "
                    "Determine if the message indicates disagreement, refusal, or rejection.\n\n"
                    "NEGATIVE INDICATORS:\n"
                    "- Direct refusals: 'no', 'nope', 'not', 'cancel', 'stop', 'quit'\n"
                    "- Polite declines: 'no thanks', 'no thank you', 'not interested', 'decline'\n"
                    "- Malay negatives: 'tidak', 'tak', 'tak mahu', 'tak nak', 'batal'\n"
                    "- With punctuation: 'no.', 'not!', 'cancel?', 'tidak.'\n\n"
                    "NON-NEGATIVE (should return POSITIVE):\n"
                    "- Affirmative responses: 'yes', 'ok', 'sure', 'correct'\n"
                    "- Field corrections: 'name is John', 'IC should be 123456'\n"
                    "- Questions: 'what about...', 'how do I...', 'can you...'\n"
                    "- Neutral responses: 'maybe', 'I think', 'not sure about that'\n\n"
                    "IMPORTANT RULES:\n"
                    "- Return 'NEGATIVE' only for clear refusal/disagreement responses\n"
                    "- Return 'POSITIVE' for affirmations, questions, corrections, or neutral content\n"
                    "- Be conservative - if unsure, return 'POSITIVE'\n"
                    "- Ignore punctuation when determining intent\n"
                    "- Consider context clues and natural language patterns\n"
                    "- Do not return anything else - just 'NEGATIVE' or 'POSITIVE'\n\n"
                    "EXAMPLES:\n"
                    "- 'no.' → NEGATIVE\n"
                    "- 'not interested!' → NEGATIVE\n"
                    "- 'cancel this' → NEGATIVE\n"
                    "- 'tidak.' → NEGATIVE\n"
                    "- 'tak mahu' → NEGATIVE\n"
                    "- 'yes please' → POSITIVE\n"
                    "- 'name is John Smith' → POSITIVE\n"
                    "- 'what about payment?' → POSITIVE\n\n"
                    f"User message: \"{msg.strip()}\"\n\n"
                    "Classification:"
                )

                # Call Bedrock with low temperature for consistent classification
                ai_response = run_agent(
                    prompt=negative_prompt,
                    max_tokens=20,
                    temperature=0.1,  # Very low temperature for consistent classification
                    top_p=0.7
                ).strip().upper()

                if _should_log():
                    logger.info('Negative detection - Input: "%s", AI Response: "%s"', msg.strip(), ai_response)

                # Check AI response
                if 'NEGATIVE' in ai_response:
                    if _should_log():
                        logger.info('AI detected negative intent: "%s"', msg.strip())
                    return True
                else:
                    if _should_log():
                        logger.info('AI classified as non-negative: "%s"', msg.strip())
                    return False
                    
            except Exception as e:
                if _should_log():
                    logger.error('Negative detection with Bedrock failed, falling back to keywords: %s', str(e))
                # Fallback to enhanced keyword matching
                return cleaned_no_punct in neg_tokens
                
        return False

    def _detect_account_selection(msg: str, available_accounts: list) -> str:
        """
        Detect user's TNB account selection from numbered options or direct account numbers.
        Uses Bedrock AI to understand natural language selections.
        
        Args:
            msg: User's message
            available_accounts: List of available TNB account numbers
            
        Returns:
            str: Selected account number, or empty string if no clear selection detected
        """
        if not msg or not isinstance(available_accounts, list) or not available_accounts:
            return ""
            
        msg_clean = msg.strip()
        
        # First try simple pattern matching for numbers or direct account numbers
        try:
            # Check if message is just a number (1, 2, 3, etc.)
            if msg_clean.isdigit():
                choice_num = int(msg_clean)
                if 1 <= choice_num <= len(available_accounts):
                    selected_account = available_accounts[choice_num - 1]
                    if _should_log():
                        logger.info('Account selection by number: "%s" -> choice %d -> account %s', 
                                  msg_clean, choice_num, selected_account)
                    return selected_account
            
            # Check if message contains a direct account number
            for account in available_accounts:
                if account in msg_clean:
                    if _should_log():
                        logger.info('Account selection by direct match: "%s" -> account %s', msg_clean, account)
                    return account
                    
        except Exception as e:
            if _should_log():
                logger.error('Pattern matching for account selection failed: %s', str(e))
        
        # Use AI for more complex selections
        try:
            # Create numbered list for AI context
            account_list = ""
            for i, account in enumerate(available_accounts, 1):
                account_list += f"{i}. {account}\n"
            
            account_prompt = (
                "SYSTEM: You are analyzing user messages to detect TNB account selection. "
                "The user was shown a numbered list of TNB accounts and asked to select one.\n\n"
                "Available TNB accounts:\n"
                f"{account_list}\n"
                "DETECTION RULES:\n"
                "- User can select by number (e.g., '1', '2', 'option 1', 'choose 2')\n"
                "- User can select by account number (e.g., '200123456789', 'account 200123456789')\n"
                "- User can use natural language (e.g., 'first one', 'second account', 'the top one')\n"
                "- Be flexible with language variations and typos\n"
                "- Handle both English and Malay responses\n\n"
                "RESPONSE FORMAT:\n"
                "- If you can clearly identify an account selection, return ONLY the account number\n"
                "- If the message is unclear or doesn't indicate a selection, return 'UNCLEAR'\n"
                "- Do not return anything else - just the account number or 'UNCLEAR'\n\n"
                "EXAMPLES:\n"
                "- '1' → (first account number)\n"
                "- 'option 2' → (second account number)\n"
                "- 'first one' → (first account number)\n"
                "- 'choose 200123456789' → 200123456789\n"
                "- 'the second account' → (second account number)\n"
                "- 'pilih 1' → (first account number)\n"
                "- 'what is billing?' → UNCLEAR\n"
                "- 'not sure' → UNCLEAR\n\n"
                f"User message: \"{msg_clean}\"\n\n"
                "Selected account:"
            )

            # Call Bedrock with low temperature for consistent parsing
            ai_response = run_agent(
                prompt=account_prompt,
                max_tokens=50,
                temperature=0.1,  # Very low temperature for consistent parsing
                top_p=0.7
            ).strip()

            if _should_log():
                logger.info('Account selection AI - Input: "%s", AI Response: "%s"', msg_clean, ai_response)

            # Check if AI returned a valid account number
            if ai_response and ai_response != 'UNCLEAR':
                # Verify the AI response is one of our available accounts
                for account in available_accounts:
                    if account == ai_response or account in ai_response:
                        if _should_log():
                            logger.info('AI detected account selection: "%s" -> account %s', msg_clean, account)
                        return account
                        
                # If AI returned something but it's not a valid account, log warning
                if _should_log():
                    logger.warning('AI returned invalid account selection: "%s" not in available accounts', ai_response)

        except Exception as e:
            if _should_log():
                logger.error('Account selection detection with Bedrock failed: %s', str(e))
        
        # No clear selection detected
        if _should_log():
            logger.info('No clear account selection detected in message: "%s"', msg_clean)
        return ""

    def _is_document_rejection(msg: str) -> bool:
        # Accept document-specific rejection responses - includes accuracy/correctness terms
        rejection_tokens = {
            'no', 'incorrect', 'wrong', 'not correct', 'not accurate', 'inaccurate',
            'false', 'mistake', 'error', 'invalid', 'salah', 'tidak betul', 'tidak tepat'
        }
        cleaned = msg.strip().lower()
        
        # Direct match for rejection terms
        if cleaned in rejection_tokens:
            return True
        
        # Check for phrases that indicate incorrectness
        rejection_phrases = ['not correct', 'not accurate', 'not right', 'tidak betul', 'tidak tepat']
        if any(phrase in cleaned for phrase in rejection_phrases):
            return True
        
        # For unclear cases, use AI as backup
        if len(cleaned) > 5 and len(cleaned) < 50:
            ai_intent = _detect_intent_with_ai(msg)
            if ai_intent == 'document_rejection':
                if _should_log():
                    logger.info('AI detected document rejection intent: %s', msg)
                return True
            
        return False

    def _detect_intent_with_ai(msg: str) -> str:
        """Use AI to detect user intent from their message"""
        try:
            intent_prompt = (
                "SYSTEM: You are an intent classifier for a government services chatbot. "
                "Analyze the user's message and determine their intent. "
                "Respond with ONLY ONE of these intent labels (nothing else):\n\n"
                "- SESSION_TERMINATION: User wants to end/exit/quit the conversation completely\n"
                "- SERVICE_CONTINUE: User wants to continue with current service or process\n"
                "- DOCUMENT_REJECTION: User says document information is wrong/incorrect\n"
                "- AFFIRMATIVE: User agrees/confirms (yes, ok, correct, etc.)\n"
                "- NEGATIVE: User disagrees/declines (no, not interested, etc.)\n"
                "- GENERAL_INQUIRY: General questions or requests for help\n"
                "- UNCLEAR: Message is ambiguous or unclear\n\n"
                "SESSION_TERMINATION examples:\n"
                "- 'I want to quit', 'exit', 'I'm done', 'cancel this', 'log out'\n"
                "- 'This is taking too long, I'll come back later'\n"
                "- 'Forget it, I don't want to do this anymore'\n"
                "- 'I'm frustrated with this process'\n"
                "- 'Can we just end this conversation?'\n"
                "- 'I'm not interested in continuing'\n\n"
                f"User message: \"{msg}\"\n\n"
                "Respond with the intent label only:"
            )
            
            # Use existing run_agent function (which calls Bedrock)
            ai_intent = run_agent(intent_prompt).strip().upper()
            
            # Validate AI response and return standard intent
            if 'SESSION_TERMINATION' in ai_intent:
                return 'session_termination'
            elif 'AFFIRMATIVE' in ai_intent:
                return 'affirmative'
            elif 'NEGATIVE' in ai_intent:
                return 'negative'
            elif 'DOCUMENT_REJECTION' in ai_intent:
                return 'document_rejection'
            else:
                return 'unclear'
                
        except Exception as e:
            if _should_log():
                logger.error('AI intent detection failed: %s', str(e))
            return 'unclear'

    def _is_session_termination_request(msg: str) -> bool:
        # First try AI-powered detection for more intelligent recognition
        ai_intent = _detect_intent_with_ai(msg)
        if ai_intent == 'session_termination':
            if _should_log():
                logger.info('AI detected session termination intent: %s', msg)
            return True
        
        # Fallback to keyword-based detection for reliability
        termination_tokens = {
            'exit', 'quit', 'end', 'stop', 'cancel', 'bye', 'goodbye', 'close',
            'terminate', 'finish', 'done', 'logout', 'log out', 'sign out', 'reset',
            'restart', 'finish', 'complete',
            'keluar', 'berhenti', 'tamat', 'selesai', 'tutup', 'habis', 'ulang'
        }
        cleaned = msg.strip().lower()
        
        # Direct match for termination terms
        if cleaned in termination_tokens:
            if _should_log():
                logger.info('Keyword detected session termination: %s', msg)
            return True
        
        # Check for phrases that start with termination words
        for term_word in ['exit', 'quit', 'end', 'stop', 'cancel', 'close', 'reset', 'keluar', 'berhenti', 'tamat']:
            if cleaned.startswith(f'{term_word} ') or cleaned == term_word:
                if _should_log():
                    logger.info('Keyword phrase detected session termination: %s', msg)
                return True
        
        # Multi-word termination phrases
        termination_phrases = ['log out', 'sign out', 'end session', 'close session', 'reset session', 'restart session', 'i want to exit', 'i want to quit', 'i want to reset']
        if any(phrase in cleaned for phrase in termination_phrases):
            if _should_log():
                logger.info('Multi-word phrase detected session termination: %s', msg)
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
                {'$set': {f'context.{active_service}_workflow_state': new_state}}
            )
            if _should_log():
                logger.info('Updated service workflow state to: %s', new_state)
            client_workflow.close()
        except Exception as e:
            if _should_log():
                logger.error('Failed to update workflow state: %s', str(e))
    
    # Check for session termination request (highest priority - checked before all other logic)
    if _is_session_termination_request(message) and not attachments:
        # User wants to end the session completely
        try:
            client_terminate = _connect_mongo()
            chats_db = client_terminate['chats']
            user_coll = chats_db[user_id]
            session_current = new_session_generated if new_session_generated else session_id
            
            # Set session status to cancelled and clear any active service
            user_coll.update_one(
                {'sessionId': session_current}, 
                {'$set': {
                    'status': 'cancelled',
                    'service': ''  # Clear active service
                }}
            )
            
            # Set intent type to force connection end
            intent_type = 'force_end_connection'
            
            if _should_log():
                logger.info('User requested session termination, marked session as cancelled')
            
            client_terminate.close()
        except Exception as e:
            if _should_log():
                logger.error('Failed to terminate session: %s', str(e))
    
    # Handle transcription failure from Layer 1 (second highest priority after session termination)
    if intent_type == 'transcription_failed' and session_doc:
        # User's transcription failed, return previous assistant message with transcription error prefix
        try:
            client_transcription = _connect_mongo()
            chats_db = client_transcription['chats']
            user_coll = chats_db[user_id]
            
            # Get the last assistant message from the session
            current_session = user_coll.find_one({'sessionId': session_id})
            last_assistant_message = None
            
            if current_session and current_session.get('messages'):
                # Find the most recent assistant message
                messages = current_session.get('messages', [])
                for msg in reversed(messages):  # Search from newest to oldest
                    if msg.get('role') == 'assistant':
                        # Extract text from content array
                        content = msg.get('content', [])
                        if content and isinstance(content, list) and len(content) > 0:
                            text_content = content[0].get('text', '') if isinstance(content[0], dict) else str(content[0])
                            if text_content and not text_content.startswith('ERROR:') and not text_content.startswith('⚠️ **Transcription Failed**'):
                                last_assistant_message = text_content
                                break
            
            # Prepare the transcription failed message
            if last_assistant_message:
                transcription_failed_message = (
                    "⚠️ **Transcription Failed**\n\n"
                    "I'm sorry, but I couldn't understand your voice message clearly. Here's my previous response:\n\n"
                    f"{last_assistant_message}\n\n"
                    "Please try typing your message or speaking more clearly if you'd like to continue."
                )
            else:
                # No previous message found, provide a generic transcription error message
                transcription_failed_message = (
                    "⚠️ **Transcription Failed**\n\n"
                    "I'm sorry, but I couldn't understand your voice message clearly. "
                    "Please try typing your message or speaking more clearly. "
                    "How can I assist you with your government service needs?"
                )
            
            # Also append this message to session history
            try:
                new_msg = {
                    'role': 'assistant',
                    'content': [{'text': transcription_failed_message}],
                    'timestamp': created_at_iso
                }
                user_coll.update_one({'sessionId': session_id}, {'$push': {'messages': new_msg}})
            except Exception:
                # Non-fatal; if this fails we'll still return the message
                if _should_log():
                    logger.exception('Failed to append transcription failure message to session')
            
            if _should_log():
                logger.info('Handled transcription failure for session: %s, found previous message: %s', 
                            session_id, bool(last_assistant_message))
            
            # Return the transcription failed message
            resp_body = {
                'status': {'statusCode': 200, 'message': 'Success'},
                'data': {
                    'messageId': message_id,
                    'message': transcription_failed_message,
                    'createdAt': created_at_z,
                    'sessionId': session_id,
                    'attachment': attachments,
                    'intent_type': 'transcription_failed'
                }
            }
            
            client_transcription.close()
            return _cors_response(200, resp_body)
            
        except Exception as e:
            if _should_log():
                logger.error('Failed to handle transcription failure: %s', str(e))
            # If transcription failure handling fails, provide a basic error message
            basic_transcription_error = (
                "⚠️ **Transcription Failed**\n\n"
                "I'm sorry, but I couldn't understand your voice message clearly. "
                "Please try typing your message or speaking more clearly. "
                "How can I assist you with your government service needs?"
            )
            
            resp_body = {
                'status': {'statusCode': 200, 'message': 'Success'},
                'data': {
                    'messageId': message_id,
                    'message': basic_transcription_error,
                    'createdAt': created_at_z,
                    'sessionId': session_id,
                    'attachment': attachments,
                    'intent_type': 'transcription_failed_fallback'
                }
            }
            return _cors_response(200, resp_body)
    
    # Check for session timeout choice response
    timeout_awaiting_choice = session_doc and session_doc.get('context', {}).get('timeout_awaiting_choice')
    
    if _should_log():
        logger.info('Checking timeout choice: session_doc_exists=%s, timeout_flag=%s', 
                    bool(session_doc), timeout_awaiting_choice)
        if session_doc:
            context_debug = session_doc.get('context', {})
            logger.info('Session context keys: %s', list(context_debug.keys()))
            logger.info('Timeout flag value in context: %s', context_debug.get('timeout_awaiting_choice'))
    
    if session_doc and timeout_awaiting_choice:
        message_clean = message.strip().lower()

        # remove timeout_awaiting_choice flag regardless of user input
        try:
            client_clear_timeout = _connect_mongo()
            chats_db = client_clear_timeout['chats']
            user_coll = chats_db[user_id]
            user_coll.update_one(
                {'sessionId': session_id}, 
                {'$set': {
                    'context.timeout_awaiting_choice': False  # Clear the flag
                }}
            )
            client_clear_timeout.close()
        except Exception as e:
            if _should_log():
                logger.error('Failed to clear timeout flag: %s', str(e))
        
        # Enhanced keyword detection for 'new' - check if 'new' appears anywhere in the message
        contains_new_keyword = 'new' in message_clean
        contains_continue_keyword = any(word in message_clean for word in ['continue', 'resume', 'yes'])
        
        if _should_log():
            logger.info('Processing timeout choice: user_message="%s", timeout_awaiting_choice=%s', 
                        message_clean, timeout_awaiting_choice)
            logger.info('Enhanced keyword detection: contains_new=%s, contains_continue=%s', 
                        contains_new_keyword, contains_continue_keyword)
        
        if contains_continue_keyword and not contains_new_keyword:
            # User wants to continue old session - clear timeout flags and return previous message
            try:
                client_continue = _connect_mongo()
                chats_db = client_continue['chats']
                user_coll = chats_db[user_id]
                
                # Get the last assistant message from the session
                current_session = user_coll.find_one({'sessionId': session_id})
                last_assistant_message = None
                
                if current_session and current_session.get('messages'):
                    # Find the most recent assistant message
                    messages = current_session.get('messages', [])
                    for msg in reversed(messages):  # Search from newest to oldest
                        if msg.get('role') == 'assistant':
                            # Extract text from content array
                            content = msg.get('content', [])
                            if content and isinstance(content, list) and len(content) > 0:
                                text_content = content[0].get('text', '') if isinstance(content[0], dict) else str(content[0])
                                if text_content and not text_content.startswith('ERROR:'):
                                    last_assistant_message = text_content
                                    break
                
                # Clear timeout flag
                user_coll.update_one(
                    {'sessionId': session_id}, 
                    {'$unset': {
                        'context.timeout_awaiting_choice': ''
                    }}
                )

                # Also append a short assistant/system message with current timestamp so the
                # session's last-activity becomes 'now' and the timeout won't re-trigger immediately.
                try:
                    new_message_text = last_assistant_message if last_assistant_message else (
                        "Welcome back! Let's continue where we left off. How can I assist you?"
                    )
                    new_msg = {
                        'role': 'assistant',
                        'content': [{'text': new_message_text}],
                        'timestamp': created_at_iso
                    }
                    user_coll.update_one({'sessionId': session_id}, {'$push': {'messages': new_msg}})
                except Exception:
                    # Non-fatal; if this fails we'll still resume but future timeout logic may fire
                    if _should_log():
                        logger.exception('Failed to append resume marker message to session')
                
                if _should_log():
                    logger.info('User chose to continue timeout session: %s, cleared timeout flags, found last message: %s', 
                                session_id, bool(last_assistant_message))
                
                # Return the previous assistant message directly
                if last_assistant_message:
                    resp_body = {
                        'status': {'statusCode': 200, 'message': 'Success'},
                        'data': {
                            'messageId': message_id,
                            'message': last_assistant_message,
                            'createdAt': created_at_z,
                            'sessionId': session_id,
                            'attachment': attachments,
                            'intent_type': 'resume_previous_context'
                        }
                    }
                    client_continue.close()
                    return _cors_response(200, resp_body)
                else:
                    # No previous message found, provide a generic continue message
                    resume_message = (
                        "Welcome back! Let's continue where we left off. "
                        "How can I assist you with your government service needs?"
                    )
                    
                    resp_body = {
                        'status': {'statusCode': 200, 'message': 'Success'},
                        'data': {
                            'messageId': message_id,
                            'message': resume_message,
                            'createdAt': created_at_z,
                            'sessionId': session_id,
                            'attachment': attachments,
                            'intent_type': 'resume_session_generic'
                        }
                    }
                    client_continue.close()
                    return _cors_response(200, resp_body)
                
            except Exception as e:
                if _should_log():
                    logger.error('Failed to resume timeout session: %s', str(e))
                # If resume fails, continue with normal processing
                intent_type = 'resume_session_error'
                    
        elif message_clean in ['new', 'fresh', 'start', 'no', 'n', '2', 'restart', 'reset'] or contains_new_keyword:
            # User wants new session - generate new sessionId and return welcome
            if _should_log():
                logger.info('User chose NEW session (keyword_match=%s, contains_new=%s), processing new session creation', 
                            message_clean in ['new', 'fresh', 'start', 'no', 'n', '2', 'restart', 'reset'], contains_new_keyword)
            
            try:
                client_new = _connect_mongo()
                chats_db = client_new['chats']
                user_coll = chats_db[user_id]
                
                # Archive the old session and clear timeout flag
                archive_result = user_coll.update_one(
                    {'sessionId': session_id}, 
                    {'$set': {'status': 'archived'}, '$unset': {'context.timeout_awaiting_choice': ''}}
                )
                
                if _should_log():
                    logger.info('Archived old session %s, matched_count=%d', session_id, archive_result.matched_count)
                
                # Generate new session
                new_session_id = str(uuid.uuid4())
                
                # Create new session document
                new_session_doc = {
                    'sessionId': new_session_id,
                    'createdAt': created_at_iso,
                    'messages': [],
                    'status': 'active',
                    'service': '',
                    'context': {}
                }
                insert_result = user_coll.insert_one(new_session_doc)
                
                if _should_log():
                    logger.info('Created new session %s, insert_id=%s', new_session_id, str(insert_result.inserted_id))
                
                # Return restart confirmation message
                restart_message = (
                    "Perfect! I've started a fresh conversation for you. �\n\n"
                    "Hi there! I'm your MyGovHub assistant. I can help you with:\n\n"
                    "• 🆔 **License Renewal** - Renew your driving license\n"
                    "• 💡 **TNB Bill Payment** - Pay your electricity bills\n"
                    "• 📄 **Permitting** - Apply for permits\n"
                    "• 📋 **Application Status** - Check the status of your applications\n\n"
                    "Just tell me what you need help with, or upload any relevant documents to get started!"
                )
                
                resp_body = {
                    'status': {'statusCode': 200, 'message': 'Success'},
                    'data': {
                        'messageId': message_id,
                        'message': restart_message,
                        'createdAt': created_at_z,
                        'sessionId': '(session-end)',  # Force client to start completely fresh
                        'attachment': attachments,
                        'intent_type': 'session_restart_confirmed'
                    }
                }
                
                client_new.close()
                
                return _cors_response(200, resp_body)
                
            except Exception as e:
                if _should_log():
                    logger.error('Failed to create new session after timeout: %s', str(e))
                    logger.exception('Full exception details for new session creation')
                # If creating new session fails, return an error rather than continuing with old session
                error_resp_body = {
                    'status': {'statusCode': 500, 'message': 'Failed to create new session'},
                    'error': str(e)
                }
                return _cors_response(500, error_resp_body)
        
        else:
            # Invalid choice - ask again (keep timeout_awaiting_choice flag set)
            if _should_log():
                logger.info('Invalid timeout choice: message="%s", contains_new=%s, contains_continue=%s', 
                           message_clean, contains_new_keyword, contains_continue_keyword)
            
            clarification_message = (
                "⚠️ **Please Choose an Option**\n\n"
                "I didn't understand your choice. Please reply:\n\n"
                "• **CONTINUE** to resume your previous session\n"
                "• **NEW** to start a fresh conversation"
            )
            
            resp_body = {
                'status': {'statusCode': 200, 'message': 'Success'},
                'data': {
                    'messageId': message_id,
                    'message': clarification_message,
                    'createdAt': created_at_z,
                    'sessionId': session_id,
                    'attachment': attachments,
                    'intent_type': 'session_timeout_clarification'
                }
            }
            return _cors_response(200, resp_body)
    
    if session_doc and not timeout_awaiting_choice:
        # Ensure timeout_awaiting_choice flag is cleared if it exists but not needed
        if 'timeout_awaiting_choice' in session_doc.get('context', {}):
            try:
                client_clear_flag = _connect_mongo()
                chats_db = client_clear_flag['chats']
                user_coll = chats_db[user_id]
                user_coll.update_one(
                    {'sessionId': session_id}, 
                    {'$unset': {
                        'context.timeout_awaiting_choice': ''
                    }}
                )
                if _should_log():
                    logger.info('Cleared stale timeout_awaiting_choice flag for session: %s', session_id)
                client_clear_flag.close()
            except Exception as e:
                if _should_log():
                    logger.error('Failed to clear stale timeout flag: %s', str(e))
    
    # Order: explicit rejection -> corrections -> affirmation
    # Rejection (needs corrections)
    if unverified_doc_key and _is_document_rejection(message):
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
            if _should_log():
                logger.info('VERIFICATION DEBUG - Document verified! message_lower="%s", intent_type="%s"', 
                           message_lower, intent_type)
    # Legacy path (affirmation first) kept for cases without document
    elif _is_affirmative(message) and not _is_document_rejection(message):
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
            
            # Auto-detect service based on document category after verification
            # Get current active service from the session
            session_for_service_check = coll_verify.find_one({'sessionId': session_to_verify})
            current_active_service = session_for_service_check.get('service') if session_for_service_check else ''
            
            if _should_log():
                logger.info('Auto-detection check: current_active_service=%s, unverified_doc_key=%s', current_active_service, unverified_doc_key)
            
            if not current_active_service:
                if _should_log():
                    logger.info('No active service, checking document category for auto-detection')
                # Get the verified document to check its category
                updated_doc = coll_verify.find_one({'sessionId': session_to_verify})
                if _should_log():
                    logger.info('Auto-detection: updated_doc exists=%s, unverified_doc_key=%s', bool(updated_doc), unverified_doc_key)
                    if updated_doc and updated_doc.get('context'):
                        logger.info('Available context keys: %s', list(updated_doc.get('context', {}).keys()))
                
                if updated_doc and updated_doc.get('context', {}).get(unverified_doc_key):
                    verified_doc_data = updated_doc['context'][unverified_doc_key]
                    category_detection = verified_doc_data.get('categoryDetection', {})
                    detected_category = category_detection.get('detected_category', '').lower()
                    
                    if _should_log():
                        logger.info('Document verification - unverified_doc_key: %s, detected_category: %s', 
                                  unverified_doc_key, detected_category)
                    
                    if detected_category == 'tnb':
                        # Set TNB bill payment service after verification
                        service_update_result = coll_verify.update_one(
                            {'sessionId': session_to_verify}, 
                            {'$set': {'service': 'pay_tnb_bill'}}
                        )
                        
                        # Update local variable
                        current_active_service = 'pay_tnb_bill'
                        
                        if _should_log():
                            logger.info('Auto-set service to pay_tnb_bill after TNB document verification. Updated: %d documents', 
                                      service_update_result.modified_count)
                        
                        # Refresh session_doc to include the updated service
                        try:
                            refreshed_session = coll_verify.find_one({'sessionId': session_to_verify})
                            if refreshed_session:
                                session_doc = refreshed_session
                                if _should_log():
                                    logger.info('Session document refreshed after service auto-detection')
                        except Exception as refresh_error:
                            if _should_log():
                                logger.error('Failed to refresh session document: %s', str(refresh_error))
                    elif detected_category in ['license', 'license-front', 'license-back']:
                        # Set license renewal service after verification
                        service_update_result = coll_verify.update_one(
                            {'sessionId': session_to_verify}, 
                            {'$set': {'service': 'renew_license'}}
                        )
                        
                        # Update local variable
                        current_active_service = 'renew_license'
                        
                        if _should_log():
                            logger.info('Auto-set service to renew_license after license document verification. Category: %s, Updated: %d documents', 
                                      detected_category, service_update_result.modified_count)
                        
                        # Refresh session_doc to include the updated service
                        try:
                            refreshed_session = coll_verify.find_one({'sessionId': session_to_verify})
                            if refreshed_session:
                                session_doc = refreshed_session
                                if _should_log():
                                    logger.info('Session document refreshed after license service auto-detection')
                        except Exception as refresh_error:
                            if _should_log():
                                logger.error('Failed to refresh session document after license auto-detection: %s', str(refresh_error))
                    elif detected_category == 'idcard':
                        # For ID card, don't auto-set service, but log for special handling
                        if _should_log():
                            logger.info('ID card document verified. Category: %s - Will prompt user for service selection', detected_category)
                    else:
                        if _should_log():
                            logger.info('Document category "%s" does not match TNB, no auto-service set', detected_category)
                else:
                    if _should_log():
                        logger.info('Document not found or context missing for auto-detection with unverified_doc_key')
                    
                    # Alternative: check all documents in context for TNB category (verified or unverified)
                    if updated_doc and updated_doc.get('context'):
                        for doc_key, doc_data in updated_doc['context'].items():
                            if isinstance(doc_data, dict) and doc_data.get('categoryDetection'):
                                category_detection = doc_data.get('categoryDetection', {})
                                detected_category = category_detection.get('detected_category', '').lower()
                                is_verified = doc_data.get('isVerified') == 'verified'
                                
                                if _should_log():
                                    logger.info('Checking doc %s: category=%s, is_verified=%s', doc_key, detected_category, is_verified)
                                
                                if detected_category == 'tnb' and is_verified:
                                    # Set TNB bill payment service after verification
                                    service_update_result = coll_verify.update_one(
                                        {'sessionId': session_to_verify}, 
                                        {'$set': {'service': 'pay_tnb_bill'}}
                                    )
                                    
                                    # Update local variable
                                    current_active_service = 'pay_tnb_bill'
                                    
                                    if _should_log():
                                        logger.info('ALTERNATIVE: Auto-set service to pay_tnb_bill after TNB document verification. Doc: %s, Updated: %d documents', 
                                                  doc_key, service_update_result.modified_count)
                                    
                                    # Refresh session_doc to include the updated service
                                    try:
                                        refreshed_session = coll_verify.find_one({'sessionId': session_to_verify})
                                        if refreshed_session:
                                            session_doc = refreshed_session
                                            if _should_log():
                                                logger.info('Session document refreshed after alternative service auto-detection')
                                    except Exception as refresh_error:
                                        if _should_log():
                                            logger.error('Failed to refresh session document in alternative: %s', str(refresh_error))
                                    break
                                elif detected_category in ['license', 'license-front', 'license-back'] and is_verified:
                                    # Set license renewal service after verification
                                    service_update_result = coll_verify.update_one(
                                        {'sessionId': session_to_verify}, 
                                        {'$set': {'service': 'renew_license'}}
                                    )
                                    
                                    # Update local variable
                                    current_active_service = 'renew_license'
                                    
                                    if _should_log():
                                        logger.info('ALTERNATIVE: Auto-set service to renew_license after license document verification. Doc: %s, Category: %s, Updated: %d documents', 
                                                  doc_key, detected_category, service_update_result.modified_count)
                                    
                                    # Refresh session_doc to include the updated service
                                    try:
                                        refreshed_session = coll_verify.find_one({'sessionId': session_to_verify})
                                        if refreshed_session:
                                            session_doc = refreshed_session
                                            if _should_log():
                                                logger.info('Session document refreshed after license service auto-detection')
                                    except Exception as refresh_error:
                                        if _should_log():
                                            logger.error('Failed to refresh session document after license auto-detection: %s', str(refresh_error))
                                    break
                                elif detected_category == 'idcard' and is_verified:
                                    # For ID card, don't auto-set service, but log for special handling
                                    if _should_log():
                                        logger.info('ID card document verified. Doc: %s, Category: %s - Will prompt user for service selection', 
                                                  doc_key, detected_category)
                                    # Don't break here, continue checking other documents
            else:
                if _should_log():
                    logger.info('Active service already exists: %s, skipping auto-detection', current_active_service)
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
                current_workflow_state = current_session['context'].get(f'{active_service}_workflow_state')
            client_check_state.close()
        except Exception:
            pass
        
        if current_workflow_state == 'license_shown':
            # User confirmed license renewal, update state
            _update_service_workflow_state('license_confirmed')
            if _should_log():
                logger.info('User confirmed license renewal, updated workflow state')
        elif current_workflow_state == 'confirming_license_payment_details':
            # User confirmed payment, process the renewal
            _update_service_workflow_state('license_payment_confirmed')
            intent_type = f'{active_service}_payment_confirmed'
            if _should_log():
                logger.info('User confirmed license renewal payment, updated workflow state')
    
    # Check for service-specific cancellation (when service is active and user says no)
    elif active_service == 'renew_license' and _is_negative(message_lower) and not unverified_doc_key:
        # Check current workflow state
        current_workflow_state = None
        try:
            client_check_state = _connect_mongo()
            chats_db = client_check_state['chats']
            user_coll = chats_db[user_id]
            session_current = new_session_generated if new_session_generated else session_id
            current_session = user_coll.find_one({'sessionId': session_current})
            if current_session and current_session.get('context'):
                current_workflow_state = current_session['context'].get(f'{active_service}_workflow_state')
            client_check_state.close()
        except Exception:
            pass
        
        if current_workflow_state == 'license_shown':
            # User declined license renewal, cancel the service
            try:
                client_cancel = _connect_mongo()
                chats_db = client_cancel['chats']
                user_coll = chats_db[user_id]
                session_current = new_session_generated if new_session_generated else session_id
                
                # Set workflow state to cancelled and session status to cancelled
                user_coll.update_one(
                    {'sessionId': session_current}, 
                    {'$set': {
                        f'context.{active_service}_workflow_state': 'action_cancelled',
                        'status': 'cancelled'
                    }}
                )
                
                # Set intent type to force connection end
                intent_type = 'force_end_connection'
                
                if _should_log():
                    logger.info('User declined license renewal, marked session as cancelled')
                
                client_cancel.close()
            except Exception as e:
                if _should_log():
                    logger.error('Failed to cancel service workflow: %s', str(e))
        elif current_workflow_state == 'confirming_license_payment_details':
            # User declined payment, cancel the service
            try:
                client_cancel = _connect_mongo()
                chats_db = client_cancel['chats']
                user_coll = chats_db[user_id]
                session_current = new_session_generated if new_session_generated else session_id
                
                # Set workflow state to cancelled and session status to cancelled
                user_coll.update_one(
                    {'sessionId': session_current}, 
                    {'$set': {
                        f'context.{active_service}_workflow_state': 'action_cancelled',
                        'status': 'cancelled'
                    }}
                )
                
                # Set intent type to force connection end
                intent_type = 'force_end_connection'
                
                if _should_log():
                    logger.info('User declined license renewal payment, marked session as cancelled')
                
                client_cancel.close()
            except Exception as e:
                if _should_log():
                    logger.error('Failed to cancel service workflow: %s', str(e))

    # Check for duration selection (when user provides number of years)
    elif active_service == 'renew_license' and not unverified_doc_key and not intent_type:
        # Check if we're in asking_duration state and user provided a number
        current_workflow_state = None
        try:
            client_check_duration = _connect_mongo()
            chats_db = client_check_duration['chats']
            user_coll = chats_db[user_id]
            session_current = new_session_generated if new_session_generated else session_id
            current_session = user_coll.find_one({'sessionId': session_current})
            if current_session and current_session.get('context'):
                current_workflow_state = current_session['context'].get(f'{active_service}_workflow_state')
            client_check_duration.close()
        except Exception:
            pass
        
        if current_workflow_state == 'asking_duration':
            # Use Bedrock AI to intelligently parse duration from user message
            years = None
            
            try:
                # Create a focused prompt for duration extraction
                duration_prompt = (
                    "SYSTEM: You are parsing license renewal duration from user messages. "
                    "Extract the number of years the user wants to renew their license for.\n\n"
                    "VALID INPUTS:\n"
                    "- Numbers: '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'\n"
                    "- Written numbers (English): 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'\n"
                    "- Written numbers (Malay): 'satu', 'dua', 'tiga', 'empat', 'lima', 'enam', 'tujuh', 'lapan', 'sembilan', 'sepuluh'\n"
                    "- With units: '3 years', '5 tahun', 'two years', 'lima tahun'\n"
                    "- Natural language: 'I want 3 years', 'Renew for 5 years', 'Make it 2 years please'\n"
                    "- Mixed: '3 years please', 'satu tahun saja', 'just 2', 'only five'\n\n"
                    "INVALID INPUTS:\n"
                    "- Out of range: '0', '11', '15', '20', 'zero', 'eleven'\n"
                    "- Non-duration: 'yes', 'no', 'help', 'I don't know', 'maybe'\n"
                    "- Unclear: 'a few', 'some', 'many', 'not sure'\n\n"
                    "INSTRUCTIONS:\n"
                    "- Only return a single number from 1 to 10 if you can clearly identify the duration\n"
                    "- Return 'INVALID' if the input is unclear, out of range, or not a duration\n"
                    "- Return 'INVALID' if you're unsure about the user's intent\n"
                    "- Be conservative - when in doubt, return 'INVALID'\n"
                    "- Do not return anything else - just the number or 'INVALID'\n\n"
                    "EXAMPLES:\n"
                    "- '3' → 3\n"
                    "- 'five years' → 5\n"
                    "- 'tiga tahun' → 3\n"
                    "- 'I want to renew for 2 years' → 2\n"
                    "- '7 years please' → 7\n"
                    "- 'sepuluh' → 10\n"
                    "- 'yes' → INVALID\n"
                    "- '15 years' → INVALID\n"
                    "- 'I don't know' → INVALID\n"
                    "- 'a few years' → INVALID\n\n"
                    f"User message: \"{message.strip()}\"\n\n"
                    "Duration (1-10 or INVALID):"
                )

                # Call Bedrock with low temperature for consistent parsing
                ai_response = run_agent(
                    prompt=duration_prompt,
                    max_tokens=20,
                    temperature=0.1,  # Very low temperature for consistent parsing
                    top_p=0.7
                ).strip()

                if _should_log():
                    logger.info('Duration parsing - Input: "%s", AI Response: "%s"', message.strip(), ai_response)

                # Parse AI response
                if ai_response.upper() == 'INVALID':
                    years = None
                    if _should_log():
                        logger.info('AI classified duration as invalid: "%s"', message.strip())
                else:
                    try:
                        # Try to extract number from AI response
                        years_candidate = int(ai_response)
                        if 1 <= years_candidate <= 10:
                            years = years_candidate
                            if _should_log():
                                logger.info('AI successfully parsed duration: %d years from "%s"', years, message.strip())
                        else:
                            years = None
                            if _should_log():
                                logger.warning('AI returned out-of-range duration: %d from "%s"', years_candidate, message.strip())
                    except (ValueError, TypeError):
                        years = None
                        if _should_log():
                            logger.warning('AI returned non-numeric duration: "%s" from "%s"', ai_response, message.strip())

            except Exception as e:
                # Fallback to simple regex parsing if Bedrock fails
                if _should_log():
                    logger.error('Duration parsing with Bedrock failed, falling back to regex: %s', str(e))
                
                import re
                # Simple fallback - extract first number from message
                duration_match = re.search(r'\b(\d{1,2})\b', message.strip())
                if duration_match:
                    try:
                        years_candidate = int(duration_match.group(1))
                        if 1 <= years_candidate <= 10:
                            years = years_candidate
                            if _should_log():
                                logger.info('Fallback regex parsed duration: %d years', years)
                    except ValueError:
                        pass
            
            # Process the parsed duration
            if years is not None:
                if 1 <= years <= 10:  # Valid range (double-check)
                    renew_fee_per_year = 30.00
                    renew_fee = years * renew_fee_per_year
                    
                    # Store the selected duration and cost
                    try:
                        client_store_duration = _connect_mongo()
                        chats_db = client_store_duration['chats']
                        user_coll = chats_db[user_id]
                        session_current = new_session_generated if new_session_generated else session_id
                        
                        user_coll.update_one(
                            {'sessionId': session_current}, 
                            {'$set': {
                                f'context.{active_service}_workflow_state': 'confirming_license_payment_details',
                                f'context.{active_service}_duration_years': years,
                                f'context.{active_service}_renew_fee': round(renew_fee, 2)
                            }}
                        )
                        
                        if _should_log():
                            logger.info('User selected %d years renewal, cost: RM %.2f', years, renew_fee)
                        
                        # Set intent to trigger payment confirmation message
                        intent_type = 'license_duration_selected'
                        
                        client_store_duration.close()
                    except Exception as e:
                        if _should_log():
                            logger.error('Failed to store duration selection: %s', str(e))
                else:
                    # This shouldn't happen with AI parsing, but safety check
                    if _should_log():
                        logger.warning('Invalid duration range after parsing: %d (must be 1-10)', years)
                    intent_type = 'invalid_duration_format'
            else:
                # AI couldn't parse a valid duration from the message
                if _should_log():
                    logger.info('No valid duration found in message: "%s"', message.strip())
                # Set intent to ask for valid numeric input
                intent_type = 'invalid_duration_format'

    # Check for TNB account selection (when service is active and eKYC accounts are available)
    elif active_service == 'pay_tnb_bill' and not unverified_doc_key and ekyc:
        # Get eKYC accounts from current request
        tnb_accounts = ekyc.get('tnb_account_no', [])
        
        if isinstance(tnb_accounts, list) and tnb_accounts:
            # Check if user is selecting an account
            selected_account = _detect_account_selection(message, tnb_accounts)
            if selected_account:
                # User selected an account - store ONLY the selected account number
                try:
                    client_account = _connect_mongo()
                    chats_db = client_account['chats']
                    user_coll = chats_db[user_id]
                    session_to_update = new_session_generated if new_session_generated else session_id
                    
                    # Store only the selected account number (not full eKYC data)
                    update_result = user_coll.update_one(
                        {'sessionId': session_to_update}, 
                        {'$set': {
                            'context.selected_tnb_account': selected_account,
                            f'context.{active_service}_workflow_state': 'account_selected'
                        }}
                    )
                    
                    if _should_log():
                        logger.info('Account selection storage: sessionId=%s, account=%s, matched=%d, modified=%d', 
                                  session_to_update, selected_account, update_result.matched_count, update_result.modified_count)
                    
                    # Refresh session document with updated context
                    try:
                        updated_session = user_coll.find_one({'sessionId': session_to_update})
                        if updated_session:
                            session_doc = updated_session
                            if _should_log():
                                logger.info('Session document refreshed after account selection')
                    except Exception as refresh_error:
                        if _should_log():
                            logger.error('Failed to refresh session document: %s', str(refresh_error))
                    
                    client_account.close()
                    if _should_log():
                        logger.info('User selected TNB account: %s', selected_account)
                    
                    # Set intent to proceed with selected account
                    intent_type = 'tnb_account_selected'
                except Exception as e:
                    if _should_log():
                        logger.error('Failed to store selected TNB account: %s', str(e))
        
        # If no account selection detected, check for other TNB bill payment confirmations
        if not intent_type and _is_affirmative(message_lower):
            # Check current workflow state
            current_workflow_state = None
            try:
                client_check_state = _connect_mongo()
                chats_db = client_check_state['chats']
                user_coll = chats_db[user_id]
                session_current = new_session_generated if new_session_generated else session_id
                current_session = user_coll.find_one({'sessionId': session_current})
                if current_session and current_session.get('context'):
                    current_workflow_state = current_session['context'].get(f'{active_service}_workflow_state')
                client_check_state.close()
            except Exception:
                pass
            
            if current_workflow_state == 'tnb_bills_shown':
                # User confirmed TNB bill payment, update state
                _update_service_workflow_state('tnb_bills_confirmed')
                intent_type = f'{active_service}_payment_confirmed'
                if _should_log():
                    logger.info('User confirmed TNB bill payment, updated workflow state')

    # Check for TNB bill payment confirmations (LEGACY - for non-eKYC flow)
    elif active_service == 'pay_tnb_bill' and _is_affirmative(message_lower) and not unverified_doc_key:
        # Check current workflow state
        current_workflow_state = None
        try:
            client_check_state = _connect_mongo()
            chats_db = client_check_state['chats']
            user_coll = chats_db[user_id]
            session_current = new_session_generated if new_session_generated else session_id
            current_session = user_coll.find_one({'sessionId': session_current})
            if current_session and current_session.get('context'):
                current_workflow_state = current_session['context'].get(f'{active_service}_workflow_state')
            client_check_state.close()
        except Exception:
            pass
        
        if current_workflow_state == 'tnb_bills_shown':
            # User confirmed TNB bill payment, update state
            _update_service_workflow_state('tnb_bills_confirmed')
            intent_type = f'{active_service}_payment_confirmed'
            if _should_log():
                logger.info('User confirmed TNB bill payment, updated workflow state')
    
    # Check for TNB bill payment cancellation (when service is active and user says no)
    elif active_service == 'pay_tnb_bill' and _is_negative(message_lower) and not unverified_doc_key:
        # Check current workflow state
        current_workflow_state = None
        try:
            client_check_state = _connect_mongo()
            chats_db = client_check_state['chats']
            user_coll = chats_db[user_id]
            session_current = new_session_generated if new_session_generated else session_id
            current_session = user_coll.find_one({'sessionId': session_current})
            if current_session and current_session.get('context'):
                current_workflow_state = current_session['context'].get(f'{active_service}_workflow_state')
            client_check_state.close()
        except Exception:
            pass
        
        if current_workflow_state == 'tnb_bills_shown':
            # User declined TNB bill payment, cancel the service
            try:
                client_cancel = _connect_mongo()
                chats_db = client_cancel['chats']
                user_coll = chats_db[user_id]
                session_current = new_session_generated if new_session_generated else session_id
                
                # Set workflow state to cancelled and session status to cancelled
                user_coll.update_one(
                    {'sessionId': session_current}, 
                    {'$set': {
                        f'context.{active_service}_workflow_state': 'action_cancelled',
                        'status': 'cancelled'
                    }}
                )
                
                # Set intent type to force connection end
                intent_type = 'force_end_connection'
                
                if _should_log():
                    logger.info('User declined TNB bill payment, marked session as cancelled')
                
                client_cancel.close()
            except Exception as e:
                if _should_log():
                    logger.error('Failed to cancel TNB bill payment workflow: %s', str(e))

    # Check for confirming_end_connection and end_connection intents
    if not intent_type and session_doc and session_doc.get('context', {}).get('redirect_to_end_connection'):
        # User was redirected to end connection confirmation
        if _is_affirmative(message_lower):
            # User wants to continue with other services, clear the redirect flag
            try:
                client_clear = _connect_mongo()
                chats_db = client_clear['chats']
                user_coll = chats_db[user_id]
                session_current = new_session_generated if new_session_generated else session_id
                
                user_coll.update_one(
                    {'sessionId': session_current}, 
                    {'$unset': {
                        'context.redirect_to_end_connection': "",
                        'context.end_connection_reason': ""
                    }}
                )
                client_clear.close()
                
                # Set intent to continue with services
                intent_type = 'continue_services'
            except Exception as e:
                if _should_log():
                    logger.error('Failed to clear end connection redirect: %s', str(e))
                    
        elif _is_negative(message_lower):
            # User wants to end the session
            intent_type = 'end_connection'
            
        else:
            # User's response is unclear, set intent to ask for clarification
            intent_type = 'confirming_end_connection'

    service_ready = False
    if active_service:
        service_ready = _service_requirements_met(active_service, session_doc, ekyc)
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
                            'sessionId': session_id if session_id not in ('(new-session)', '(session-end)') else new_session_generated,
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
                    extracted_ic = extracted_data.get('userId')
                    if detected_category == 'idcard' and extracted_ic:
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
                                    'sessionId': session_id if session_id not in ('(new-session)', '(session-end)') else new_session_generated,
                                    'attachment': attachments,
                                    'intent_type': 'identity_mismatch'
                                }
                            }
                            return _cors_response(200, resp_body)
                except Exception as sec_e:
                    if _should_log():
                        logger.error('Failed during identity mismatch check: %s', str(sec_e))

                # Check document category if there's an active service
                detected_category = ocr_result.get('category_detection', {}).get('detected_category', 'unknown')
                
                # Validate document category against active service requirements
                if active_service:
                    required_category_sets = {
                        'renew_license': {'allowed': {'idcard', 'license', 'license-front'}},
                        'pay_tnb_bill': {'allowed': {'tnb'}},
                    }
                    
                    allowed_categories = required_category_sets.get(active_service, {}).get('allowed', set())
                    
                    if detected_category not in allowed_categories:
                        # Wrong document category for active service
                        if active_service == 'renew_license':
                            wrong_doc_message = (
                                "❌ **Document Type Mismatch**\n\n"
                                f"I detected this document as: **{detected_category}**\n\n"
                                "For license renewal, please upload:\n"
                                "📸 **Your current driving license** (front side), or\n"
                                "📸 **Your IC (Identity Card)** (front side)\n\n"
                                "Please upload the correct document type to proceed with your license renewal."
                            )
                        elif active_service == 'pay_tnb_bill':
                            wrong_doc_message = (
                                "❌ **Document Type Mismatch**\n\n"
                                f"I detected this document as: **{detected_category}**\n\n"
                                "For TNB bill payment, please upload:\n"
                                "📸 **Your TNB electricity bill** (showing account number and amount)\n\n"
                                "Please upload your TNB bill to proceed with the payment."
                            )
                        else:
                            wrong_doc_message = (
                                f"❌ **Document Type Mismatch**\n\n"
                                f"I detected this document as: **{detected_category}**\n\n"
                                f"This document type is not supported for the {active_service} service. "
                                "Please upload the correct document type."
                            )
                        
                        # Return early with wrong document message
                        resp_body = {
                            'status': {'statusCode': 200, 'message': 'Success'},
                            'data': {
                                'messageId': message_id,
                                'message': wrong_doc_message,
                                'createdAt': created_at_z,
                                'sessionId': session_id if session_id not in ('(new-session)', '(session-end)') else new_session_generated,
                                'attachment': attachments,
                                'intent_type': 'wrong_document_category'
                            }
                        }
                        return _cors_response(200, resp_body)

                intent_type = 'document_processing'
                session_to_save = new_session_generated if new_session_generated else session_id
                _save_document_context_to_session(user_id, session_to_save, ocr_result, attachment['name'])
                
                if _should_log():
                    logger.info('Document processed successfully. Category: %s, Intent type: %s', 
                                detected_category, intent_type)

    # Re-check service readiness if service was set during verification
    if active_service and not service_ready:
        service_ready = _service_requirements_met(active_service, session_doc, ekyc)
        if _should_log():
            logger.info('Re-checked service readiness: service=%s ready=%s', active_service, service_ready)

    # Determine prompt for Bedrock.
    try:
        # If a service is active and requirements are met, bypass model with deterministic next-step prompt
        if active_service and service_ready and intent_type not in (
            'document_processing', 'document_correction_needed', 'document_correction_provided', 'force_end_connection',
            'confirming_end_connection', 'end_connection', 'continue_services'
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

            if intent_type == 'renew_license':
                # License renewal intent - use existing prompt logic
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
                # TNB bill payment intent - check for eKYC accounts first
                tnb_accounts = ekyc.get('tnb_account_no', []) if ekyc else []
                
                if _should_log():
                    logger.info('TNB bill payment - eKYC data: %s, TNB accounts: %s', bool(ekyc), tnb_accounts)
                
                if isinstance(tnb_accounts, list) and tnb_accounts:
                    # eKYC has TNB accounts - offer account selection
                    if _should_log():
                        logger.info('Found %d eKYC TNB accounts, showing selection prompt', len(tnb_accounts))
                    
                    account_list = ""
                    for i, account in enumerate(tnb_accounts, 1):
                        account_list += f"{i}. **{account}**\n"
                    
                    # Return direct message instead of using AI prompt to ensure correct response
                    response_text = (
                        "I can help you pay your TNB electricity bill! ⚡\n\n"
                        "I found the following TNB accounts linked to your profile:\n\n"
                        f"{account_list}\n"
                        "Please select which account you'd like to pay bills for by replying with:\n"
                        "• The **number** (e.g., \"1\" or \"2\")\n"
                        "• The **account number** directly\n\n"
                        "Which TNB account would you like to pay bills for?"
                    )
                    # Skip AI model call for this direct message
                    model_error = None
                else:
                    # No eKYC accounts - use document upload prompt directly
                    if _should_log():
                        logger.info('No eKYC TNB accounts found, using document upload prompt')
                    
                    # Return direct message instead of using AI prompt to ensure correct response
                    response_text = (
                        "I can help you pay your TNB electricity bill! ⚡\n\n"
                        "To process your bill payment, I need to verify your account details and bill information. Please upload:\n\n"
                        "📸 TNB Bill Document: Take a photo of your TNB bill (the upper portion showing your account number and amount due)\n\n"
                        "Please ensure the photo is clear and all important details are visible. I'll extract the account information to help you with the payment process."
                    )
                    # Skip AI model call for this direct message
                    model_error = None
            elif intent_type == 'document_processing' and ocr_result:
                # Use document analysis prompt for processed documents (higher priority than session conditions)
                if _should_log():
                    logger.info('Using document analysis prompt for document processing')
                prompt = _generate_document_analysis_prompt(ocr_result, message)
            elif intent_type == 'document_processing':
                # Document processing without OCR result - this shouldn't happen but let's log it
                if _should_log():
                    logger.warning('Document processing intent but no OCR result available')
                prompt = (
                    "SYSTEM: A document was uploaded but processing failed. "
                    "Provide a helpful message asking the user to try uploading the document again."
                )
            elif session_id == '(new-session)':
                # For first-time connection without service intent, request a welcome message
                prompt = (
                    "SYSTEM: You are a friendly assistant that composes a short welcome message "
                    "for a government services portal called MyGovHub. The message MUST mention "
                    "that MyGovHub provides these services: license renewal, bill payments, "
                    "permit applications, checking application status, and accessing official documents. "
                    "Keep it concise (max ~120 words), helpful, and end with a call-to-action such as "
                    "'How can I help you today?'.\n\n"
                    "IMPORTANT: Respond ONLY with the welcome message text (no JSON, no explanations, no metadata)."
                )
            elif session_id == '(session-end)':
                # For session-end without service intent, request a welcome message
                prompt = (
                    "SYSTEM: You are a friendly assistant that composes a short welcome message "
                    "for a government services portal called MyGovHub. The message MUST mention "
                    "that MyGovHub provides these services: license renewal, bill payments, "
                    "permit applications, checking application status, and accessing official documents. "
                    "Keep it concise (max ~120 words), helpful, and end with a call-to-action such as "
                    "'How can I help you today?'.\n\n"
                    "IMPORTANT: Respond ONLY with the welcome message text (no JSON, no explanations, no metadata)."
                )
            elif session_id == '(continue-session)':
                # For continue session, provide direct services menu like continue_services intent
                response_text = (
                    "Perfect! I'm here to help with any other government services you need. "
                    "You can:\n\n"
                    "🔄 Renew your driving license\n"
                    "💡 Pay TNB electricity bills\n"
                    "📄 Apply for permits\n"
                    "📋 Check application status\n"
                    "📁 Access official documents\n\n"
                    "What would you like to do next?"
                )
                model_error = None  # No model error since we're bypassing the AI model
            elif intent_type == 'document_verified':
                # Document verified - provide category-specific suggestions if no active service
                if not active_service:
                    # Get the detected category from the verified document
                    detected_category = None
                    if session_doc and session_doc.get('context'):
                        for doc_key, doc_data in session_doc['context'].items():
                            if doc_data.get('isVerified') == 'verified' and doc_data.get('categoryDetection'):
                                detected_category = doc_data['categoryDetection'].get('detected_category')
                                break
                    
                    # Provide category-specific suggestions
                    if detected_category == 'tnb':
                        prompt = (
                            "SYSTEM: The user has verified their TNB electricity bill document. "
                            "Provide a brief acknowledgment and specifically suggest TNB bill payment service. "
                            "Keep it helpful and concise.\n\n"
                            f"User message: {message}\n\n"
                            "Respond with: 'Thank you for verifying your TNB bill information! "
                            "I can help you pay this electricity bill right away. Would you like me to "
                            "proceed with the TNB bill payment process? Just reply YES to continue.'"
                        )
                    elif detected_category in ('license', 'license-front', 'license-back'):
                        prompt = (
                            "SYSTEM: The user has verified their driving license document. "
                            "Provide a brief acknowledgment and specifically suggest license renewal service. "
                            "Keep it helpful and concise.\n\n"
                            f"User message: {message}\n\n"
                            "Respond with: 'Thank you for verifying your license information! "
                            "I can help you renew your driving license right away. Would you like me to "
                            "proceed with the license renewal process? Just reply YES to continue.'"
                        )
                    elif detected_category == 'idcard':
                        # For ID card, prompt user to choose what service they need
                        response_text = (
                            "Thank you for verifying your ID card information! I can help you with various government services:\n\n"
                            "🔄 **Renew driving license** - Type: \"renew license\" or \"license renewal\"\n"
                            "💡 **Pay TNB electricity bill** - Type: \"pay TNB bill\" or \"TNB payment\"\n"
                            "📄 **Apply for permits** - Type: \"apply permit\"\n"
                            "📋 **Check application status** - Type: \"check status\"\n"
                            "📁 **Access official documents** - Type: \"get documents\"\n\n"
                            "What would you like to do today?"
                        )
                        # Skip AI model call for this direct message
                        model_error = None
                    else:
                        # Generic fallback for unknown categories
                        prompt = (
                            "SYSTEM: The user has just verified their uploaded document information. "
                            "Provide a brief acknowledgment and suggest relevant government services they might need. "
                            "Keep it helpful and concise. "
                            "For ID cards or licenses, suggest license renewal services. "
                            "For bills, suggest bill payment services. "
                            "Always end with asking how you can help them today.\n\n"
                            f"User message: {message}\n\n"
                            "Respond with a helpful message acknowledging the document verification and suggesting next steps."
                        )
                else:
                    # Active service exists - let the service workflow handle the verified document
                    # This will be handled by the service next-step logic above
                    prompt = (
                        "SYSTEM: The user has verified their document and an active service is in progress. "
                        "Provide a brief acknowledgment and proceed with the service workflow.\n\n"
                        f"Active service: {active_service}\n"
                        f"User message: {message}\n\n"
                        "Acknowledge the verification and continue with the service process."
                    )
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

            elif intent_type == 'force_end_connection':
                # User declined service, end the session and force restart
                prompt = (
                    "SYSTEM: Respond ONLY with the following message (no extra sentences).\n\n"
                    "USER-FACING MESSAGE:\n"
                    "No problem! You can always return later if you change your mind. "
                    "Thank you for using MyGovHub services. Have a great day!\n\n"
                    "MyGovHub Support Team"
                )
            elif intent_type == 'confirming_end_connection':
                # Ask user if they want to continue with other services or end session - use direct response
                response_text = (
                    "Is there anything else I can help you with today? "
                    "Reply **YES** if you need other services, or **NO** to end our session.\n\n"
                    "MyGovHub Support Team"
                )
                model_error = None  # No model error since we're bypassing the AI model
            elif intent_type == 'end_connection':
                # User wants to end the session, thank them and close - use direct response
                response_text = (
                    "Thank you for using MyGovHub! We're glad we could assist you today. "
                    "Feel free to return anytime for your government service needs. Have a wonderful day! 🌟\n\n"
                    "MyGovHub Support Team"
                )
                model_error = None  # No model error since we're bypassing the AI model
            elif intent_type == 'continue_services':
                # User wants to continue with other services - use direct response
                response_text = (
                    "Perfect! I'm here to help with any other government services you need. "
                    "You can:\n\n"
                    "🔄 Renew your driving license\n"
                    "💡 Pay TNB electricity bills\n"
                    "📄 Apply for permits\n"
                    "📋 Check application status\n"
                    "📁 Access official documents\n\n"
                    "What would you like to do next?"
                )
                model_error = None  # No model error since we're bypassing the AI model
            elif intent_type == 'invalid_duration_format':
                # User provided invalid duration format - use direct response
                response_text = (
                    "⚠️ **Invalid Format**\n\n"
                    "Please enter the number of years (1-10) in one of these formats:\n\n"
                    "**Numbers:** `2`, `5`, `10`\n"
                    "**English words:** `two`, `five`, `ten`\n"
                    "**Malay words:** `dua`, `lima`, `sepuluh`\n\n"
                    "You can also add 'years' or 'tahun': `2 years`, `lima tahun`\n\n"
                    "How many years would you like to renew your license for?"
                )
                model_error = None  # No model error since we're bypassing the AI model
            else:
                # Check if user needs to be prompted for service selection
                if not active_service and not attachments and message.strip() and not session_id.startswith('('):
                    # No active service, no document upload, and user sent a message
                    # Check if message contains service intent that wasn't detected
                    service_intent = _detect_service_intent(message)
                    
                    if not service_intent:
                        # No service detected - prompt user to select a service
                        if _should_log():
                            logger.info('No active service and no service intent detected, prompting for service selection')
                        
                        response_text = (
                            "I'd be happy to help you! Please let me know what you'd like to do:\n\n"
                            "🔄 **Renew driving license** - Type: \"renew license\" or \"license renewal\"\n"
                            "💡 **Pay TNB electricity bill** - Type: \"pay TNB bill\" or \"TNB payment\"\n"
                            "📄 **Apply for permits** - Type: \"apply permit\"\n"
                            "📋 **Check application status** - Type: \"check status\"\n"
                            "📁 **Access official documents** - Type: \"get documents\"\n\n"
                            "You can also upload a document directly, and I'll automatically detect what service you need!\n\n"
                            "What would you like to do today?"
                        )
                        model_error = None  # No model error since we're bypassing the AI model
                        
                        # Skip the generic prompt building below
                        parts = None
                    else:
                        # Service intent was detected but not processed - let it fall through to generic handling
                        parts = []
                else:
                    # Generic context-building order: 1) Document context summary 2) Prior messages 3) Current user message
                    parts = []
                # Only build generic prompt if parts is not None (i.e., we haven't set a direct response)
                if parts is not None:
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
                    # 2. Prior messages
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
                    # 3. Current user message
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
                
                # Clean response text to remove unwanted prefixes
                if response_text:
                    # Remove common AI response prefixes
                    prefixes_to_remove = [
                        'ASSISTANT: ', 'SYSTEM: ', 'USER: ', 'AI: ', 'BOT: ',
                        'Assistant: ', 'System: ', 'User: ', 'Ai: ', 'Bot: '
                    ]
                    for prefix in prefixes_to_remove:
                        if response_text.startswith(prefix):
                            response_text = response_text[len(prefix):].strip()
                            break
                            
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
            
            if response_text is not None:
                assistant_msg_doc = {
                    'messageId': assistant_message_id,
                    'timestamp': created_at_iso,
                    'role': 'assistant',
                    'content': [{'text': str(response_text)}]
                }
            else:
                assistant_msg_doc = {
                    'messageId': assistant_message_id,
                    'timestamp': created_at_iso,
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

        # Handle continue_services by creating new session
        continue_services_new_session = None
        if intent_type == 'continue_services':
            try:
                client_continue = _connect_mongo()
                db_continue = client_continue['chats']
                coll_continue = db_continue[user_id]
                
                # Mark current session as completed
                session_to_complete = new_session_generated if new_session_generated else session_id
                coll_continue.update_one(
                    {'sessionId': session_to_complete}, 
                    {'$set': {'status': 'completed'}}
                )
                
                # Create new session for continue services
                continue_services_new_session = str(uuid.uuid4())
                
                # Archive any other active sessions
                coll_continue.update_many({'status': 'active'}, {'$set': {'status': 'archived'}})
                
                # Create new session document
                new_session_doc = {
                    'sessionId': continue_services_new_session,
                    'createdAt': created_at_iso,
                    'messages': [],
                    'status': 'active',
                    'service': '',
                    'context': {}
                }
                coll_continue.insert_one(new_session_doc)
                
                if _should_log():
                    logger.info('Created new session for continue_services: %s', continue_services_new_session)
                
                client_continue.close()
            except Exception as e:
                if _should_log():
                    logger.error('Failed to create new session for continue_services: %s', str(e))
        
        # Update session status to 'completed' if in confirming end connection state
        elif intent_type == 'confirming_end_connection':
            try:
                client_complete = _connect_mongo()
                db_complete = client_complete['chats']
                coll_complete = db_complete[user_id]
                session_to_complete = new_session_generated if new_session_generated else session_id
                
                coll_complete.update_one(
                    {'sessionId': session_to_complete}, 
                    {'$set': {'status': 'completed'}}
                )
                
                if _should_log():
                    logger.info('Updated session status to completed for %s intent: %s', intent_type, session_to_complete)
                
                client_complete.close()
            except Exception as e:
                if _should_log():
                    logger.error('Failed to update session status to completed: %s', str(e))

        # Prepare the MCP response payload. If model failed, still return 200 but include modelError flag
        resp_body = {
            'status': {'statusCode': 200, 'message': 'Success'},
            'data': {
                'messageId': message_id,
                'message': response_text if response_text is not None else 'ERROR: assistant failed to respond',
                'createdAt': created_at_iso,
                'sessionId': '(session-end)' if intent_type in ('force_end_connection', 'end_connection') else (continue_services_new_session if intent_type == 'continue_services' else session_to_update),
                'attachment': body.get('attachment') or []
            }
        }

        if intent_type:
            resp_body['data']['intent_type'] = intent_type
            if _should_log():
                logger.info('Final response includes intent_type: %s', intent_type)
                if intent_type == 'force_end_connection':
                    logger.info('Force end connection - returning (session-end) to indicate session termination')
                elif intent_type == 'end_connection':
                    logger.info('End connection - returning (session-end) to indicate session termination')
                elif intent_type == 'continue_services':
                    logger.info('Continue services - returning new UUID session: %s', continue_services_new_session)

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
