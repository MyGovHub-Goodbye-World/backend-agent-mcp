# Backend Agent MCP

A serverless AWS Lambda function providing AI-powered government service assistance for MyGovHub, handling license renewal and TNB bill payments.

© 2025 Goodbye World team, for Great AI Hackathon Malaysia 2025 usage.

## Request Format

JSON payload with the following structure:
```json
{
  "message": "User message or service request",
  "userId": "Malaysian IC number (e.g., 010203040506)",
  "createdAt": "ISO 8601 timestamp (e.g., 2025-10-02T01:03:00.000Z)",
  "sessionId": "Session identifier or '(new-session)'",
  "attachment": [],
  "ekyc": {}
}
```

## Response Format

JSON response containing:
```json
{
  "status": {
    "statusCode": 200,
    "message": "Success"
  },
  "data": {
    "messageId": "UUID",
    "message": "AI-generated response text",
    "createdAt": "ISO 8601 timestamp",
    "sessionId": "Session identifier",
    "attachment": [],
    "intent_type": "Optional intent classification"
  }
}
```

## Intent

- `renew_license`: Driving license renewal service
- `pay_tnb_bill`: TNB electricity bill payment service

## Workflow for License Renewal

1. Document upload and OCR verification
2. License record lookup from database
3. Duration selection (1-10 years)
4. Payment confirmation via Billplz
5. License record update and PDF generation
6. Receipt generation

## Workflow for TNB Bill Payment

1. Document upload and OCR verification
2. Bill lookup from database
3. Account selection (if multiple)
4. Payment confirmation via Billplz
5. Bill status update to paid
6. Receipt generation

## APIs Used

- **AWS Bedrock**: AI model for responses and intent detection
- **MongoDB Atlas**: Data storage for sessions, licenses, and bills
- **OCR API**: Document text extraction and categorization
- **Billplz API**: Payment processing
- **License Generator API**: PDF license document creation
- **Receipt Generator API**: PDF receipt generation

## Environment Configuration

Required environment variables:
- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `AWS_REGION1`: AWS region for Bedrock

- `BEDROCK_MODEL_ID`: AI model identifier
- `BEDROCK_MAX_TOKENS`: Maximum response tokens
- `BEDROCK_TEMPERATURE`: AI creativity level
- `BEDROCK_TOP_P`: Token selection probability

- `SHOW_CLOUDWATCH_LOGS`: Logging enable flag

- `JPJ_COLLECTION_ID`: License payment collection identifier
- `TNB_COLLECTION_ID`: Bill payment collection identifier
- `JPJ_API_KEY`: License payment API key
- `TNB_API_KEY`: Bill payment API key

- `OCR_ANALYZE_API_URL`: Document processing endpoint
- `PAYMENT_CREATE_BILL_API_URL`: Payment creation endpoint
- `LICENSE_GENERATOR_API_URL`: License PDF generation endpoint
- `GENERATE_RECEIPT_API_URL`: Receipt PDF generation endpoint

- `ATLAS_URI`: MongoDB connection string
- `ATLAS_DB_NAME`: Database name
