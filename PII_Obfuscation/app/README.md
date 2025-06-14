# PII Obfuscation App

## Overview
The **PII Obfuscation App** is a secure Flask-based API designed to detect and redact Personally Identifiable Information (PII) from text input. It leverages a **pre-trained token classification model** to identify sensitive data and replace it with obfuscated text. The app ensures authentication using **HTTP Basic Authentication** and is suitable for integration into security-conscious workflows.

## Features
- **PII Detection & Redaction**: Uses a **DistilBERT-based model** to identify and mask PII entities.
- **Secure Authentication**: Requires HTTP Basic Authentication to prevent unauthorized access.
- **JSON API**: Accepts text input via JSON requests and returns redacted output.
- **Custom Model & Tokenizer**: Loads a fine-tuned token classification model for PII recognition.

## Installation
1. **Clone the repository**:
   ```sh
   git clone https://github.com/your-repo/pii-obfuscation-app.git
   cd pii-obfuscation-app
   ```
2. **Set up a virtual environment (optional but recommended)**:
    ```sh 
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3. **Install dependencies**:
    ```sh 
    pip install -r requirements.txt
    ```

## Usage
### API Endpoint: /
- Method: POST
- Headers: Authorization: Basic <base64-encoded-credentials>

**Request Body (JSON)**:
```json 
{
  "text": "
Host: Hello! How can I assist you today? 
Customer: I would like to update my address
Host: I can assist you with that, can you please confirm your name, email address, username and date of birth for me?
Customer: My name is John Doe and my username is JDoe123 and my email is John.Doe@mail.com and my DOB is 1995/02/16
"
}
```
**Response (JSON)**:
```json 
{
  "redacted_text": "
Host: Hello! How can I assist you today? 
Customer: I would like to update my address
Host: I can assist you with that, can you please confirm your name, email address, username and date of birth for me?
Customer: My name is ###### #### and my username is ########## and my email is ############### and my DOB is ##########
"
}
```
## Security Considerations
- Ensure strong credentials are used for authentication.
- Run the application in a secure environment to prevent exposure of sensitive data.
- HTTPS should be used when deploying to production to encrypt data in transit.
## Future Enhancements
- Support for additional PII types.
- Integration with logging and monitoring for auditability.
- Deployment using Docker for ease of use.