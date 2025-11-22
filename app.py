import os
import requests
import json
import base64
from urllib.parse import unquote
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins="*", allow_headers=["Content-Type", "Authorization"], methods=["GET", "POST", "OPTIONS"])

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        print(f"Global OPTIONS handler for {request.path}")
        print(f"Request headers: {dict(request.headers)}")
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "Content-Type,Authorization,Accept,Accept-Language,Accept-Encoding,Connection,Host,Origin,Referer,Sec-Fetch-Dest,Sec-Fetch-Mode,Sec-Fetch-Site,User-Agent")
        response.headers.add('Access-Control-Allow-Methods', "GET,PUT,POST,DELETE,OPTIONS")
        response.headers.add('Access-Control-Max-Age', "3600")
        return response, 200

# ------------------------------------------------------------------
# UPDATED: Get the token from AWS App Runner Environment Variables
# ------------------------------------------------------------------
BEARER_TOKEN = os.environ.get("BEARER_TOKEN", "")

def call_bedrock_api(messages, model="claude-sonnet-4"):
    """Call Bedrock API using Claude Code's authentication method"""
    try:
        # Safety Check: Ensure token exists before processing
        if not BEARER_TOKEN:
            print("CRITICAL ERROR: BEARER_TOKEN is missing. Check App Runner Configuration.")
            return "Error: Server API Token is not configured."

        # Decode the bearer token to get the pre-signed URL
        encoded_url = BEARER_TOKEN.replace("bedrock-api-key-", "")
        decoded_url = base64.b64decode(encoded_url + "===").decode('utf-8')

        # Extract system messages and convert to Claude format
        system_prompt = ""
        claude_messages = []

        for msg in messages:
            if msg.get("role") == "system":
                # Collect system messages into a single system prompt
                system_prompt += msg.get("content", "") + "\n"
            elif msg.get("role") in ["user", "assistant"]:
                content = msg.get("content", "")

                # Handle multimodal content (text + images)
                if isinstance(content, list):
                    claude_content = []
                    for item in content:
                        if item.get("type") == "text":
                            claude_content.append({
                                "type": "text",
                                "text": item.get("text", "")
                            })
                        elif item.get("type") == "image_url":
                            # Extract base64 image data
                            image_url = item.get("image_url", {}).get("url", "")
                            if image_url.startswith("data:image/"):
                                # Extract mime type and base64 data
                                header, data = image_url.split(',', 1)
                                mime_type = header.split(':')[1].split(';')[0]
                                claude_content.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": mime_type,
                                        "data": data
                                    }
                                })
                    claude_messages.append({
                        "role": msg.get("role"),
                        "content": claude_content
                    })
                else:
                    # Simple text message
                    claude_messages.append({
                        "role": msg.get("role"),
                        "content": content
                    })

        # Prepare the Bedrock request payload
        bedrock_payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": claude_messages,
            "max_tokens": 1000
        }

        # Add system prompt if present
        if system_prompt.strip():
            bedrock_payload["system"] = system_prompt.strip()

        # Try to make a direct call to Bedrock using the pre-signed URL approach
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {BEARER_TOKEN}"
        }

        # Use the Bedrock runtime endpoint
        bedrock_url = f"https://bedrock-runtime.eu-central-1.amazonaws.com/model/eu.anthropic.claude-sonnet-4-5-20250929-v1:0/invoke"

        response = requests.post(
            bedrock_url,
            headers=headers,
            json=bedrock_payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            return result.get("content", [{}])[0].get("text", "No response")
        else:
            print(f"Bedrock API error: {response.status_code} - {response.text}")
            return f"Error calling Bedrock: {response.status_code}"

    except Exception as e:
        print(f"Exception calling Bedrock: {str(e)}")
        return f"Error: {str(e)}"

@app.route('/v1/models', methods=['GET', 'OPTIONS'])
def list_models():
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
    models = {
        "object": "list",
        "data": [
            {
                "id": "claude-sonnet-4.5",
                "object": "model",
                "created": 1234567890,
                "owned_by": "anthropic",
                "permission": [],
                "root": "claude-sonnet-4",
                "parent": None
            }
        ]
    }
    return jsonify(models)

@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
def chat_completions():
    if request.method == 'OPTIONS':
        print("Handling OPTIONS request for /v1/chat/completions")
        response = jsonify({})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.status_code = 200
        return response
    try:
        data = request.json
        messages = data.get('messages', [])
        model = data.get('model', 'claude-sonnet-4')

        # Call the actual Bedrock API
        bedrock_response = call_bedrock_api(messages, model)

        response = {
            "id": "chatcmpl-bedrock",
            "object": "chat.completion",
            "created": 1234567890,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": bedrock_response
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Ensure this matches the port configured in App Runner (8080)
    print("Starting Bedrock proxy on http://0.0.0.0:8080")
    app.run(host='0.0.0.0', port=8080, debug=True)