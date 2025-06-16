#!/usr/bin/env python3
"""
Sample script to test OpenAI API calls to Ollama model.
This demonstrates how to send a text chat message and receive a response.
"""

import os
import asyncio
import re
import logging
import httpx
from openai import AsyncOpenAI, OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Configure logging to see HTTP requests
def setup_http_logging():
    """Enable detailed HTTP logging to see the actual requests."""
    # Enable httpx logging
    logging.basicConfig(level=logging.DEBUG)

    # Create a custom logger for HTTP requests
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.DEBUG)

    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    httpx_logger.addHandler(handler)

def clean_response(text):
    """Remove <think> tags and their content from the response."""
    if not text:
        return text

    # Remove complete <think>...</think> blocks
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # Remove incomplete <think> tags (when response is cut off)
    cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL)

    # Clean up extra whitespace and newlines
    cleaned = re.sub(r'\n\s*\n+', '\n', cleaned.strip())
    cleaned = re.sub(r'^\s+|\s+$', '', cleaned)

    return cleaned

def manual_curl_example():
    """Show what the equivalent curl command would look like."""
    base_url = os.getenv("OLLAMA_URL", "http://localhost:11434/v1")

    curl_command = f'''
curl -X POST "{base_url}/chat/completions" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer ollama" \\
  -d '{{
    "model": "qwen3:8b",
    "messages": [
      {{"role": "user", "content": "who are you"}}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }}'
'''

    print("🔧 Equivalent curl command:")
    print(curl_command)
    return curl_command.strip()

def execute_curl_command():
    """Actually execute the curl command and show the result."""
    import subprocess
    import json

    base_url = os.getenv("OLLAMA_URL", "http://localhost:11434/v1")

    # Prepare the curl command
    curl_cmd = [
        "curl", "-X", "POST", f"{base_url}/chat/completions",
        "-H", "Content-Type: application/json",
        "-H", "Authorization: Bearer ollama",
        "-d", json.dumps({
            "model": "qwen3:8b",
            "messages": [
                {"role": "user", "content": "who are you"}
            ],
            "max_tokens": 100,
            "temperature": 0.7,
            "think":False,
            "stream":False
        })
    ]

    try:
        print(f"🚀 Executing curl command...{''.join(curl_cmd)}")
        print(f"📡 URL: {base_url}/chat/completions")

        # Execute the curl command
        result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=30)

        print(f"\n📋 Curl Exit Code: {result.returncode}")

        if result.returncode == 0:
            print("✅ Curl command successful!")
            print(f"📄 Raw Response:")
            print(result.stdout)

            # Try to parse and format the JSON response
            try:
                response_data = json.loads(result.stdout)
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    content = response_data['choices'][0]['message']['content']
                    cleaned_content = clean_response(content)
                    print(f"\n✨ Cleaned Response: {cleaned_content}")
            except json.JSONDecodeError:
                print("⚠️  Response is not valid JSON")

        else:
            print("❌ Curl command failed!")
            print(f"Error: {result.stderr}")

    except subprocess.TimeoutExpired:
        print("❌ Curl command timed out after 30 seconds")
    except Exception as e:
        print(f"❌ Error executing curl: {e}")

def test_with_custom_http_client():
    """Test using a custom HTTP client to see the raw requests."""
    import json

    # Create a custom HTTP client with logging
    class LoggingHTTPClient(httpx.Client):
        def request(self, method, url, **kwargs):
            print(f"\n🌐 HTTP Request:")
            print(f"   Method: {method}")
            print(f"   URL: {url}")

            if 'headers' in kwargs:
                print(f"   Headers: {dict(kwargs['headers'])}")

            if 'content' in kwargs:
                try:
                    content = kwargs['content']
                    if isinstance(content, bytes):
                        content = content.decode('utf-8')
                    parsed = json.loads(content)
                    print(f"   Body: {json.dumps(parsed, indent=2)}")
                except:
                    print(f"   Body: {kwargs['content']}")

            response = super().request(method, url, **kwargs)

            print(f"\n📡 HTTP Response:")
            print(f"   Status: {response.status_code}")
            print(f"   Headers: {dict(response.headers)}")

            return response

    # Use custom client with OpenAI
    custom_client = LoggingHTTPClient()

    client = OpenAI(
        api_key="ollama",
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
        http_client=custom_client
    )

    try:
        print("🔍 Testing with custom HTTP client (shows raw requests)...")

        response = client.chat.completions.create(
            model="qwen3:8b",
            messages=[
                {"role": "user", "content": "who are you"}
            ],
            max_tokens=150
        )
        print(response)
        print(f"\n✅ Response received:")
        print(f"   Text: {clean_response(response.choices[0].message.content)}")

    except Exception as e:
        print(f"❌ Custom client test failed: {e}")

def list_available_models():
    """List all available models in Ollama."""
    client = OpenAI(
        api_key="ollama",
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
    )

    try:
        print("📋 Available models in Ollama:")
        models = client.models.list()
        for model in models.data:
            print(f"  • {model.id}")
        return [model.id for model in models.data]
    except Exception as e:
        print(f"❌ Could not list models: {e}")
        return []

async def test_ollama_chat():
    """Test function to send a chat message to Ollama via OpenAI API."""

    # Configure the OpenAI client to point to Ollama
    client = AsyncOpenAI(
        api_key="ollama",  # Ollama doesn't require a real API key
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
    )

    try:
        print("🤖 Connecting to Ollama...")
        print(f"📡 Base URL: {client.base_url}")
        print(f"🎯 Model: qwen3:8b")
        print("-" * 50)

        # Send the chat message with explicit instructions to not show thinking
        response = await client.chat.completions.create(
            model="qwen3:8b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a friendly, helpful assistant. Respond directly without showing your thinking process or reasoning steps. Do not include <think> tags or explain your thought process. Just give the direct answer."
                },
                {
                    "role": "user",
                    "content": "who are you"
                }
            ],
            max_tokens=150,
            temperature=0.7,
        )

        # Extract and clean the response
        raw_response = response.choices[0].message.content
        cleaned_response = clean_response(raw_response)

        print("👤 User: who are you")
        print(f"🤖 Raw Response: {raw_response}")
        print(f"✨ Cleaned Response: {cleaned_response}")
        print("-" * 50)
        print("✅ Test completed successfully!")

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print(f"🔍 Error type: {type(e).__name__}")

        # Check if Ollama is running
        print("\n💡 Troubleshooting tips:")
        print("1. Make sure Ollama is running: `ollama serve`")
        print("2. Check if the model is available: `ollama list`")
        print("3. Pull the model if needed: `ollama pull qwen3:8b`")
        print("4. Verify the base URL is correct")

def test_simple_chat():
    """Simple synchronous chat test with response cleaning."""
    client = OpenAI(
        api_key="ollama",
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
    )

    try:
        print("🤖 Simple chat test...")

        response = client.chat.completions.create(
            model="qwen3:8b",
            messages=[
                {"role": "user", "content": "who are you"}
            ],
            max_tokens=100,
        )

        raw_response = response.choices[0].message.content
        cleaned_response = clean_response(raw_response)

        print("👤 User: who are you")
        print(f"🤖 Raw: {raw_response}")
        print(f"✨ Cleaned: {cleaned_response}")
        return True

    except Exception as e:
        print(f"❌ Simple test failed: {e}")
        return False

def test_with_available_model(models):
    """Test with the first available model that's not qwen3:8b."""
    if not models:
        return False

    # Try to find a model that might not have thinking behavior
    preferred_models = ["llama3.2:1b", "llama3.2:3b", "llama3.1:8b", "phi3", "gemma2"]
    test_model = None

    for preferred in preferred_models:
        if preferred in models:
            test_model = preferred
            break

    if not test_model and models:
        # Use the first available model that's not qwen3:8b
        test_model = next((m for m in models if m != "qwen3:8b"), models[0])

    if not test_model:
        print("❌ No alternative models available")
        return False

    client = OpenAI(
        api_key="ollama",
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
    )

    try:
        print(f"🤖 Testing with {test_model}...")

        response = client.chat.completions.create(
            model=test_model,
            messages=[
                {"role": "user", "content": "who are you"}
            ],
            max_tokens=100,
        )

        raw_response = response.choices[0].message.content
        cleaned_response = clean_response(raw_response)

        print("👤 User: who are you")
        print(f"🤖 {test_model}: {cleaned_response}")
        return True

    except Exception as e:
        print(f"❌ {test_model} test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Ollama API Test with HTTP Request Inspection")
    print("=" * 70)

    # Show equivalent curl command
    #print("\n📋 Step 1: Equivalent curl command")
    #manual_curl_example()

    # Execute the actual curl command
    print("\n📋 Step 2: Execute curl command")
    execute_curl_command()

    # Test with custom HTTP client to see raw requests
    print("\n📋 Step 3: Testing with custom HTTP client (shows raw requests)")
    test_with_custom_http_client()

    # Optional: Enable detailed HTTP logging (uncomment to see all HTTP traffic)
    print("\n📋 Step 4: Enable detailed HTTP logging? (y/n)")
    #enable_logging = input().lower().strip() == 'y'
    enable_logging = True

    if enable_logging:
        print("🔧 Enabling detailed HTTP logging...")
        setup_http_logging()

    # First, list available models
    print("\n📋 Step 5: Listing available models")
    available_models = list_available_models()

    # Test simple chat
    print("\n📋 Step 6: Simple chat test with qwen3:8b")
    simple_success = test_simple_chat()

    if simple_success:
        print("\n📋 Step 7: Async chat test with response cleaning")
        asyncio.run(test_ollama_chat())

        if available_models:
            print("\n📋 Step 8: Testing with alternative model")
            test_with_available_model(available_models)
    else:
        print("\n⚠️  Skipping other tests due to connection failure")

    print("\n🏁 All tests completed!")
    print("\n💡 Key Insights:")
    print("   • qwen3:8b shows <think> tags by design")
    print("   • Use clean_response() function to remove them")
    print("   • Different models have different behaviors")
    print("   • System prompts can help but may not eliminate thinking completely")
    print("   • HTTP requests are made to /chat/completions endpoint")
    print("   • Authorization header uses 'Bearer ollama' (dummy key)")
