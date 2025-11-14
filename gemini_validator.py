import httpx
import json
import time
import re
import base64
from io import BytesIO
from PIL import Image

# --- Configuration ---
API_KEY = "AIzaSyD-EviX3jsLIuIYACD5OVPBNH0O3gmsa9U"
TEXT_MODEL = "gemini-2.5-flash-preview-09-2025"
VISION_MODEL = "gemini-2.5-flash-preview-09-2025"
MAX_RETRIES = 3
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/"

# --- System Prompts ---

FACT_CHECK_SYSTEM_PROMPT = """
You are a meticulous fact-checker. Your task is to analyze the user's text and determine its factual accuracy.
You MUST use the Google Search tool to find real-time, verifiable information.

You will be given a 'main_claim' from the user.
Classify it into exactly ONE of these:
- TRUE
- FALSE
- PARTIALLY

Return ONLY valid JSON in this exact format:
{
  "classification": "TRUE" | "FALSE" | "PARTIALLY",
}
"""
#"verdict_details": "short explanation"

IMAGE_ANALYSIS_SYSTEM_PROMPT = """
You are a digital media forensics expert. Analyze the image and classify it as:
- TRUE (authentic)
- FALSE (fake)
- PARTIALLY (real but misleading)

Return ONLY valid JSON in this format:
{
  "classification": "TRUE" | "FALSE" | "PARTIALLY",
}
"""
#"verdict_details": "short explanation"

# --- Helper Functions ---

def make_api_call(url, payload):
    headers = {'Content-Type': 'application/json'}
    for attempt in range(MAX_RETRIES):
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload, headers=headers)

            if response.status_code == 200:
                return response.json()
            else:
                print(f"HTTP Error: {response.status_code} {response.text}")
                if 400 <= response.status_code < 500:
                    return {"error": response.text}
                time.sleep(2 ** attempt)

        except httpx.RequestError as e:
            print(f"Request failed: {e}. Retrying ({attempt + 1}/{MAX_RETRIES})...")
            time.sleep(2 ** attempt)

    return {"error": "API call failed after several retries."}

def extract_json_from_text(text):
    """
    Robust JSON extractor — handles markdown, code blocks, and text clutter.
    """
    if not text:
        return None

    # Remove markdown/codeblock formatting if present
    cleaned = text.replace("```json", "").replace("```", "").strip()

    # Try to find JSON braces even if surrounded by text
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        json_string = match.group(0)
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            print("Retrying JSON parse after cleanup...")
            try:
                cleaned = re.sub(r"^[^{]*", "", cleaned)
                cleaned = re.sub(r"[^}]*$", "", cleaned)
                return json.loads(cleaned)
            except Exception as e:
                print("JSON still invalid:", e)
                return None
    return None

def create_error_response(message):
    return {
        'prediction': 'false',
        'verdict': 'Analysis Failed',
        'details': f"An internal error occurred: {message}. Please check your connection and try again.",
        'color': 'red'
    }

# --- Main Functions ---

def validate_text(text_to_check):
    print("Gemini Validator: Fact-checking text...")
    url = f"{BASE_URL}{TEXT_MODEL}:generateContent?key={API_KEY}"
    
    payload = {
        "contents": [{"parts": [{"text": f"main_claim: \"{text_to_check}\""}]}],
        "systemInstruction": {"parts": [{"text": FACT_CHECK_SYSTEM_PROMPT}]},
        "tools": [{"google_search": {}}]
    }

    api_result = make_api_call(url, payload)

    if "error" in api_result:
        return create_error_response(api_result["error"])

    try:
        raw_text = api_result["candidates"][0]["content"]["parts"][0]["text"]
        parsed_json = extract_json_from_text(raw_text)

        if not parsed_json:
            return create_error_response("Could not parse AI response.")

        prediction = parsed_json.get("classification", "").lower()
        verdict = parsed_json.get("verdict_details", "No details provided.")

        # Fallback safety
        if prediction not in ["true", "false", "partially"]:
            prediction = "false"

        color_map = {'true': 'green', 'false': 'red', 'partially': 'yellow'}

        return {
            'prediction': prediction,
            'verdict': verdict,
            'details': verdict,
            'color': color_map.get(prediction, 'red')
        }

    except Exception as e:
        print("Error processing API response:", e)
        print("Full response:", api_result)
        return create_error_response(f"Could not parse API response: {e}")

def validate_image(image_file):
    print("Gemini Validator: Analyzing image...")
    try:
        img = Image.open(image_file.stream)
        img_format = img.format
        if not img_format:
            return create_error_response("Could not determine image format.")

        image_file.stream.seek(0)
        img_bytes = image_file.stream.read()
        base64_image = base64.b64encode(img_bytes).decode('utf-8')
        mime_type = f"image/{img_format.lower()}"

    except Exception as e:
        print(f"Error processing image: {e}")
        return create_error_response(f"Invalid image file: {e}")

    url = f"{BASE_URL}{VISION_MODEL}:generateContent?key={API_KEY}"
    payload = {
        "contents": [{
            "parts": [
                {"text": "Analyze this image based on the system prompt."},
                {"inlineData": {"mimeType": mime_type, "data": base64_image}}
            ]
        }],
        "systemInstruction": {"parts": [{"text": IMAGE_ANALYSIS_SYSTEM_PROMPT}]}
    }

    api_result = make_api_call(url, payload)

    if "error" in api_result:
        return create_error_response(api_result["error"])

    try:
        raw_text = api_result["candidates"][0]["content"]["parts"][0]["text"]
        parsed_json = extract_json_from_text(raw_text)

        if not parsed_json:
            return create_error_response("Could not parse AI response.")

        prediction = parsed_json.get("classification", "").lower()
        verdict = parsed_json.get("verdict_details", "No details provided.")

        if prediction not in ["true", "false", "partially"]:
            prediction = "false"

        color_map = {'true': 'green', 'false': 'red', 'partially': 'yellow'}

        return {
            'prediction': prediction,
            'verdict': verdict,
            'details': verdict,
            'color': color_map.get(prediction, 'red')
        }

    except Exception as e:
        print("Error processing API response:", e)
        print("Full response:", api_result)
        return create_error_response(f"Could not parse API response: {e}")
