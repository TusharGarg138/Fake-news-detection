import os
from flask import Flask, render_template, request, jsonify
from gemini_validator import validate_text  # Import the mock function

# --- Initialize Flask App ---
app = Flask(__name__)

# Create an uploads folder if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# --- MODEL INFERENCE FUNCTIONS ---

def predict_english_text(text):
    """
    Calls the Gemini validator for English text (synchronous).
    """
    print(f"Analyzing English text (with Gemini): {text[:50]}...")
    res = validate_text(text)
    prediction = res.get('prediction', '').lower()

    if prediction == 'true':
        return {
            'prediction': 'true',
            'verdict': '✅ True Statement',
            'details': 'The content appears to be accurate and factual.',
            'color': 'green'
        }
    elif prediction == 'false':
        return {
            'prediction': 'false',
            'verdict': '❌ False Statement',
            'details': 'This statement contains false or misleading information.',
            'color': 'red'
        }
    elif prediction == 'partially':
        return {
            'prediction': 'partially',
            'verdict': '⚠️ Partially True Statement',
            'details': 'Some parts of this content are accurate while others may be misleading.',
            'color': 'yellow'
        }
    else:
        return {
            'prediction': 'false',
            'verdict': '❌ False Statement',
            'details': 'No reliable data found. Treated as false.',
            'color': 'red'
        }


def predict_hindi_text(text):
    """
    Calls the Gemini validator for Hindi text (synchronous).
    """
    print(f"Analyzing Hindi text (with Gemini): {text[:50]}...")
    res = validate_text(text)
    prediction = res.get('prediction', '').lower()

    if prediction == 'true':
        return {
            'prediction': 'true',
            'verdict': '✅ सत्य कथन',
            'details': 'सामग्री सटीक और तथ्यात्मक है।',
            'color': 'green'
        }
    elif prediction == 'false':
        return {
            'prediction': 'false',
            'verdict': '❌ असत्य कथन',
            'details': 'यह कथन भ्रामक या गलत है।',
            'color': 'red'
        }
    elif prediction == 'partially':
        return {
            'prediction': 'partially',
            'verdict': '⚠️ आंशिक रूप से सत्य कथन',
            'details': 'कुछ भाग सही हैं जबकि कुछ भ्रामक हो सकते हैं।',
            'color': 'yellow'
        }
    else:
        return {
            'prediction': 'false',
            'verdict': '❌ असत्य कथन',
            'details': 'कोई स्पष्ट डेटा नहीं मिला।',
            'color': 'red'
        }


def predict_image(image_file):
    """
    Loads the Fakeddit multimodal model and returns a prediction for the image.
    Currently uses mock data for demo purposes.
    """
    filename = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
    image_file.save(filename)
    print(f"Analyzing image: {filename}")

    # Demo mock data
    return {
        'prediction': 'partially',
        'verdict': '⚠️ Partially True Statement',
        'details': 'Image analysis shows some authentic and some misleading elements.',
        'analysis': {
            'true_parts': ["Image element A appears authentic."],
            'false_parts': ["Text overlay B could not be verified."]
        },
        'color': 'yellow'
    }


# --- FLASK ROUTES ---

@app.route('/')
def home():
    """Render the main HTML page."""
    # This MUST match the name of your file in the 'templates' folder
    return render_template('index1.html')



@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Receives data from the frontend, calls the appropriate model,
    and returns the result as JSON.
    """
    try:
        input_type = request.form.get('input_type')
        result = None

        if input_type == 'english':
            text = request.form.get('text_input')
            if text and text.strip():
                result = predict_english_text(text)

        elif input_type == 'hindi':
            text = request.form.get('text_input')
            if text and text.strip():
                result = predict_hindi_text(text)

        elif input_type == 'image':
            if 'image_input' in request.files:
                file = request.files['image_input']
                if file.filename:
                    result = predict_image(file)

        if result:
            return jsonify(result)
        else:
            return jsonify({'error': 'Invalid input or missing data.'}), 400

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': f'An internal error occurred: {e}'}), 500


# --- MAIN EXECUTION ---
if __name__ == '__main__':
    app.run(debug=True)
