from flask import Flask, render_template
from flask import  request, jsonify
import fitz  # PyMuPDF
import os
import re
import ftfy
from unicodedata import normalize
import requests

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
API_KEY = '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'

def clean_text_ai(text):
    url = 'https://openrouter.ai/api/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }

    prompt = f"Clean the following text by removing unnecessary symbols, fixing grammar, and improving readability:\n\n{text}"

    data = {
        'model': 'mistralai/mistral-small-3.1-24b-instruct',
        'messages': [{'role': 'user', 'content': prompt}]
    } 

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        cleaned_text = result['choices'][0]['message']['content'].strip()
        return cleaned_text
    else:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

extracted_text = ""
def clean_text(text):
    """
    Cleans the input text by performing the following steps:
    1. Fixes encoding issues.
    2. Normalizes text (lowercase, diacritics removal).
    3. Removes special characters and symbols.
    4. Removes extra whitespace, tabs, and line breaks.
    """
    # Fix encoding issues
    text = ftfy.fix_text(text)

    # Normalize Unicode characters (e.g., é → e)
    text = normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

    # Remove special characters and symbols (keep alphanumeric and basic punctuation)
    text = re.sub(r'[^\w\s.,!?]', '', text)

    # Remove extra whitespace, tabs, and line breaks
    text = re.sub(r'\s+', ' ', text).strip()

    return text


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/team')
def teams():
    return render_template('team.html')


@app.route('/essay')
def Essay():
    return render_template('Essay.html')


@app.route('/flashcard')
def FlashCard():
    return render_template('FlashCard.html')

@app.route('/input')
def inputPage():
    return render_template('input.html')

@app.route("/upload", methods=["POST"])
def upload_file():
    global extracted_text  # Referencing the global variable

    print("Upload route triggered")  # Debugging statement

    if "file" not in request.files:
        print("No file found in request")
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        print("No file selected")
        return jsonify({"error": "No selected file"}), 400

    # Save file
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    print(f"File saved: {file_path}")  # Debugging statement

    # Read PDF content
    if file.filename.endswith(".pdf"):
        extracted_text = extract_text_from_pdf(file_path)  # Store extracted text in global variable
        print("Raw Extracted Text:", extracted_text)  # Debugging statement

        ai_cleaned_text = clean_text_ai(extracted_text)
        print("AI Cleaned Text:", ai_cleaned_text)  # Debugging statement

        final_cleaned_text = clean_text(ai_cleaned_text)
        extracted_text = final_cleaned_text 
        print("Final Cleaned Text:", extracted_text)  # Debugging statement

        return jsonify({"message": "File uploaded and processed successfully."})

    return jsonify({"message": "File uploaded successfully"})

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

if __name__ == '__main__':
    app.run(debug=True)
