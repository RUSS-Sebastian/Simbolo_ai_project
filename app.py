from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_socketio import SocketIO
import fitz  # PyMuPDF
import os
import re
import ftfy
from unicodedata import normalize
import requests
import time
import threading  # Needed for background processing
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from flask_socketio import emit
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask and SocketIO
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
API_KEY = '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'
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'default-secret-key')
# Model configuration
ESSAY_SCORER_PATH = r"D:\model_for_Q\essay_scorer"
os.makedirs(ESSAY_SCORER_PATH, exist_ok=True)

# Initialize Flask-SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")
qa_pairs = [] 

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

def clean_text(text):
    text = ftfy.fix_text(text)
    text = normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'[^\w\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()  # Read the entire content of the file
        return content
    except FileNotFoundError:
        return "File not found."
    except Exception as e:
        return f"An error occurred: {str(e)}"

def QuestionGenerator(context,no_qa,difficulty):

  model_name = r"D:\model_for_Q\final_ver"
  tokenizer = T5Tokenizer.from_pretrained(model_name)
  model = T5ForConditionalGeneration.from_pretrained(model_name)
  prompt = f"Generate five {difficulty} diverse questions based on the following paragraph: {context}"
  # Tokenize input
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids

  # Generate multiple questions
  output_ids = model.generate(input_ids, max_length=100, num_return_sequences=no_qa, do_sample=True, top_k=50)
  questions = [tokenizer.decode(output, skip_special_tokens=True) for output in output_ids]
  return questions

def AnswerGenerator(context,questions):
  model_name = "D:\model_for_Q\original_t5_base_before_finetune"  # You can use "t5-base" or "flan-t5" for better performance
  tokenizer = T5Tokenizer.from_pretrained(model_name)
  model = T5ForConditionalGeneration.from_pretrained(model_name)
  answers= []
  for i in range(len(questions)):
        input_text = f"question: {questions[i]} context: {context}"
        inputs = tokenizer(input_text, return_tensors="pt")
        output = model.generate(**inputs, max_length=30, repetition_penalty=2.5, num_beams=5)
        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        answers.append(answer)

  return answers

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/team")
def teams():
    return render_template("team.html")

@app.route("/essay")
def Essay():
    return render_template("Essay.html")

def load_essay_scorer():
    try:
        logger.info("Loading essay scoring model...")
        tokenizer = AutoTokenizer.from_pretrained(ESSAY_SCORER_PATH)
        model = AutoModelForCausalLM.from_pretrained(ESSAY_SCORER_PATH)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        logger.info("Model loaded successfully on device: %s", device)
        
        return tokenizer, model
    except Exception as e:
        logger.error("Failed to load model: %s", str(e))
        return None, None

essay_tokenizer, essay_model = load_essay_scorer()


@app.route("/score", methods=["POST"])
def score_essay():
    if not essay_model:
        return jsonify({"error": "Scoring service unavailable"}), 503
        
    data = request.get_json()
    essay_text = data.get("essay", "").strip()
    
    if not essay_text:
        return jsonify({"error": "No essay provided"}), 400
    
    word_count = len(essay_text.split())
    if word_count < 50:
        return jsonify({"error": f"Essay must be at least 50 words (current: {word_count})"}), 400
    
    try:
        thread = threading.Thread(target=process_essay, args=(essay_text,))
        thread.daemon = True
        thread.start()
        return jsonify({"message": "Essay scoring started"})
    except Exception as e:
        logger.error("Error starting scoring thread: %s", str(e))
        return jsonify({"error": str(e)}), 500

def process_essay(essay_text):
    try:
        # Progress update
        socketio.emit("essay_progress", {
            "progress": 20,
            "message": "Processing text"
        })
        
        # Clean text
        cleaned_text = re.sub(r'\s+', ' ', essay_text).strip()
        
        # Create prompt
        prompt = f"""Evaluate this essay and provide a score from 1-6:

            Essay: {cleaned_text}

            Please provide a score between 1 and 6 where:
            1 = Very Poor
            2 = Poor
            3 = Below Average  
            4 = Average
            5 = Good
            6 = Excellent

            Score:"""
        
        # Tokenize
        socketio.emit("essay_progress", {
            "progress": 50,
            "message": "Analyzing content"
        })
        
        inputs = essay_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(essay_model.device)
        
        # Generate response
        with torch.no_grad():
            output = essay_model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.7,
                do_sample=True,
                pad_token_id=essay_tokenizer.eos_token_id
            )
        
        # Process output
        response = essay_tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info("Model response: %s", response)
        
        # Extract score
        score = extract_score_from_response(response)
        feedback = generate_feedback(score)
        
        # Send results
        socketio.emit("essay_result", {
            "score": score,
            "feedback": feedback
        })
        
    except Exception as e:
        logger.error("Error processing essay: %s", str(e))
        socketio.emit("essay_error", {
            "error": "Failed to process essay. Please try again."
        })

def extract_score_from_response(response):
    # Try multiple patterns to find the score
    patterns = [
        r"Score:\s*([1-6](?:\.[0-9])?)",  # "Score: X"
        r"([1-6](?:\.[0-9])?)\s*out of 6",  # "X out of 6"
        r"\b([1-6](?:\.[0-9])?)\b"  # Just a number 1-6
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            try:
                score = float(match.group(1))
                return min(max(round(score, 1), 1), 6)  # Clamp to 1-6 with 1 decimal
            except ValueError:
                continue
    
    # If no score found, use average
    return 3.5

def generate_feedback(score):
    if score >= 5.5:
        return "Excellent essay! Clear thesis, strong arguments, and excellent organization."
    elif score >= 4.5:
        return "Good essay. Solid content with minor areas for improvement in development."
    elif score >= 3.5:
        return "Average essay. Needs work on structure and supporting evidence."
    elif score >= 2.5:
        return "Below average. Focus on clarity and providing more examples."
    else:
        return "Needs significant improvement. Work on basic structure and content."

@app.route("/flashcard")
def FlashCard():
    return render_template("Flashcard.html", qa_pairs=qa_pairs)  # Pass data to the template

@app.route("/input")
def inputPage():
    return render_template("input.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)
    
    thread = threading.Thread(target=process_file, args=(file_path,))
    thread.start()
    
    return jsonify({"message": "File uploaded successfully. Processing started."})

@app.route("/upload-text", methods=["POST"])
def upload_text():
    # Get the text from the incoming request
    data = request.get_json()
    text = data.get('text')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    # Save the text to a temporary file or directly process it
    # In this case, we simulate a temporary file by saving it to the system
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], "uploaded_text.txt")
    with open(file_path, 'w') as f:
        f.write(text)
    
    # Start a background thread to process the text (same as file processing)
    thread = threading.Thread(target=process_text, args=(file_path,))
    thread.start()
    
    return jsonify({"message": "Text uploaded successfully. Processing started."})

def process_text(file_path):
    """Processes the uploaded file and emits progress updates."""
    global qa_pairs  # Ensure we can modify the global variable
    
    with app.app_context():
        socketio.emit("progress_update", {"progress": 25})
        time.sleep(2)
        extracted_text = read_text_file(file_path)
        ai_cleaned_text = clean_text_ai(extracted_text)
        final_cleaned_text = clean_text(ai_cleaned_text)

        socketio.emit("progress_update", {"progress": 50})
        time.sleep(2)
        questions = QuestionGenerator(final_cleaned_text, 6, "medium")
        

        socketio.emit("progress_update", {"progress": 70})
        time.sleep(2)
        answers = AnswerGenerator(final_cleaned_text, questions)

        socketio.emit("progress_update", {"progress": 100})
        time.sleep(2)
        # Store generated Q&A pairs in global variable
        qa_pairs = [{"question": q, "answer": a} for q, a in zip(questions, answers)]
        

        # Redirect to flashcard page
        socketio.emit("redirect", {"url": url_for("FlashCard")})

def process_file(file_path):
    """Processes the uploaded file and emits progress updates."""
    global qa_pairs  # Ensure we can modify the global variable
    
    with app.app_context():
        socketio.emit("progress_update", {"progress": 25})
        time.sleep(2)
        extracted_text = extract_text_from_pdf(file_path)
        ai_cleaned_text = clean_text_ai(extracted_text)
        final_cleaned_text = clean_text(ai_cleaned_text)

        socketio.emit("progress_update", {"progress": 50})
        time.sleep(2)
        questions = QuestionGenerator(final_cleaned_text, 6, "medium")
        

        socketio.emit("progress_update", {"progress": 70})
        time.sleep(2)
        answers = AnswerGenerator(final_cleaned_text, questions)

        socketio.emit("progress_update", {"progress": 100})
        time.sleep(2)
        # Store generated Q&A pairs in global variable
        qa_pairs = [{"question": q, "answer": a} for q, a in zip(questions, answers)]
        

        # Redirect to flashcard page
        socketio.emit("redirect", {"url": url_for("FlashCard")})


app.config["SERVER_NAME"] = "127.0.0.1:5000"

if __name__ == "__main__":
    socketio.run(app, debug=True)

