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

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
API_KEY = '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'

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
