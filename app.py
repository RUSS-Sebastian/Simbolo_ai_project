from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import os
import re
import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'default-secret-key')
socketio = SocketIO(app, cors_allowed_origins="*")

# Model configuration
ESSAY_SCORER_PATH = r"D:\model_for_Q\essay_scorer"
os.makedirs(ESSAY_SCORER_PATH, exist_ok=True)

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

@app.route("/")
def home():
    return render_template("essay.html")

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

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)