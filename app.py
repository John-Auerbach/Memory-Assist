import platform
import subprocess
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
import openai
import os
import webbrowser
from threading import Timer
from datetime import datetime

# Load environment variables from a .env file
load_dotenv()
api_key = os.getenv("API_KEY")

# Initialize the Flask app
app = Flask(__name__)
openai.api_key = api_key  # Set the OpenAI API key

# Dynamically find the path of the folder that the app is in and then the documents folder inside it
APP_FOLDER = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_FOLDER = os.path.join(APP_FOLDER, 'documents')

def load_documents():
    """
    Load all .txt documents from the documents folder and store their contents in a dictionary.
    """
    documents = {}
    for filename in os.listdir(DOCUMENTS_FOLDER):
        if filename.endswith('.txt'):
            with open(os.path.join(DOCUMENTS_FOLDER, filename), 'r', encoding='utf-8') as file:
                documents[filename] = file.read()
    return documents

# Load documents at startup
documents = load_documents()

def query_gpt(prompt, context):
    """
    Query the GPT model with a user prompt and context from documents.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # Updated model
        messages=[
            {"role": "system", "content": "Answer the question given the following context"},
            {"role": "user", "content": f"{context}\n\nUser question: {prompt}"}
        ],
        max_tokens=150
    )
    return response.choices[0].message['content'].strip()

@app.route('/')
def index():
    """
    Render the index.html template at the root URL.
    """
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    """
    Handle POST requests to the /query endpoint, querying GPT with the user's input.
    """
    user_query = request.form.get('query')
    
    # Combine all document contents to provide context
    combined_documents = "\n\n".join(documents.values())
    
    response_text = query_gpt(user_query, combined_documents)
    return jsonify({'response': response_text})

def open_browser():
    """
    Open the default web browser to the app's URL.
    """
    url = "http://127.0.0.1:5000/"
    if platform.system() == "Windows":
        chrome_path = "C:/Program Files/Google/Chrome/Application/chrome.exe %s"
        webbrowser.get(chrome_path).open(url)
    elif platform.system() == "Darwin":  # macOS
        subprocess.run(["open", "-a", "Google Chrome", url])
    else:  # Linux
        subprocess.run(["google-chrome", url])

def generate_summary(transcription):
    """
    Generate a summary for the given transcription using GPT.
    """
    prompt = f"Provide a short summary for the following text:\n\n{transcription}"
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # Updated model
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    summary = response.choices[0].message['content'].strip()
    return summary

def generate_keywords(transcription):
    """
    Extract relevant keywords from the given transcription using GPT.
    """
    prompt = f"Extract a list of comma-separated relevant keywords from the following text:\n\n{transcription}"
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # Updated model
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50
    )
    keywords = response.choices[0].message['content'].strip()
    return keywords

@app.route('/save_transcription', methods=['POST'])
def save_transcription():
    """
    Save a transcription with a summary and keywords, and store it in the documents folder.
    """
    data = request.get_json()
    transcription = data.get('transcription', '')
    
    if transcription:
        # Create a timestamped filename for the transcription
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        date = datetime.now().strftime('%Y-%m-%d')
        time = datetime.now().strftime('%H:%M:%S')
        filename = f'transcription_{timestamp}.txt'
        file_path = os.path.join(DOCUMENTS_FOLDER, filename)
        
        # Generate summary and keywords
        summary = generate_summary(transcription)
        keywords = generate_keywords(transcription)
        
        # Save the transcription, summary, and keywords to a file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(f"Date: {date}\n")
            file.write(f"Time: {time}\n")
            file.write(f"Keywords: {keywords}\n")
            file.write(f"Summary: {summary}\n\n")
            file.write(transcription)
        
        return jsonify({'message': f'Transcription saved as {filename}'})
    
    return jsonify({'message': 'No transcription to save'}), 400

if __name__ == '__main__':
    # Open the browser after a short delay
    Timer(1, open_browser).start()
    app.run(debug=False)