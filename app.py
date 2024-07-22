import platform
import subprocess

# existing imports
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
import openai
import os
import webbrowser
from threading import Timer
from datetime import datetime

load_dotenv()
api_key = os.getenv("API_KEY")

app = Flask(__name__)
openai.api_key = api_key  # Replace with your OpenAI API key

# Dynamically find the path of the folder that the app is in and then the documents folder inside it
APP_FOLDER = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_FOLDER = os.path.join(APP_FOLDER, 'documents')

def load_documents():
    documents = {}
    for filename in os.listdir(DOCUMENTS_FOLDER):
        if filename.endswith('.txt'):
            with open(os.path.join(DOCUMENTS_FOLDER, filename), 'r', encoding='utf-8') as file:
                documents[filename] = file.read()
    return documents

documents = load_documents()

def query_gpt(prompt, context):
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
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    user_query = request.form.get('query')
    
    # Combine all document contents to provide context
    combined_documents = "\n\n".join(documents.values())
    
    response_text = query_gpt(user_query, combined_documents)
    return jsonify({'response': response_text})

def open_browser():
    url = "http://127.0.0.1:5000/"
    if platform.system() == "Windows":
        chrome_path = "C:/Program Files/Google/Chrome/Application/chrome.exe %s"
        webbrowser.get(chrome_path).open(url)
    elif platform.system() == "Darwin":  # macOS
        subprocess.run(["open", "-a", "Google Chrome", url])
    else:  # Linux
        subprocess.run(["google-chrome", url])

@app.route('/save_transcription', methods=['POST'])
def save_transcription():
    data = request.get_json()
    transcription = data.get('transcription', '')
    
    if transcription:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'transcription_{timestamp}.txt'
        file_path = os.path.join(DOCUMENTS_FOLDER, filename)
        
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(transcription)
        
        return jsonify({'message': f'Transcription saved as {filename}'})
    
    return jsonify({'message': 'No transcription to save'}), 400

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(debug=False)
