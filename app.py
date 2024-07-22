import platform
import subprocess
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
import openai
import os
import webbrowser
from threading import Timer
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
    summaries = {}
    for filename in os.listdir(DOCUMENTS_FOLDER):
        if filename.endswith('.txt'):
            with open(os.path.join(DOCUMENTS_FOLDER, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                documents[filename] = content
                summaries[filename] = extract_summary(content)
    return documents, summaries

def extract_summary(text):
    """
    Extract summary from the given text using GPT.
    """
    prompt = f"Extract a short summary for the following text:\n\n{text}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    summary = response.choices[0].message['content'].strip()
    return summary

# Load documents and their summaries at startup
documents, summaries = load_documents()

def vectorize_documents(summaries):
    """
    Vectorize the summaries of the documents.
    """
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(summaries.values())
    return vectorizer, vectors

vectorizer, vectors = vectorize_documents(summaries)

def search_documents(query):
    """
    Search documents based on a query and return ranked results.
    """
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, vectors).flatten()
    ranked_indices = np.argsort(similarities)[::-1]
    ranked_files = [list(summaries.keys())[i] for i in ranked_indices]
    return ranked_files, similarities[ranked_indices]

def query_gpt(prompt, context):
    """
    Query the GPT model with a user prompt and context from documents.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
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
    
    # Log user's search query
    print(f"User's search query: {user_query}")
    
    # Search documents based on the user's query
    ranked_files, similarities = search_documents(user_query)
    
    # Log the ranked files and their similarities
    print(f"Ranked files: {ranked_files}")
    print(f"Similarities: {similarities}")
    
    # Read the top N most relevant files for context
    top_n = 5
    relevant_context = ""
    for file in ranked_files[:top_n]:
        relevant_context += f"Summary of {file}:\n{summaries[file]}\n\n"
    
    # Log the relevant context
    print(f"Relevant context for GPT:\n{relevant_context}")
    
    # Query GPT with the relevant context
    response_text = query_gpt(user_query, relevant_context)
    
    # Log the GPT response
    print(f"GPT response: {response_text}")
    
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
        model="gpt-4",
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
        model="gpt-4",
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
        
        # Update documents and summaries
        documents[filename] = transcription
        summaries[filename] = summary
        vectorizer, vectors = vectorize_documents(summaries)
        
        # Log the save operation
        print(f"Transcription saved as {filename}")
        
        return jsonify({'message': f'Transcription saved as {filename}'})
    
    return jsonify({'message': 'No transcription to save'}), 400

if __name__ == '__main__':
    # Open the browser after a short delay
    Timer(1, open_browser).start()
    app.run(debug=False)
