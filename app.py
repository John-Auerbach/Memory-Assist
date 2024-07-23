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
import textwrap

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
    Extract the first few lines of the given text as a summary.
    """
    lines = text.split('\n')
    summary = '\n'.join(lines[:5])  # Assuming the first 5 lines are the summary
    return summary

# Load documents and their summaries at startup
documents, summaries = load_documents()

def vectorize_documents(documents):
    """
    Vectorize the contents of the documents.
    """
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents.values())
    return vectorizer, vectors

vectorizer, vectors = vectorize_documents(documents)

def search_documents(query):
    """
    Search documents based on a query and return ranked results.
    """
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, vectors).flatten()
    ranked_indices = np.argsort(similarities)[::-1]
    ranked_files = [list(documents.keys())[i] for i in ranked_indices]
    return ranked_files, similarities[ranked_indices]

def query_gpt(prompt, context):
    """
    Query the GPT model with a user prompt and context from documents.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Answer the question given the following context"},
            {"role": "user", "content": f"{context}\n\nUser question: {prompt}"}
        ],
        max_tokens=200
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
    print(f"\nUser query: {user_query}")
    
    # Search documents based on the user's query
    ranked_files, similarities = search_documents(user_query)
    
    # Log the ranked files and their similarities
    print("\nRanked files and their similarities:")
    for i, file in enumerate(ranked_files[:5]):
        branch = "└──" if i == 4 else "├──"
        print(f"{branch} {file} (Similarity: {similarities[i]:.4f})")
    
    # Read the top N most relevant files for context
    top_n = 5
    relevant_context = ""
    for file in ranked_files[:top_n]:
        relevant_context += f"Contents of {file}:\n{documents[file]}\n\n"
    
    # Log only the summaries (first 5 lines) of the relevant context
    relevant_summaries = ""
    for file in ranked_files[:top_n]:
        summary = summaries[file]
        lines = summary.split('\n')
        date = lines[0]
        time = lines[1]
        keywords = lines[2]
        summary_content = '\n'.join(lines[3:])
        
        relevant_summaries += f"   {file}:\n"
        relevant_summaries += f"   ├── {date}\n"
        relevant_summaries += f"   ├── {time}\n"
        wrapped_keywords = textwrap.fill(keywords, width=70, subsequent_indent='   |   ')
        relevant_summaries += f"   ├── {wrapped_keywords}\n"
        wrapped_summary_content = textwrap.fill(summary_content, width=70, subsequent_indent='       ')
        relevant_summaries += f"   └── {wrapped_summary_content}\n\n"
    
    # Log the relevant summaries
    print("\nRelevant file summaries:\n")
    for line in relevant_summaries.split('\n'):
        print(f"{line}")
    
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
    Generate a short summary for the given transcription using GPT.
    """
    prompt = f"Briefly summarize the following text:\n\n{transcription}"
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    summary = response.choices[0].message['content'].strip()
    return summary

def generate_keywords(transcription):
    """
    Extract relevant keywords from the given transcription using GPT.
    """
    prompt = f"Extract a list of comma-separated relevant keywords from the following text:\n\n{transcription}"
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
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
        vectorizer, vectors = vectorize_documents(documents)
        
        # Log the save operation
        print(f"Transcription saved as {filename}")
        
        return jsonify({'message': f'Transcription saved as {filename}'})
    
    return jsonify({'message': 'No transcription to save'}), 400

if __name__ == '__main__':
    # Open the browser after a short delay
    Timer(1, open_browser).start()
    app.run(debug=False)
