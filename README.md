# Memory Assist

Forgetting something? Memory Assist is a tool that helps you manage and recall important information effectively; from lectures to meetings, and even daily life. While this program was initially intended for individuals who experience frequent forgetfulness due to conditions like ADHD, alzheimers, and amnesia, Memory Assist can be instrumental in helping students and professionals remember important tasks, due dates, and homework assignments.

## Table of Contents

- [How It Works](#how-it-works)
  - [Audio Transcription](#audio-transcription)
  - [Semantic Search Engine](#semantic-search-engine)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## How It Works

### Audio Transcription

Memory Assist allows you to record audio transcriptions of important information. When you save a transcription, it is processed using the GPT API to generate a concise summary and extract relevant keywords. The transcription, along with its summary, keywords, and metadata (date and time of recording), is added to the `documents` folder as a .txt file.

### Semantic Search Engine

Memory Assist uses a semantic search engine to find relevant information through TF-IDF vectorization. The search engine begins by vectorizing all saved transcription documents. When a query is entered, the program also vectorizes the user query, then calculates the cosine similarity between the query vector and the document vectors. This similarity is used to rank documents based on their relevance to the query, and the most relevant documents are selected to provide context for answering the question. This context is sent to the GPT API, along with the original user query, and the generated response is displayed to the user.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/John-Auerbach/Memory-Assist.git
    cd memory-assist
    ```

2. Install the necessary dependencies:

    ```bash
    pip install
    ```

## Configuration

Before using Memory Assist, you need to configure your API key.

1. Create a `.env` file in the `memory-assist` folder.

    ```bash
    touch .env
    ```

2. Open the `.env` file in a text editor and add your API key:

    ```text
    API_KEY=your_api_key
    ```

    Replace `your_api_key` with your actual API key.

## Usage

To start the application, run:

```bash
python app.py
