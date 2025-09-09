import os
import markdown
from flask import Flask, request, send_from_directory, render_template_string
from llama_cpp import Llama
import math
import json

app = Flask(__name__, static_url_path='/static', static_folder='static')

# Load the sentence embedding model
# Qwen/Qwen3-Embedding-0.6B-GGUF
# Qwen3-Embedding-0.6B-Q8_0.gguf
llm = Llama.from_pretrained(repo_id="Qwen/Qwen3-Embedding-0.6B-GGUF", filename="Qwen3-Embedding-0.6B-Q8_0.gguf", embedding=True)

# Dictionary to store embeddings
embeddings = {}

# Function to load markdown files and create embeddings
def load_markdown_files_and_embeddings(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()
                embedding = llm.embed(content, normalize=True)
                embeddings[file_path] = embedding

# Load markdown files and embeddings on startup
load_markdown_files_and_embeddings('content')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/content/<path:filename>')
def content(filename):
    with open(os.path.join('content', filename), 'r') as f:
        content = f.read()
    html = markdown.markdown(content)
    return render_template_string(html)

@app.route('/search')
def search():
    query = request.args.get('query')
    query_embedding = llm.embed(query, normalize=True)
    
    similarities = {}
    for path, embedding in embeddings.items():
        similarity = cosine_similarity(query_embedding, embedding)
        similarities[path] = similarity
    
    sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    top_10_matches = [path for path, similarity in sorted_similarities[:10]]
    
    return json.dumps(top_10_matches)

# Pure Python implementation of cosine similarity
def cosine_similarity(vec1, vec2):
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(v1**2 for v1 in vec1))
    magnitude2 = math.sqrt(sum(v2**2 for v2 in vec2))
    return dot_product / (magnitude1 * magnitude2)

if __name__ == '__main__':
    app.run(debug=True)