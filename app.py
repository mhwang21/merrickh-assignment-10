from flask import Flask, request, render_template
import os
import numpy as np
import torch
from PIL import Image
from open_clip import get_tokenizer, create_model_and_transforms
import pandas as pd
import torch.nn.functional as F
from sklearn.decomposition import PCA

app = Flask(__name__)

# Ensure device is properly set
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model, tokenizer, and preprocess
model, _, preprocess = create_model_and_transforms('ViT-B/32', pretrained='openai')
tokenizer = get_tokenizer('ViT-B-32')
model = model.to(device)
model.eval()

# Load embeddings and files
df = pd.read_pickle('image_embeddings.pickle')
embeddings = np.vstack(df['embedding'].values)
file_names = [os.path.join('static/coco_images_resized', fname) for fname in df['file_name'].values]

# PCA setup
pca = PCA(n_components=10)
pca_embeddings = pca.fit_transform(embeddings)

# Helper functions
def compute_similarity(query_embedding, embeddings):
    cosine_similarities = np.dot(embeddings, query_embedding.T).squeeze()
    top_indices = np.argsort(cosine_similarities)[::-1][:5]
    return [(file_names[idx], cosine_similarities[idx]) for idx in top_indices]

def process_text_query(text):
    text_token = tokenizer([text])
    text_embedding = F.normalize(model.encode_text(text_token).to(device)).detach().cpu().numpy()
    return text_embedding

def process_image_query(image_file):
    image = preprocess(Image.open(image_file).convert("RGB")).unsqueeze(0).to(device)
    image_embedding = F.normalize(model.encode_image(image)).detach().cpu().numpy()
    return image_embedding

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    error = None
    if request.method == 'POST':
        query_type = request.form.get('query_type')
        weight = float(request.form.get('weight', 0.5))
        use_pca = request.form.get('use_pca') == 'on'
        query_embedding = None

        try:
            #text query
            if query_type == 'text':
                text = request.form.get('text_query')
                query_embedding = process_text_query(text)
                if use_pca:
                    query_embedding = pca.transform(query_embedding.reshape(1, -1))

            #image query
            elif query_type == 'image':
                image_file = request.files['image_query']
                temp_path = os.path.join('static', 'temp', image_file.filename)
                image_file.save(temp_path)
                query_embedding = process_image_query(temp_path)
                if use_pca:
                    query_embedding = pca.transform(query_embedding.reshape(1, -1))

            #combined query
            elif query_type == 'both':
                text = request.form.get('text_query')
                image_file = request.files['image_query']
                temp_path = os.path.join('static', 'temp', image_file.filename)
                image_file.save(temp_path)

                text_embedding = process_text_query(text)
                image_embedding = process_image_query(temp_path)
                combined_embedding = F.normalize(weight * torch.tensor(text_embedding, dtype=torch.float32) +
                                                  (1.0 - weight) * torch.tensor(image_embedding, dtype=torch.float32), p=2, dim=1)
                combined_embedding_numpy = combined_embedding.detach().cpu().numpy()
                query_embedding = pca.transform(combined_embedding_numpy) if use_pca else combined_embedding_numpy

            #top 5 matches
            results = compute_similarity(query_embedding.squeeze(), pca_embeddings if use_pca else embeddings)
        except Exception as e:
            error = f"An error occurred: {e}"

    return render_template('index.html', results=results, error=error)

if __name__ == '__main__':
    if not os.path.exists('static/temp'):
        os.makedirs('static/temp')
    app.run(debug=True)