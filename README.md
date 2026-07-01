# Image-Text Semantic Search using CLIP Embeddings

A sleek web app that lets you search through a local folder of images using natural language text queries. 

### What it does?

* Takes a text query from the user via a Streamlit input box.
* Automatically reads all images from a local `images/` folder.
* Converts both the text and images into vectors using OpenAI's CLIP model (`clip-ViT-B-32`).
* Computes the cosine similarity between the text vector and all image vectors.
* Displays the top 3 most visually matching images alongside their confidence scores.

### The Tech Stack

* **Streamlit** — for the clean, minimalist UI web interface.
* **Sentence-Transformers** — for loading the CLIP model and running vector utilities.
* **Pillow (PIL)** — for handling and opening local image files.
* **PyTorch** — the backend framework powering the vector math.
* **No API Keys Needed** — everything runs completely locally on your hardware.

### How do you run this?

1. **Setup your folders:**
   Make sure you have an `images/` directory in your project folder and drop 5-10 random images (`.jpg`, `.png`, etc.) inside it.

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
