import os
import streamlit as st
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer, util

# --- 1. Page Configuration & Title ---
st.set_page_config(page_title="CLIP Semantic Search", page_icon="🔍", layout="wide")
st.title("🔍 Image-Text Semantic Search using CLIP Embeddings")
st.markdown(
    "Type a description below, and the local CLIP model will find the most visually similar images from your `images/` directory."
)

# --- 2. Cache Model Loading ---
# This ensures the model only loads into memory once, making the app lightning fast after setup.
@st.cache_resource
def load_model():
    # Using the standard OpenAI CLIP model via sentence-transformers
    model = SentenceTransformer('clip-ViT-B-32')
    return model

model = load_model()

# --- 3. Image Directory Setup ---
IMAGE_DIR = "images"

# Create the directory automatically if it doesn't exist yet
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# Fetch all valid image files
supported_extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(supported_extensions)]

# --- 4. Main App Logic ---
if not image_files:
    st.warning(f"⚠️ No images found in the `{IMAGE_DIR}/` directory!")
    st.info(f"Please drop some images (e.g., dog.jpg, cat.jpg, car.jpg) into the `{IMAGE_DIR}/` folder and refresh the page.")
else:
    # Sidebar status
    st.sidebar.header("Project Info")
    st.sidebar.success(f"Indexed {len(image_files)} images from your local folder.")
    
    # User Search Input
    query_text = st.text_input("What are you looking for?", placeholder="e.g., 'a furry animal sleeping' or 'sports car'")
    
    if query_text:
        with st.spinner("Searching through image embeddings..."):
            # A. Load and open all images using Pillow
            opened_images = []
            for img_name in image_files:
                img_path = os.path.join(IMAGE_DIR, img_name)
                opened_images.append(Image.open(img_path))
            
            # B. Generate Embeddings (Vectors)
            # Encode images (Vision Encoder) and the text query (Text Encoder)
            image_embeddings = model.encode(opened_images, convert_to_tensor=True)
            text_embedding = model.encode(query_text, convert_to_tensor=True)
            
            # C. Compute Similarities
            # util.cos_sim returns a matrix of similarity scores between 0 and 1
            cos_scores = util.cos_sim(text_embedding, image_embeddings)[0]
            
            # D. Get Top Results
            # Find top 3 results (or fewer if total images < 3)
            top_k = min(3, len(image_files))
            top_results = torch.topk(cos_scores, k=top_k)
            
        # --- 5. Display Results ---
        st.subheader(f"🎯 Top {top_k} Most Similar Images")
        
        # Create flexible side-by-side columns based on top_k
        cols = st.columns(top_k)
        
        for idx, (score, image_idx) in enumerate(zip(top_results.values, top_results.indices)):
            actual_idx = image_idx.item()
            img_name = image_files[actual_idx]
            img_obj = opened_images[actual_idx]
            similarity_pct = float(score) * 100
            
            with cols[idx]:
                # Display the image neatly along with its calculated score
                st.image(img_obj, use_container_width=True)
                st.markdown(f"**Filename:** `{img_name}`")
                st.metric(label="Match Confidence", value=f"{similarity_pct:.2f}%")

#python code


                
