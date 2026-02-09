# app.py

import streamlit as st
from pathlib import Path
from PIL import Image
import tempfile
import os
from dotenv import load_dotenv

from src.fashion_matcher.clothing_recommender import ClothingRecommender

load_dotenv()
images_dir = Path(os.getenv("IMAGES_DIR", "data/images"))


@st.cache_resource
def load_recommender(engine_type: str) -> ClothingRecommender:
    """Load and cache the recommender to avoid reloading on every interaction."""
    return ClothingRecommender(
        recommendation_engine_type=engine_type,
        images_dir=images_dir
    )


def main():
    st.set_page_config(
        page_title="FindMyFit",
        page_icon="👗",
        layout="wide"
    )
    
    st.title("👗 FindMyFit")
    st.markdown("Upload a clothing item and get outfit recommendations!")
    
    # Sidebar for settings
    st.sidebar.header("Settings")
    
    engine_type = st.sidebar.selectbox(
        "Recommendation Engine",
        options=["metric", "cosine"],
        index=0,
        help="Choose the recommendation algorithm"
    )
    
    max_recommendations = st.sidebar.slider(
        "Max Recommendations",
        min_value=1,
        max_value=20,
        value=5,
        help="Number of recommendations to show"
    )
    
    # Get allowed categories
    allowed_categories = ClothingRecommender.get_allowed_categories()
    
    target_category = st.sidebar.selectbox(
        "Target Item Category",
        options=allowed_categories,
        index=allowed_categories.index("top") if "top" in allowed_categories else 0,
        help="Category of the item you're uploading"
    )
    
    match_categories = st.sidebar.multiselect(
        "Match Categories",
        options=allowed_categories,
        default=["pants", "skirt", "shoes"],
        help="Categories to find recommendations from"
    )
    
    # Load recommender
    recommender = load_recommender(engine_type)
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Upload Your Item")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload a clothing item image"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Your Item", use_container_width=True)
    
    with col2:
        st.subheader("Recommendations")
        
        if uploaded_file is not None and match_categories:
            if st.button("🔍 Get Recommendations", type="primary"):
                with st.spinner("Finding matching items..."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                        image.save(tmp_file.name)
                        tmp_path = tmp_file.name
                    
                    try:
                        recommendations = recommender.get_recommendations(
                            image_path=tmp_path,
                            target_category=target_category,
                            match_categories=match_categories,
                            max_recommendations=max_recommendations
                        )
                        
                        if not recommendations:
                            st.warning("No recommendations found. Try different categories.")
                        else:
                            # Display recommendations in a grid
                            cols_per_row = 5
                            for i in range(0, len(recommendations), cols_per_row):
                                cols = st.columns(cols_per_row)
                                for j, col in enumerate(cols):
                                    idx = i + j
                                    if idx < len(recommendations):
                                        rec = recommendations[idx]
                                        with col:
                                            # Try to load the image
                                            rec_path = rec.recommended_item.image_path
                                            
                                            # Handle file extension
                                            if not rec_path.exists():
                                                rec_path = Path(str(rec_path) + ".jpg")
                                            
                                            if rec_path.exists():
                                                rec_img = Image.open(rec_path)
                                                st.image(rec_img, use_container_width=True)
                                            else:
                                                st.error("Image not found")
                                            
                                            st.caption(
                                                f"**{rec.recommended_item.category}**\n\n"
                                                f"Score: {rec.confidence_score:.3f}"
                                            )
                    
                    except Exception as e:
                        st.error(f"Error getting recommendations: {str(e)}")
                    
                    finally:
                        # Clean up temp file
                        os.unlink(tmp_path)
        
        elif uploaded_file is None:
            st.info("👈 Upload an image to get started")
        
        elif not match_categories:
            st.warning("Please select at least one match category")


if __name__ == "__main__":
    main()