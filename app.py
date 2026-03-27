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
def load_recommender() -> ClothingRecommender:
    return ClothingRecommender(
        recommendation_engine_type="metric",
        images_dir=images_dir
    )


# ---------- MODAL STATE ----------
if "show_info" not in st.session_state:
    st.session_state.show_info = False


def toggle_info():
    st.session_state.show_info = not st.session_state.show_info


# ---------- UI COMPONENTS ----------

def render_centered_hero():
    st.markdown(
        """
        <div style='text-align: center; padding-top: 10px;'>
            <h1>FindMyFit</h1>
            <p style='font-size:18px; color: gray;'>
                Upload a clothing item and get AI-powered outfit recommendations
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_card(rec):
    rec_path = rec.recommended_item.image_path

    if not rec_path.exists():
        rec_path = Path(str(rec_path) + ".jpg")

    with st.container(border=True):
        if rec_path.exists():
            rec_img = Image.open(rec_path)
            # st.markdown('<div style="text-align:center;">', unsafe_allow_html=True)
            st.image(rec_img, width="content")
            # st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("Image not found")

        st.markdown(
            f"""
            <div style="text-align:center;">
                <b>{rec.recommended_item.category.upper()}</b><br>
                <span style="color: gray;">
                    {rec.confidence_score * 100:.1f}% match
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )


# ---------- MAIN ----------

def main():
    st.set_page_config(layout="wide")

    recommender = load_recommender()

    if "recommendations" not in st.session_state:
        st.session_state.recommendations = None

    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None

    # Hero
    render_centered_hero()

    # Upload
    st.markdown("### Upload Your Item")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.session_state.uploaded_image = image

    # After upload → show controls
    if st.session_state.uploaded_image:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(st.session_state.uploaded_image, caption="Your Item")

        with col2:
            st.markdown("### Customize Recommendations")

            allowed_categories = ClothingRecommender.get_allowed_categories()

            target_category = st.selectbox(
                "Item category",
                allowed_categories
            )

            match_categories = st.multiselect(
                "Match with",
                allowed_categories,
                default=["pants", "shoes"]
            )

            max_recommendations = st.slider(
                "Number of results",
                1, 20, 6
            )

            st.markdown("")

            # CTA
            generate = st.button(
                "Generate Recommendations",
                width="stretch"
            )

            if generate:
                with st.spinner("Finding matches..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                        st.session_state.uploaded_image.save(tmp_file.name)
                        tmp_path = tmp_file.name

                    try:
                        results = recommender.get_recommendations(
                            image_path=tmp_path,
                            target_category=target_category,
                            match_categories=match_categories,
                            max_recommendations=max_recommendations
                        )

                        st.session_state.recommendations = results

                    except Exception as e:
                        st.error(str(e))

                    finally:
                        os.unlink(tmp_path)

    # Results
    # st.markdown("---")
    st.markdown("### Recommendations")

    results = st.session_state.recommendations

    if results is None:
        st.info("Upload an item to get started.")

    elif not results:
        st.warning("No matches found. Try different categories.")

    else:
        cols = st.columns(3)

        for idx, rec in enumerate(results):
            col = cols[idx % 3]

            with col:
                render_card(rec)


if __name__ == "__main__":
    main()