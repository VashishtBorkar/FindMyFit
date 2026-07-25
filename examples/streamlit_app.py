"""Secondary Streamlit demo backed by the same public runtime facade."""

import os
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from findmyfit import ClothingRecommender, Settings


@st.cache_resource
def load_recommender() -> ClothingRecommender:
    settings = Settings.from_env()
    return ClothingRecommender(settings.recommender_engine, settings=settings)


def render_card(recommendation) -> None:
    item = recommendation.recommended_item
    with st.container(border=True):
        if item.image_path.is_file():
            st.image(Image.open(item.image_path), width="content")
        else:
            st.error("Image unavailable")
        st.markdown(
            f"**{item.category.upper()}**  \n"
            f"{recommendation.confidence_score * 100:.1f}% match"
        )


def main() -> None:
    st.set_page_config(page_title="FindMyFit", layout="wide")
    st.title("FindMyFit")
    st.caption("Upload a clothing item and find compatible catalog pieces.")

    recommender = load_recommender()
    categories = recommender.get_allowed_categories()
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is None:
        return

    preview = Image.open(uploaded).convert("RGB")
    st.image(preview, caption="Uploaded item", width=320)
    target_category = st.selectbox("Item category", categories)
    match_categories = st.multiselect("Match with", categories, default=["pants", "shoes"])
    max_recommendations = st.slider("Number of results", 1, 20, 6)

    if not st.button("Generate recommendations"):
        return
    suffix = Path(uploaded.name).suffix or ".png"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
            temporary_path = temporary.name
        preview.save(temporary_path)
        results = recommender.get_recommendations(
            temporary_path,
            target_category,
            match_categories,
            max_recommendations,
        )
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)

    columns = st.columns(3)
    for index, recommendation in enumerate(results):
        with columns[index % 3]:
            render_card(recommendation)


if __name__ == "__main__":
    main()
