const API_BASE = "http://localhost:8000";

export async function fetchCategories() {
  const response = await fetch(`${API_BASE}/categories`);

  if (!response.ok) {
    throw new Error("Failed to fetch categories");
  }

  return response.json();
}

export async function generateRecommendations({
  imageFile,
  targetCategory,
  matchCategories,
  maxRecommendations,
}) {
  const formData = new FormData();
  formData.append("image", imageFile);
  formData.append("target_category", targetCategory);
  formData.append("match_categories", JSON.stringify(matchCategories));
  formData.append("max_recommendations", maxRecommendations);

  const response = await fetch(`${API_BASE}/recommend`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new Error(error?.detail || "Failed to generate recommendations");
  }

  const data = await response.json();

  return {
    recommendations: data.recommendations.map((item) => ({
      ...item,
      image: item.image ? `${API_BASE}${item.image}` : null,
    })),
  };
}