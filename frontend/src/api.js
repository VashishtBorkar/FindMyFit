const API_BASE = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

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
  matchCategories.forEach((category) => {
    formData.append("match_categories", category);
  });
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
      id: item.item_id,
      category: item.category,
      score: Math.round(item.score * 1000) / 10,
      image: item.image_url ? `${API_BASE}${item.image_url}` : null,
    })),
  };
}
