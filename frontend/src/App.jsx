import { useEffect, useState } from "react";
import HeroSection from "./components/HeroSection";
import UploadSection from "./components/UploadSection";
import SettingsPanel from "./components/SettingsPanel";
import RecommendationsGrid from "./components/RecommendationsGrid";
import { fetchCategories, generateRecommendations } from "./api";

export default function App() {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [categories, setCategories] = useState([]);
  const [targetCategory, setTargetCategory] = useState("");
  const [matchCategories, setMatchCategories] = useState([]);
  const [maxRecommendations, setMaxRecommendations] = useState(6);
  const [recommendations, setRecommendations] = useState(null);
  const [loadingCategories, setLoadingCategories] = useState(true);
  const [loadingRecommendations, setLoadingRecommendations] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadCategories() {
      try {
        setLoadingCategories(true);
        const data = await fetchCategories();
        const cats = data.categories || [];
        setCategories(cats);

        if (cats.length > 0) {
          setTargetCategory(cats[0]);

          if (cats.includes("pants") && cats.includes("shoes")) {
            setMatchCategories(["pants", "shoes"]);
          } else {
            setMatchCategories(cats.slice(0, 2));
          }
        }
      } catch (err) {
        setError(err.message || "Failed to load categories");
      } finally {
        setLoadingCategories(false);
      }
    }

    loadCategories();
  }, []);

  const handleUpload = (file) => {
    setUploadedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setRecommendations(null);
    setError("");
  };

  const toggleMatchCategory = (category) => {
    setMatchCategories((prev) =>
      prev.includes(category)
        ? prev.filter((item) => item !== category)
        : [...prev, category]
    );
  };

  const handleGenerate = async () => {
    if (!uploadedFile) return;

    if (!targetCategory) {
      setError("Please select an item category.");
      return;
    }

    if (matchCategories.length === 0) {
      setError("Please select at least one category to match with.");
      return;
    }

    try {
      setLoadingRecommendations(true);
      setError("");

      const data = await generateRecommendations({
        imageFile: uploadedFile,
        targetCategory,
        matchCategories,
        maxRecommendations,
      });

      setRecommendations(data.recommendations);
    } catch (err) {
      setError(err.message || "Failed to generate recommendations");
    } finally {
      setLoadingRecommendations(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#f8f6f1] text-zinc-900">
      <div className="relative overflow-hidden">
        <div className="absolute -left-24 top-8 h-72 w-72 rounded-full bg-[#eadcc3]/60 blur-3xl" />
        <div className="absolute -right-24 top-64 h-80 w-80 rounded-full bg-[#f3ead8]/70 blur-3xl" />

        <main className="relative z-10 mx-auto w-full max-w-7xl px-4 pb-20 pt-10 sm:px-6 lg:px-8">
          <HeroSection />

          <UploadSection onUpload={handleUpload} previewUrl={previewUrl} />

          {uploadedFile && !loadingCategories && (
            <SettingsPanel
              categories={categories}
              targetCategory={targetCategory}
              setTargetCategory={setTargetCategory}
              matchCategories={matchCategories}
              onToggleMatchCategory={toggleMatchCategory}
              maxRecommendations={maxRecommendations}
              setMaxRecommendations={setMaxRecommendations}
              onGenerate={handleGenerate}
              loading={loadingRecommendations}
            />
          )}

          <div className="mt-4 space-y-4">
            {error && (
              <div className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                {error}
              </div>
            )}

            {!uploadedFile && !error && (
              <div className="rounded-2xl border border-black/5 bg-white/70 px-4 py-3 text-sm text-zinc-500 shadow-sm backdrop-blur">
                Upload an item to get started.
              </div>
            )}

            {uploadedFile &&
              recommendations &&
              recommendations.length === 0 &&
              !loadingRecommendations && (
                <div className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
                  No matches found. Try different categories.
                </div>
              )}
          </div>

          {recommendations && recommendations.length > 0 && (
            <RecommendationsGrid recommendations={recommendations} />
          )}
        </main>
      </div>
    </div>
  );
}