import { useEffect, useMemo, useState } from "react";
import HeroSection from "./components/HeroSection";
import UploadSection from "./components/UploadSection";
import SettingsPanel from "./components/SettingsPanel";
import RecommendationsGrid from "./components/RecommendationsGrid";
import OutfitView from "./components/OutfitView";
import { fetchCategories, generateRecommendations } from "./api";

const DEFAULT_FETCH_COUNT = 20;

function groupRecommendationsByCategory(recommendations) {
  return recommendations.reduce((acc, item) => {
    const category = item.category;
    if (!acc[category]) acc[category] = [];
    acc[category].push(item);
    return acc;
  }, {});
}

function interleaveRecommendations(grouped, selectedCategories) {
  const categoryQueues = selectedCategories
    .filter((category) => grouped[category]?.length)
    .map((category) => ({
      category,
      items: [...grouped[category]],
    }));

  const result = [];

  let added = true;
  while (added) {
    added = false;
    for (const queue of categoryQueues) {
      if (queue.items.length > 0) {
        result.push(queue.items.shift());
        added = true;
      }
    }
  }

  return result;
}

export default function App() {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [categories, setCategories] = useState([]);
  const [targetCategory, setTargetCategory] = useState("");
  const [matchCategories, setMatchCategories] = useState([]);
  const [recommendations, setRecommendations] = useState(null);
  const [loadingCategories, setLoadingCategories] = useState(true);
  const [loadingRecommendations, setLoadingRecommendations] = useState(false);
  const [error, setError] = useState("");
  const [viewMode, setViewMode] = useState("grid");

  useEffect(() => {
    async function loadCategories() {
      try {
        setLoadingCategories(true);
        const data = await fetchCategories();
        const cats = data.categories || [];
        setCategories(cats);

        setTargetCategory("");
        setMatchCategories([]);
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
    setViewMode("grid");
    setTargetCategory("");
    setMatchCategories([]);
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
        maxRecommendations: DEFAULT_FETCH_COUNT,
      });

      setRecommendations(data.recommendations);
      setViewMode("grid");
    } catch (err) {
      setError(err.message || "Failed to generate recommendations");
    } finally {
      setLoadingRecommendations(false);
    }
  };

  const groupedRecommendations = useMemo(() => {
    if (!recommendations) return {};
    return groupRecommendationsByCategory(recommendations);
  }, [recommendations]);

  const balancedRecommendations = useMemo(() => {
    if (!recommendations) return [];
    return interleaveRecommendations(groupedRecommendations, matchCategories);
  }, [groupedRecommendations, matchCategories, recommendations]);

  const hasResults = recommendations && recommendations.length > 0;

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

          {hasResults && (
            <div className="mt-8 flex flex-wrap items-center justify-between gap-4">
              <div>
                <h2 className="text-2xl font-semibold text-zinc-900">
                  Your Recommendations
                </h2>
                <p className="mt-1 text-sm text-zinc-500">
                  Browse balanced results or preview them together in an outfit layout.
                </p>
              </div>

              <div className="inline-flex rounded-2xl border border-black/10 bg-white p-1 shadow-sm">
                <button
                  onClick={() => setViewMode("grid")}
                  className={`rounded-xl px-4 py-2 text-sm font-medium transition ${
                    viewMode === "grid"
                      ? "bg-zinc-900 text-white"
                      : "text-zinc-600 hover:text-zinc-900"
                  }`}
                >
                  Grid View
                </button>
                <button
                  onClick={() => setViewMode("outfit")}
                  className={`rounded-xl px-4 py-2 text-sm font-medium transition ${
                    viewMode === "outfit"
                      ? "bg-zinc-900 text-white"
                      : "text-zinc-600 hover:text-zinc-900"
                  }`}
                >
                  Outfit View
                </button>
              </div>
            </div>
          )}

          {hasResults && viewMode === "grid" && (
            <RecommendationsGrid recommendations={balancedRecommendations} />
          )}

          {hasResults && viewMode === "outfit" && (
            <OutfitView
              uploadedImage={previewUrl}
              targetCategory={targetCategory}
              groupedRecommendations={groupedRecommendations}
              matchCategories={matchCategories}
            />
          )}
        </main>
      </div>
    </div>
  );
}
