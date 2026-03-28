export default function SettingsPanel({
  categories,
  targetCategory,
  setTargetCategory,
  matchCategories,
  onToggleMatchCategory,
  maxRecommendations,
  setMaxRecommendations,
  onGenerate,
  loading,
}) {
  return (
    <section className="mt-7 rounded-[28px] border border-white/60 bg-white/70 p-6 shadow-[0_10px_30px_rgba(0,0,0,0.06)] backdrop-blur-xl sm:p-8">
      <div>
        <h2 className="text-xl font-semibold text-zinc-900">
          Customize Recommendations
        </h2>
        <p className="mt-1 text-sm text-zinc-500">
          Fine-tune the matching criteria before generating results.
        </p>
      </div>

      <div className="mt-6 grid gap-5">
        <div className="rounded-3xl border border-black/5 bg-white/80 p-5">
          <label className="mb-3 block text-sm font-semibold text-zinc-800">
            Item Category
          </label>
          <select
            value={targetCategory}
            onChange={(e) => setTargetCategory(e.target.value)}
            className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm text-zinc-900 outline-none transition focus:border-[#b5935a]/60"
          >
            {categories.map((category) => (
              <option key={category} value={category}>
                {category}
              </option>
            ))}
          </select>
        </div>

        <div className="rounded-3xl border border-black/5 bg-white/80 p-5">
          <label className="mb-3 block text-sm font-semibold text-zinc-800">
            Match With
          </label>
          <div className="flex flex-wrap gap-3">
            {categories.map((category) => {
              const active = matchCategories.includes(category);

              return (
                <button
                  key={category}
                  type="button"
                  onClick={() => onToggleMatchCategory(category)}
                  className={`rounded-full px-4 py-2 text-sm font-medium transition ${
                    active
                      ? "bg-gradient-to-b from-[#b5935a] to-[#947445] text-white shadow-md"
                      : "border border-black/10 bg-white text-zinc-800 hover:border-[#b5935a]/40"
                  }`}
                >
                  {category}
                </button>
              );
            })}
          </div>
        </div>

        <div className="rounded-3xl border border-black/5 bg-white/80 p-5">
          <label className="mb-3 block text-sm font-semibold text-zinc-800">
            Number of Results: {maxRecommendations}
          </label>
          <input
            type="range"
            min="1"
            max="20"
            value={maxRecommendations}
            onChange={(e) => setMaxRecommendations(Number(e.target.value))}
            className="h-2 w-full cursor-pointer appearance-none rounded-lg bg-[#eadfcf]"
          />
        </div>
      </div>

      <button
        onClick={onGenerate}
        disabled={loading}
        className="mt-6 inline-flex w-full items-center justify-center rounded-2xl bg-zinc-950 px-5 py-4 text-sm font-semibold text-white shadow-lg transition hover:-translate-y-0.5 hover:bg-zinc-900 disabled:cursor-not-allowed disabled:opacity-70"
      >
        {loading ? "Finding Matches..." : "Generate Recommendations"}
      </button>
    </section>
  );
}