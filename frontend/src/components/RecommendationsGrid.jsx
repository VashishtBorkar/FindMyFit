import RecommendationCard from "./RecommendationCard";

export default function RecommendationsGrid({ recommendations }) {
  return (
    <section className="mt-10">
      <div>
        <h2 className="text-2xl font-semibold text-zinc-900">Recommendations</h2>
        <p className="mt-1 text-sm text-zinc-500">
          Curated matches ranked by compatibility score.
        </p>
      </div>

      <div className="mt-6 columns-1 gap-5 sm:columns-2 lg:columns-3">
        {recommendations.map((item) => (
          <div key={item.id} className="mb-5 break-inside-avoid">
            <RecommendationCard item={item} />
          </div>
        ))}
      </div>
    </section>
  );
}