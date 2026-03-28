export default function RecommendationCard({ item }) {
  return (
    <article className="overflow-hidden rounded-[24px] border border-black/5 bg-white shadow-[0_10px_30px_rgba(0,0,0,0.06)] transition duration-200 hover:-translate-y-1 hover:shadow-[0_18px_45px_rgba(0,0,0,0.08)]">
      <div className="bg-white p-4">
        {item.image ? (
          <img
            src={item.image}
            alt={item.category}
            className="h-auto w-full rounded-2xl object-cover"
          />
        ) : (
          <div className="grid min-h-[240px] place-items-center rounded-2xl bg-[#f6f3ed] text-sm text-zinc-500">
            Image unavailable
          </div>
        )}
      </div>

      <div className="px-5 pb-5 pt-1">
        <div className="flex items-center justify-between gap-3">
          <span className="text-sm font-semibold uppercase tracking-[0.14em] text-zinc-900">
            {item.category}
          </span>
          <span className="text-sm font-semibold text-[#8d7245]">
            {item.score}% match
          </span>
        </div>
      </div>
    </article>
  );
}