export default function HeroSection() {
  return (
    <section className="px-4 py-10 text-center sm:px-6 sm:py-14">
      <div className="mb-5 inline-flex items-center rounded-full border border-[#b89b68]/25 bg-white/70 px-4 py-2 text-xs font-medium uppercase tracking-[0.2em] text-[#8d7245] shadow-sm backdrop-blur">
        AI-Powered Fashion Matching
      </div>

      <h1 className="font-serif text-5xl tracking-tight text-zinc-950 sm:text-6xl lg:text-8xl">
        FindMyFit
      </h1>

      <p className="mx-auto mt-5 max-w-2xl text-base leading-7 text-zinc-600 sm:text-lg">
        Upload a clothing item and receive refined, personalized outfit
        recommendations tailored by your recommendation engine.
      </p>
    </section>
  );
}