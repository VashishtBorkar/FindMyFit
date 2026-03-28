export default function UploadSection({ onUpload, previewUrl }) {
  const handleChange = (e) => {
    const file = e.target.files?.[0];
    if (file) onUpload(file);
  };

  return (
    <section className="rounded-[28px] border border-white/60 bg-white/70 p-6 shadow-[0_10px_30px_rgba(0,0,0,0.06)] backdrop-blur-xl sm:p-8">
      <div>
        <h2 className="text-xl font-semibold text-zinc-900">Upload Your Item</h2>
        <p className="mt-1 text-sm text-zinc-500">
          Choose a clothing image to begin generating recommendations.
        </p>
      </div>

      <label className="mt-6 block cursor-pointer overflow-hidden rounded-3xl border border-dashed border-[#b89b68]/40 bg-gradient-to-b from-[#fffdf9] to-[#f7f1e8] transition duration-200 hover:-translate-y-0.5 hover:shadow-lg">
        <input type="file" accept="image/*" hidden onChange={handleChange} />

        {!previewUrl ? (
          <div className="flex min-h-[320px] flex-col items-center justify-center px-6 py-10 text-center">
            <div className="mb-5 flex h-16 w-16 items-center justify-center rounded-full border border-[#b89b68]/20 bg-white text-2xl text-[#b5935a] shadow-sm">
              ↑
            </div>
            <p className="text-lg font-semibold text-zinc-900">
              Click to upload an image
            </p>
            <p className="mt-2 text-sm text-zinc-500">
              JPG, JPEG, PNG, WEBP supported
            </p>
          </div>
        ) : (
          <div className="flex min-h-[320px] items-center justify-center bg-white p-5">
            <img
              src={previewUrl}
              alt="Uploaded preview"
              className="max-h-[460px] w-auto max-w-full rounded-2xl object-contain"
            />
          </div>
        )}
      </label>
    </section>
  );
}