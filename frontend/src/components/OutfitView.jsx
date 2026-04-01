import { useEffect, useMemo, useState } from "react";

const CATEGORY_SLOT_MAP = {
  hats: "headwear",
  hairwear: "headwear",

  top: "upper",

  outwear: "outerwear",

  pants: "lower",
  skirt: "lower",
  legwear: "lower",

  shoes: "footwear",

  bag: "accessory",
  bracelet: "accessory",
  brooch: "accessory",
  earrings: "accessory",
  eyewear: "accessory",
  gloves: "accessory",
  necklace: "accessory",
  neckwear: "accessory",
  rings: "accessory",
  watches: "accessory",

  dress: "accessory",
  jumpsuit: "accessory",
};

function normalizeCategory(category) {
  return String(category || "").trim().toLowerCase();
}

function getSlotForCategory(category) {
  return CATEGORY_SLOT_MAP[normalizeCategory(category)] || "accessory";
}

function buildSlotCollections({
  targetCategory,
  uploadedImage,
  groupedRecommendations,
  matchCategories,
}) {
  const slots = {
    headwear: { uploaded: null, items: [] },
    upper: { uploaded: null, items: [] },
    outerwear: { uploaded: null, items: [] },
    lower: { uploaded: null, items: [] },
    footwear: { uploaded: null, items: [] },
    accessory: { uploaded: null, items: [] },
  };

  const uploadedSlot = getSlotForCategory(targetCategory);

  if (uploadedImage) {
    slots[uploadedSlot].uploaded = {
      id: "uploaded-item",
      category: targetCategory,
      image: uploadedImage,
      score: null,
      isUploaded: true,
    };
  }

  matchCategories.forEach((category) => {
    const slot = getSlotForCategory(category);
    const items = groupedRecommendations[category] || [];

    items.forEach((item) => {
      slots[slot].items.push({
        ...item,
        originalCategory: category,
      });
    });
  });

  return slots;
}

function ArrowButton({ direction, onClick, disabled }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="flex h-8 w-8 items-center justify-center rounded-full border border-black/10 bg-white text-sm text-zinc-700 shadow-sm transition hover:border-[#b5935a]/40 hover:text-zinc-900 disabled:cursor-not-allowed disabled:opacity-40"
    >
      {direction === "left" ? "←" : "→"}
    </button>
  );
}

function SlotToolbar({
  label,
  score,
  currentIndex,
  total,
  onPrev,
  onNext,
  locked,
}) {
  return (
    <div className="mb-3 flex items-center justify-between gap-3">
      <div>
        <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-zinc-500">
          {label}
        </div>
        {typeof score === "number" && (
          <div className="mt-1 text-xs font-semibold text-[#8d7245]">
            {score}% match
          </div>
        )}
        {locked && (
          <div className="mt-1 text-xs text-zinc-400">Uploaded item</div>
        )}
      </div>

      {!locked && total > 0 && (
        <div className="flex items-center gap-2">
          <ArrowButton direction="left" onClick={onPrev} disabled={total <= 1} />
          <span className="min-w-[44px] text-center text-xs text-zinc-500">
            {currentIndex + 1}/{total}
          </span>
          <ArrowButton direction="right" onClick={onNext} disabled={total <= 1} />
        </div>
      )}
    </div>
  );
}

function OutfitSlotCard({
  label,
  item,
  currentIndex = 0,
  total = 0,
  onPrev,
  onNext,
  locked = false,
  imageClassName = "",
  cardClassName = "",
  imageWrapClassName = "",
}) {
  if (!item?.image) return null;

  return (
    <div
      className={`rounded-[28px] border border-black/5 bg-white/80 p-4 shadow-[0_10px_30px_rgba(0,0,0,0.05)] backdrop-blur ${cardClassName}`}
    >
      <SlotToolbar
        label={label}
        score={item.score}
        currentIndex={currentIndex}
        total={total}
        onPrev={onPrev}
        onNext={onNext}
        locked={locked}
      />

      <div
        className={`flex items-center justify-center overflow-hidden rounded-[24px] bg-gradient-to-b from-[#fffdf9] to-[#f6f0e6] ${imageWrapClassName}`}
      >
        <img
          src={item.image}
          alt={item.category || label}
          className={`max-w-full object-contain ${imageClassName}`}
        />
      </div>
    </div>
  );
}

function AccessoryRail({ title, items }) {
  if (!items.length) return null;

  return (
    <div className="space-y-3">
      <div className="px-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-zinc-500">
        {title}
      </div>

      {items.map((item) => (
        <div
          key={item.id}
          className="rounded-[24px] border border-black/5 bg-white/80 p-3 shadow-[0_8px_24px_rgba(0,0,0,0.05)]"
        >
          <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.16em] text-zinc-500">
            {item.category}
          </div>
          <div className="flex h-[120px] items-center justify-center rounded-2xl bg-[#faf7f1] p-2">
            <img
              src={item.image}
              alt={item.category}
              className="max-h-full max-w-full object-contain"
            />
          </div>
          {typeof item.score === "number" && (
            <div className="mt-2 text-xs font-medium text-[#8d7245]">
              {item.score}% match
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export default function OutfitView({
  uploadedImage,
  targetCategory,
  groupedRecommendations,
  matchCategories,
}) {
  const slots = useMemo(
    () =>
      buildSlotCollections({
        targetCategory,
        uploadedImage,
        groupedRecommendations,
        matchCategories,
      }),
    [uploadedImage, targetCategory, groupedRecommendations, matchCategories]
  );

  const [selectedIndexes, setSelectedIndexes] = useState({
    headwear: 0,
    upper: 0,
    outerwear: 0,
    lower: 0,
    footwear: 0,
  });

  useEffect(() => {
    setSelectedIndexes({
      headwear: 0,
      upper: 0,
      outerwear: 0,
      lower: 0,
      footwear: 0,
    });
  }, [slots]);

  const getItemForSlot = (slotName) => {
    const slot = slots[slotName];
    if (slot.uploaded) return slot.uploaded;
    if (!slot.items.length) return null;
    return slot.items[selectedIndexes[slotName] ?? 0] || slot.items[0];
  };

  const changeSlotItem = (slotName, direction) => {
    if (slots[slotName].uploaded) return;

    const total = slots[slotName].items.length;
    if (total <= 1) return;

    setSelectedIndexes((prev) => {
      const current = prev[slotName] ?? 0;
      const next =
        direction === "next"
          ? (current + 1) % total
          : (current - 1 + total) % total;

      return {
        ...prev,
        [slotName]: next,
      };
    });
  };

  const headwearItem = getItemForSlot("headwear");
  const upperItem = getItemForSlot("upper");
  const outerwearItem = getItemForSlot("outerwear");
  const lowerItem = getItemForSlot("lower");
  const footwearItem = getItemForSlot("footwear");

  const accessoryItems = slots.accessory.items || [];
  const leftAccessories = accessoryItems.filter((_, i) => i % 2 === 0).slice(0, 2);
  const rightAccessories = accessoryItems.filter((_, i) => i % 2 === 1).slice(0, 2);

  return (
     <section className="mt-6 space-y-6">
      {/* <div className="rounded-[28px] border border-black/5 bg-white/70 p-6 shadow-[0_10px_30px_rgba(0,0,0,0.06)] backdrop-blur">
        <h3 className="text-xl font-semibold text-zinc-900">Outfit View</h3>
        <p className="mt-1 max-w-2xl text-sm text-zinc-500">
          Review the full look in a structured outfit layout and swap pieces within
          each category without losing the overall composition.
        </p>
      </div> */}


      <div className="grid gap-6 xl:grid-cols-[220px_minmax(0,720px)_220px] xl:items-start xl:justify-center">
        <div className="order-2 xl:order-1">
          <AccessoryRail title="Accessories" items={leftAccessories} />
        </div>

        <div className="order-1 xl:order-2">
          <div className="rounded-[32px] border border-black/5 bg-white/75 p-4 shadow-[0_10px_30px_rgba(0,0,0,0.06)] backdrop-blur sm:p-6">
            <div className="mx-auto flex max-w-[620px] flex-col gap-4">
              {headwearItem && (
                <div className="mx-auto w-full max-w-[240px]">
                  <OutfitSlotCard
                    label="Headwear"
                    item={headwearItem}
                    currentIndex={selectedIndexes.headwear ?? 0}
                    total={slots.headwear.uploaded ? 1 : slots.headwear.items.length}
                    onPrev={() => changeSlotItem("headwear", "prev")}
                    onNext={() => changeSlotItem("headwear", "next")}
                    locked={Boolean(slots.headwear.uploaded)}
                    imageWrapClassName="min-h-[120px] px-4 py-3"
                    imageClassName="max-h-[100px]"
                  />
                </div>
              )}

              {(upperItem || outerwearItem) && (
                <div className="mx-auto flex w-fit flex-wrap justify-center gap-4">
                    {upperItem && (
                    <div className="w-[280px] shrink-0">
                        <OutfitSlotCard
                        label="Top"
                        item={upperItem}
                        currentIndex={selectedIndexes.upper ?? 0}
                        total={slots.upper.uploaded ? 1 : slots.upper.items.length}
                        onPrev={() => changeSlotItem("upper", "prev")}
                        onNext={() => changeSlotItem("upper", "next")}
                        locked={Boolean(slots.upper.uploaded)}
                        imageWrapClassName="min-h-[240px] px-5 py-4"
                        imageClassName="max-h-[220px]"
                        />
                    </div>
                    )}

                    {outerwearItem && (
                    <div className="w-[280px] shrink-0">
                        <OutfitSlotCard
                        label="Outerwear"
                        item={outerwearItem}
                        currentIndex={selectedIndexes.outerwear ?? 0}
                        total={slots.outerwear.uploaded ? 1 : slots.outerwear.items.length}
                        onPrev={() => changeSlotItem("outerwear", "prev")}
                        onNext={() => changeSlotItem("outerwear", "next")}
                        locked={Boolean(slots.outerwear.uploaded)}
                        imageWrapClassName="min-h-[240px] px-5 py-4"
                        imageClassName="max-h-[220px]"
                        />
                    </div>
                    )}
                </div>
                )}

              {lowerItem && (
                <div className="mx-auto w-full max-w-[300px]">
                  <OutfitSlotCard
                    label="Bottoms"
                    item={lowerItem}
                    currentIndex={selectedIndexes.lower ?? 0}
                    total={slots.lower.uploaded ? 1 : slots.lower.items.length}
                    onPrev={() => changeSlotItem("lower", "prev")}
                    onNext={() => changeSlotItem("lower", "next")}
                    locked={Boolean(slots.lower.uploaded)}
                    imageWrapClassName="min-h-[230px] px-5 py-4"
                    imageClassName="max-h-[600px]"
                  />
                </div>
              )}

              {footwearItem && (
                <div className="mx-auto w-full max-w-[280px]">
                  <OutfitSlotCard
                    label="Shoes"
                    item={footwearItem}
                    currentIndex={selectedIndexes.footwear ?? 0}
                    total={slots.footwear.uploaded ? 1 : slots.footwear.items.length}
                    onPrev={() => changeSlotItem("footwear", "prev")}
                    onNext={() => changeSlotItem("footwear", "next")}
                    locked={Boolean(slots.footwear.uploaded)}
                    imageWrapClassName="min-h-[160px] px-5 py-4"
                    imageClassName="max-h-[120px]"
                  />
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="order-3">
          <AccessoryRail title="More Accessories" items={rightAccessories} />
        </div>
      </div>
    </section>
  );
}