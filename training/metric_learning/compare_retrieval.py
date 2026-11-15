"""Train compatibility models on fixed Polyvore complementary-retrieval examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import torch
from sqlalchemy.engine import make_url
from torch.nn.functional import normalize
from torch.utils.data import DataLoader

from findmyfit.config import Settings
from findmyfit.db.embedding_repository import SqliteEmbeddingRepository
from findmyfit.db.session import create_session_factory
from findmyfit.models.metric import FashionCompatibilityModel
from findmyfit.models.two_tower import TwoTowerCompatibilityModel
from findmyfit.models.category_heads import CategoryHeadCompatibilityModel
from findmyfit.storage.local_images import LocalImageStore
from training.metric_learning.data import load_embeddings
from training.metric_learning.loss import CandidateContrastiveLoss, CandidateRankingLoss
from training.metric_learning.ranking_data import RankingDataset, make_ranking_splits


def score_candidates(model, anchors, categories, candidates, metric="cosine"):
    if isinstance(model, (TwoTowerCompatibilityModel, CategoryHeadCompatibilityModel)):
        return model(anchors, categories, candidates)
    if model is None:
        if metric == "l2":
            return -(candidates - anchors.unsqueeze(1)).square().sum(dim=-1)
        return torch.bmm(
            normalize(candidates, dim=-1), normalize(anchors, dim=-1).unsqueeze(-1)
        ).squeeze(-1)
    queries = model(anchors)
    items = model(candidates.flatten(0, 1)).reshape(
        candidates.size(0), candidates.size(1), -1
    )
    return torch.bmm(items, queries.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def evaluate(model, loader, device, metric="cosine"):
    if model is not None:
        model.eval()
    totals = {}
    for batch in loader:
        anchors, categories, candidates, positives = (value.to(device) for value in batch)
        scores = score_candidates(model, anchors, categories, candidates, metric=metric)
        if not torch.isfinite(scores).all():
            raise RuntimeError("Evaluation scores are not finite")
        ordered = scores.argsort(dim=1, descending=True, stable=True)
        ranks = (ordered == positives.unsqueeze(1)).nonzero()[:, 1] + 1
        for category in categories.unique():
            selected = ranks[categories == category]
            key = str(category.item())
            stats = totals.setdefault(key, [0, 0, 0, 0.0])
            stats[0] += selected.numel()
            stats[1] += (selected <= 5).sum().item()
            stats[2] += (selected <= 10).sum().item()
            stats[3] += selected.float().reciprocal().sum().item()
    def summarize(stats):
        count, hits5, hits10, reciprocal_rank = stats
        return {
            "queries": count, "recall_at_5": hits5 / count,
            "recall_at_10": hits10 / count, "mrr": reciprocal_rank / count,
        }
    overall = [sum(stats[i] for stats in totals.values()) for i in range(4)]
    return {**summarize(overall), "by_category_id": {
        key: summarize(stats) for key, stats in totals.items()
    }}


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def check_training_batch(model, dataset, device, criterion=None):
    """Check real inputs and gradients before starting an experiment."""
    loader = DataLoader(dataset, batch_size=min(8, len(dataset)))
    batch = (value.to(device) for value in next(iter(loader)))
    anchors, categories, candidates, positives = batch
    model.to(device).train()
    criterion = criterion if criterion is not None else CandidateRankingLoss()
    loss = criterion(score_candidates(model, anchors, categories, candidates), positives)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    if not math.isfinite(loss.item()) or any(
        gradient is None or not torch.isfinite(gradient).all() for gradient in gradients
    ):
        raise RuntimeError("Real-data batch failed loss/gradient validation")
    model.zero_grad(set_to_none=True)
    return {"batch_size": positives.numel(), "loss": loss.item(), "finite_gradients": True}


def train(model, datasets, args, device, checkpoint, metadata):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = (
        CandidateContrastiveLoss(args.margin)
        if metadata["objective"] == "contrastive" else CandidateRankingLoss(getattr(args, "temperature", 0.1))
    )
    loaders = {
        split: DataLoader(
            dataset, batch_size=args.batch_size, shuffle=split == "train",
            generator=torch.Generator().manual_seed(args.training_seed),
        )
        for split, dataset in datasets.items()
    }
    history = []
    best_mrr = -1.0
    best_epoch = None
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        if hasattr(datasets["train"], "set_epoch"):
            datasets["train"].set_epoch(epoch)
        model.train()
        loss_sum = 0.0
        for batch in loaders["train"]:
            anchors, categories, candidates, positives = (value.to(device) for value in batch)
            optimizer.zero_grad()
            loss = criterion(score_candidates(model, anchors, categories, candidates), positives)
            if not math.isfinite(loss.item()):
                raise RuntimeError("Training loss is not finite")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * positives.numel()
        validation = evaluate(model, loaders["validation"], device)
        history.append({
            "epoch": epoch, "training_loss": loss_sum / len(datasets["train"]),
            "validation": validation, "seconds": time.perf_counter() - started,
        })
        if validation["mrr"] > best_mrr:
            best_mrr = validation["mrr"]
            best_epoch = epoch
            stale_epochs = 0
            temporary = checkpoint.with_suffix(".tmp")
            torch.save({
                **metadata, "best_epoch": epoch,
                "model_state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
            }, temporary)
            temporary.replace(checkpoint)
        else:
            stale_epochs += 1
        write_json(checkpoint.with_suffix(".history.json"), history)
        print(f"{type(model).__name__} epoch {epoch}: {history[-1]}", flush=True)
        if args.patience and stale_epochs >= args.patience:
            print(f"Early stopping after {stale_epochs} epochs without validation improvement.", flush=True)
            break
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state_dict"])
    return {
        "history": history, "best_epoch": best_epoch,
        "objective": metadata["objective"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "stopped_early": len(history) < args.epochs,
        "checkpoint_reloaded": True,
        "test": None if args.defer_test else evaluate(model, loaders["test"], device),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder-name", help="Registered source embedding model name")
    parser.add_argument("--encoder-version", help="Registered source embedding model version")
    parser.add_argument("--reference-candidates", type=Path, help="Require generated candidates to match this frozen manifest")
    parser.add_argument("--outfit-dir", type=Path, default=Path("data"))
    parser.add_argument("--models", nargs="+",
                        choices=["shared_contrastive", "shared_projector", "two_tower",
                                 "contrastive_wide", "contrastive_deep", "contrastive_regularized", "contrastive_wider",
                                 "contrastive_plain_six", "contrastive_residual"],
                        default=["shared_contrastive", "shared_projector", "two_tower"])
    parser.add_argument("--anchor-category", default="top")
    parser.add_argument("--target-categories", nargs="+", default=["pants", "shoes"])
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--defer-test", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=0)
    args = parser.parse_args()
    if args.candidates < 6:
        parser.error("--candidates must be at least 6 for a meaningful Recall@5")
    if min(args.epochs, args.batch_size, args.threads) < 1:
        parser.error("--epochs, --batch-size and --threads must be positive")
    if not 0 < args.learning_rate < float("inf"):
        parser.error("--learning-rate must be positive and finite")
    if not 0 < args.margin < float("inf") or args.patience < 0:
        parser.error("--margin must be positive and finite, and --patience must be nonnegative")
    if args.training_seed is None:
        args.training_seed = args.seed
    if args.output.exists():
        parser.error("Choose a new output directory for each run")
    overrides = {}
    if args.encoder_name:
        overrides["clip_catalog_model_name"] = args.encoder_name
    if args.encoder_version:
        overrides["clip_model_version"] = args.encoder_version
    settings = Settings.from_env(**overrides)
    if not args.outfit_dir.is_absolute():
        args.outfit_dir = settings.project_root / args.outfit_dir
    for filename in ("train_no_dup.json", "valid_no_dup.json", "test_no_dup.json"):
        if not (args.outfit_dir / filename).is_file():
            parser.error(f"Missing official split file: {filename}")
    url = make_url(settings.database_url)
    if url.get_backend_name() == "sqlite" and not Path(url.database or "").is_file():
        parser.error(f"SQLite catalog not found: {url.database}")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    repository = SqliteEmbeddingRepository(
        create_session_factory(settings.database_url), LocalImageStore(settings.images_dir)
    )
    embeddings, categories = load_embeddings(
        repository, model_name=settings.clip_catalog_model_name,
        model_version=settings.clip_model_version,
    )
    if not embeddings:
        parser.error("No encoder vectors are available")
    model_record = repository.get_model(settings.clip_catalog_model_name, settings.clip_model_version)
    examples, audit = make_ranking_splits(
        args.outfit_dir, embeddings, seed=args.seed, candidate_count=args.candidates,
        anchor_category=args.anchor_category, target_categories=args.target_categories,
    )
    if args.reference_candidates is not None:
        reference = json.loads(args.reference_candidates.read_text())
        if json.loads(json.dumps(examples)) != reference:
            parser.error("Generated candidates differ from the frozen reference manifest")
    print(json.dumps(audit, indent=2), flush=True)
    if not all(examples.values()):
        parser.error("A split has no usable examples. Inspect coverage and category filters.")
    category_ids = {category: index for index, category in enumerate(sorted(categories))}
    datasets = {
        split: RankingDataset(rows, embeddings, category_ids) for split, rows in examples.items()
    }
    embedding_dim = next(iter(embeddings.values()))["embedding"].size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=False)
    candidates_path = args.output / "candidates.json"
    write_json(candidates_path, examples)
    write_json(args.output / "audit.json", audit)
    vector_digest = hashlib.sha256()
    for item, record in sorted(embeddings.items()):
        vector_digest.update(item.encode() + b"\0")
        vector_digest.update(record["embedding"].tobytes())
    report = {
        "settings": {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items() if key != "output"},
        "encoder": [settings.clip_catalog_model_name, settings.clip_model_version],
        "encoder_fingerprint": model_record.artifact_fingerprint,
        "vector_sha256": vector_digest.hexdigest(),
        "candidates_sha256": hashlib.sha256(candidates_path.read_bytes()).hexdigest(),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_changes": subprocess.check_output(["git", "diff"], text=True),
        "category_ids": category_ids, "split_audit": audit,
        "torch_version": str(torch.__version__), "device": str(device), "models": {},
        "random_expected": {
            "recall_at_5": 5 / args.candidates,
            "mrr": sum(1 / rank for rank in range(1, args.candidates + 1)) / args.candidates,
        },
    }
    # Include untracked experiment sources in the run provenance.
    source_dir = args.output / "source"
    source_dir.mkdir()
    for source in (
        Path(__file__), Path(__file__).with_name("ranking_data.py"),
        Path(__file__).with_name("loss.py"), Path(__file__).with_name("data.py"),
        settings.project_root / "src/findmyfit/models/metric.py",
        settings.project_root / "src/findmyfit/models/two_tower.py",
    ):
        (source_dir / source.name).write_bytes(source.read_bytes())
    report["cosine_validation"] = evaluate(
        None, DataLoader(datasets["validation"], batch_size=args.batch_size), device
    )
    report["real_batch_checks"] = {}
    for name, model_type in (
        ("shared_contrastive", FashionCompatibilityModel),
        ("shared_projector", FashionCompatibilityModel),
        ("two_tower", TwoTowerCompatibilityModel),
        ("contrastive_wide", FashionCompatibilityModel),
        ("contrastive_deep", FashionCompatibilityModel),
        ("contrastive_regularized", FashionCompatibilityModel),
        ("contrastive_wider", FashionCompatibilityModel),
        ("contrastive_plain_six", FashionCompatibilityModel),
        ("contrastive_residual", FashionCompatibilityModel),
    ):
        if name not in args.models:
            continue
        torch.manual_seed(args.training_seed)
        model_kwargs = {"embedding_dim": embedding_dim}
        if name == "two_tower":
            model_kwargs["num_categories"] = len(category_ids)
        if name == "contrastive_wide":
            model_kwargs["hidden_dims"] = [512]
        if name in ("contrastive_deep", "contrastive_regularized"):
            model_kwargs["hidden_dims"] = [512, 256]
        if name == "contrastive_regularized":
            model_kwargs.update(dropout=0.15, layer_norm=True)
        if name == "contrastive_wider":
            model_kwargs["hidden_dims"] = [1024]
        if name == "contrastive_plain_six":
            model_kwargs["hidden_dims"] = [256] * 5
        if name == "contrastive_residual":
            model_kwargs.update(hidden_dims=[256], residual_blocks=2)
        model = model_type(**model_kwargs)
        objective = "contrastive" if "contrastive" in name else "ranking"
        criterion = CandidateContrastiveLoss(args.margin) if objective == "contrastive" else CandidateRankingLoss()
        report["real_batch_checks"][name] = check_training_batch(
            model, datasets["train"], device, criterion
        )
        if not args.prepare_only and name in args.models:
            torch.manual_seed(args.training_seed)
            model = model_type(**model_kwargs)
            report["models"][name] = train(
                model, datasets, args, device, args.output / f"{name}.pt",
                {"encoder": report["encoder"],
                 "encoder_fingerprint": report["encoder_fingerprint"],
                 "vector_sha256": report["vector_sha256"],
                 "model_kwargs": model_kwargs, "category_ids": category_ids,
                 "hidden_dim": model_kwargs.get("hidden_dims", [256])[0], "output_dim": 128,
                 "objective": objective, "margin": args.margin,
                 "training_seed": args.training_seed,
                 "candidates_sha256": report["candidates_sha256"]},
            )
        write_json(args.output / "metrics.json", report)
    if not args.prepare_only and not args.defer_test:
        report["cosine_test"] = evaluate(
            None, DataLoader(datasets["test"], batch_size=args.batch_size), device
        )
        write_json(args.output / "metrics.json", report)
    print(f"Saved run to {args.output}", flush=True)


if __name__ == "__main__":
    main()
