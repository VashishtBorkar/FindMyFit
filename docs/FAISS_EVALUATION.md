# Exact FAISS Evaluation

This report is generated from the real local catalog by the benchmark CLI. FAISS uses exact flat indexes, so the expected result is identical ranking with lower retrieval latency—not an approximate-recall tradeoff.

## Environment

- Generated: 2026-07-29T06:42:56.643557+00:00
- Python: 3.14.3
- FAISS: 1.14.3
- CPU threads: 12
- Database size: 530.0 MiB
- FAISS index size: 265.3 MiB
- Combined index build time: 84.91 s

## Retrieval benchmark

| Model | Workload | Size | SQLite mean | SQLite p50 | SQLite p95 | FAISS mean | FAISS p50 | FAISS p95 | Speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clip/vit-b32 | all | all | 100.613 ms | 85.599 ms | 243.517 ms | 19.543 ms | 19.187 ms | 32.312 ms | 4.46x |
| clip/vit-b32 | one_category | large | 133.400 ms | 146.433 ms | 169.894 ms | 17.013 ms | 16.114 ms | 23.580 ms | 9.09x |
| clip/vit-b32 | one_category | medium | 57.863 ms | 58.375 ms | 83.725 ms | 14.196 ms | 12.291 ms | 19.837 ms | 4.75x |
| clip/vit-b32 | one_category | small | 9.242 ms | 7.421 ms | 24.542 ms | 10.833 ms | 10.575 ms | 11.658 ms | 0.70x |
| clip/vit-b32 | three_categories | large | 171.998 ms | 167.592 ms | 266.139 ms | 26.967 ms | 26.163 ms | 41.718 ms | 6.41x |
| clip/vit-b32 | three_categories | medium | 49.989 ms | 45.071 ms | 67.055 ms | 20.550 ms | 19.412 ms | 29.625 ms | 2.32x |
| findmyfit/v1 | all | all | 113.548 ms | 98.035 ms | 263.493 ms | 11.987 ms | 10.485 ms | 19.070 ms | 9.35x |
| findmyfit/v1 | one_category | large | 146.062 ms | 159.560 ms | 192.806 ms | 9.017 ms | 8.212 ms | 10.929 ms | 19.43x |
| findmyfit/v1 | one_category | medium | 69.021 ms | 68.942 ms | 97.196 ms | 11.774 ms | 10.692 ms | 17.906 ms | 6.45x |
| findmyfit/v1 | one_category | small | 11.183 ms | 9.326 ms | 29.381 ms | 10.859 ms | 10.581 ms | 11.744 ms | 0.88x |
| findmyfit/v1 | three_categories | large | 191.422 ms | 189.252 ms | 299.494 ms | 12.901 ms | 13.863 ms | 19.102 ms | 13.65x |
| findmyfit/v1 | three_categories | medium | 61.966 ms | 53.566 ms | 85.348 ms | 14.136 ms | 9.631 ms | 28.970 ms | 5.56x |

Initialization and throughput:

| Model | SQLite init | FAISS init | SQLite QPS | FAISS QPS |
|---|---:|---:|---:|---:|
| clip/vit-b32 | 6.612 s | 1.373 s | 9.94 | 51.17 |
| findmyfit/v1 | 5.010 s | 1.210 s | 8.81 | 83.42 |

Item ordering was identical and scores matched within `1e-6`.

## API benchmark

| Backend | Startup | Mean | p50 | p95 |
|---|---:|---:|---:|---:|
| SQLite | 26.981 s | 202.083 ms | 197.018 ms | 279.102 ms |
| FAISS | 7.910 s | 134.370 ms | 128.076 ms | 162.358 ms |

End-to-end API p50 speedup: **1.54x**. This includes upload validation, decoding, CLIP inference, optional metric projection, retrieval, and serialization, so the gain can be smaller than retrieval-only speedup.
