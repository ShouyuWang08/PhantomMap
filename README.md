# PhantomMap

**COMP 646 Spring 2026 — Final Project**
*PhantomMap: A First-Party Atlas of Where Vision–Language Models Place Objects That Don't Exist.*

---

## What this repository contains

| Path | Purpose |
|---|---|
| `src/` | All our analysis code (inference driver, bbox parsing, features, detector, atlas, figures, stretch HITs cross-reference) |
| `data/` | POPE, AMBER, COCO val2014 images (fetched by `download_data.py`; not checked in) |
| `results/` | Raw VLM outputs (`.jsonl`) + detector metrics + atlas stats |
| `report/` | CVPR-template LaTeX source + bibliography + figures |
| `notebooks/` | Exploratory notebooks only — final figures are reproduced by `src/make_figures.py` |
| `requirements.txt` | Pinned Python dependencies |

## Reproducing our results (free Colab T4)

```bash
# 1. Environment
pip install -r requirements.txt

# 2. Data (POPE jsonl + AMBER jsonl + COCO val2014 images referenced
#    by those benchmarks; ~1.5 GB download).
python src/download_data.py --out data

# 3. Run Qwen2.5-VL-7B across all POPE splits (~80 min each on T4, bf16).
python src/run_vlm.py --model qwen  --split pope_random      --out results/qwen_pope_random.jsonl
python src/run_vlm.py --model qwen  --split pope_popular     --out results/qwen_pope_popular.jsonl
python src/run_vlm.py --model qwen  --split pope_adversarial --out results/qwen_pope_adversarial.jsonl
python src/run_vlm.py --model qwen  --split amber            --out results/qwen_amber.jsonl

# 4. Repeat for LLaVA-NeXT-Mistral-7B.
python src/run_vlm.py --model llava --split pope_random      --out results/llava_pope_random.jsonl
# ... etc.

# 5. Atlas.
python src/atlas.py \
  --inputs results/*.jsonl \
  --out-fig report/figures/fig3_atlas.pdf \
  --out-stats results/atlas_stats.json

# 6. Detector (Qwen, then LLaVA).
python src/detector.py \
  --inputs results/qwen_*.jsonl \
  --model-filter "Qwen/Qwen2.5-VL-7B-Instruct" \
  --out results/detector_qwen.json

python src/detector.py \
  --inputs results/llava_*.jsonl \
  --model-filter "llava-hf/llava-v1.6-mistral-7b-hf" \
  --out results/detector_llava.json

# 7. All figures.
python src/make_figures.py \
  --predictions results/*.jsonl \
  --detector-metrics results/detector_qwen.json results/detector_llava.json

# 8. (Stretch) HITs cross-reference on Qwen.
python src/hits_cross_ref.py \
  --predictions results/qwen_pope_adversarial.jsonl \
  --out results/hits_crossref.jsonl --limit 200

# 9. Build the PDF.
cd report
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
```

Runtime budget on a free Colab T4 (bf16):

| Step | Wall time |
|---|---|
| POPE (all 3 splits) × Qwen2.5-VL-7B | ~4 h |
| POPE (all 3 splits) × LLaVA-NeXT-7B | ~4 h |
| AMBER × 2 models | ~2 h |
| Atlas + detector + figures | ~3 min |
| Stretch HITs cross-reference | ~40 min |

## What is our code vs. third-party

Per the syllabus academic-integrity clause, here is the explicit split:

### Ours (original)
- `src/prompts.py`, `src/parse_bbox.py`, `src/run_vlm.py`, `src/features.py`,
  `src/detector.py`, `src/atlas.py`, `src/make_figures.py`,
  `src/hits_cross_ref.py`, `src/metrics.py`, `src/download_data.py` —
  every line written by us for this project.
- `report/report.tex`, `report/egbib.bib`, the full analysis, the figure
  design, and every number reported.

### Third-party, reused verbatim (with citation)
- `Qwen/Qwen2.5-VL-7B-Instruct` model weights and `qwen-vl-utils` helper ([arXiv:2502.13923](https://arxiv.org/abs/2502.13923)).
- `llava-hf/llava-v1.6-mistral-7b-hf` model weights ([LLaVA-NeXT blog post](https://llava-vl.github.io/blog/2024-01-30-llava-next/)).
- POPE benchmark JSONL files from [RUCAIBox/POPE](https://github.com/RUCAIBox/POPE).
- AMBER discriminative queries from [junyangwang0410/AMBER](https://github.com/junyangwang0410/AMBER).
- COCO val2014 images from the official COCO release ([arXiv:1405.0312](https://arxiv.org/abs/1405.0312)).
- HuggingFace `transformers`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn` as libraries (standard dependencies, not copied into the repo).

## Team

- **Shouyu Wang** — evaluation code, VLM runs, detector, report §4–§5.
- **Limeng Wang** — atlas visualisations, figures 1/3/5, related work, report §2–§3.

## Contact

- Shouyu Wang: `sw188@rice.edu`
- Limeng Wang: `lw158@rice.edu`

## License

Academic use only. Third-party components retain their original licenses.
