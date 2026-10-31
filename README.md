# COMP8410 Assignment 1: AI Systems Evaluation
- **Author:** S M Zain (Student ID: 60801131)
- **Repository:** https://github.com/smzainbd/COMP8410_Assignment1

## Structure
- `configs/sla_config.json` — Latency, GPU TDP, and cloud cost constants
- `src/profiler.py` — Data loader, error/drift metrics, and incremental CSV logger
- `src/w2_editing.py` — Pair 1 (Week 2): Prompt-to-Prompt vs. StyleAligned prototypes
- `src/w3_personalization.py` — Pair 2 (Week 3): Custom Diffusion vs. DreamBooth prototypes
- `src/w4_recon3d.py` — Pair 3 (Week 4): 3D Gaussian Splatting vs. NeRF prototypes
- `src/w5_text3d.py` — Pair 4 (Week 5): DreamFields vs. DreamFusion prototypes
- `src/w6_video_gen.py` — Pair 5 (Week 6): Text2Video-Zero vs. AnimateDiff prototypes
- `src/w7_video_edit.py` — Pair 6 (Week 7): TokenFlow vs. Tune-A-Video prototypes
- `src/w8_agents.py` — Pair 7 (Week 8): Verifiable Policy Agent vs. Code-Actuation Agent prototypes
- `60801131_Evaluation_Notebook.ipynb` — Main execution notebook
- `results/metrics/metrics.csv` & `results/figures/` — Generated metrics and plots

## Replication Instructions
1. `pip install -r requirements.txt`
2. Open `60801131_Evaluation_Notebook.ipynb` and click **Run All**.
