# COMP8410 Assignment 1: AI Systems Evaluation (60801131)

- Student Name: S M Zain
- Student ID: 60801131
- Repository: https://github.com/smzainbd/COMP8410_Assignment1

## Overview and Use Case
This folder contains the evaluation code, test data loaders, benchmark logs, and output figures for IconicStyle ANZ across seven weekly AI model pairs (Weeks 2 to 8, 14 models in total). Each pair compares a fast, lightweight model (System 1) against a heavier, higher-fidelity model (System 2) against a 3.20 ms batch latency SLA on an NVIDIA Tesla T4 GPU.

## Folder Contents
- configs/sla_config.json: SLA latency target (3.20 ms), T4 GPU power (70W), and cloud cost rate ($0.35/hr).
- src/profiler.py: Shared Fashion-MNIST loader (seed 42, 2048 train / 512 test), MSE/drift/fairness calculators, and 22-column CSV logger.
- src/w2_editing.py: Pair 1 (Week 2) Prompt-to-Prompt attention injection vs. StyleAligned shared K/V attention.
- src/w3_personalization.py: Pair 2 (Week 3) Custom Diffusion (tuning W_K, W_V only, 8.91% params) vs. DreamBooth full U-Net + CFG.
- src/w4_recon3d.py: Pair 3 (Week 4) 3D Gaussian Splatting (256 splats) vs. Mini-NeRF (16-sample ray MLP) adapted from the Week 4 practical notebooks.
- src/w5_text3d.py: Pair 4 (Week 5) DreamFields semantic prior alignment vs. DreamFusion Score Distillation Sampling (SDS).
- src/w6_video_gen.py: Pair 5 (Week 6) Text2Video-Zero zero-shot latent warp vs. AnimateDiff 1D temporal attention.
- src/w7_video_edit.py: Pair 6 (Week 7) TokenFlow keyframe token matching vs. Tune-A-Video sparse-causal tuning.
- src/w8_agents.py: Pair 7 (Week 8) Verifiable Policy Agent vs. Code-Actuation Agent on 20 store support tickets, plus RubricJudgeMLP bias check.
- 60801131_Evaluation_Notebook.ipynb: Main notebook that runs all 7 pairs and saves the outputs.
- results/metrics/metrics.csv: Saved 14-row benchmark table produced by the notebook.
- results/metrics/user_feedback_protocol.md: Visual inspection and qualitative review notes across Pairs 1 to 7.
- results/figures/: Saved visual comparison grids (pair1_w2_outputs.png to pair7_w8_outputs.png).
- requirements.txt: Required Python packages.

## How to Replicate the Evaluation
1. Install the required packages (Python 3.10+ with PyTorch, torchvision, numpy, pandas, matplotlib):
   pip install -r requirements.txt

2. Open 60801131_Evaluation_Notebook.ipynb in Google Colab (Tesla T4 GPU runtime) or local Jupyter from inside the 60801131_Evaluation folder.

3. Click Runtime -> Restart and run all to execute all 20 cells in order. Cell numbers below are the notebook order, starting at 1.
   - Cell 1: Mount Google Drive.
   - Cell 2: Check the date, Python version, GPU (nvidia-smi), and PyTorch.
   - Cell 3: Heading only ("Fix path Accordingly").
   - Cell 4: Set the working folder to 60801131_Evaluation. Fashion-MNIST downloads automatically into data/ on the first run.
   - Cell 5: Heading for Pair 1. Cell 6 runs src/w2_editing.py on 512 held-out test garments, logs System 1 (Prompt-to-Prompt) and System 2 (StyleAligned) to results/metrics/metrics.csv, and saves results/figures/pair1_w2_outputs.png.
   - Cell 7: Heading for Pair 2. Cell 8 runs src/w3_personalization.py comparing Custom Diffusion (5,504 trainable params) against DreamBooth (61,761 trainable params), logs rows 3-4, and saves results/figures/pair2_w3_outputs.png.
   - Cell 9: Heading for Pair 3. Cell 10 runs src/w4_recon3d.py on the 48x48 shaded sphere (trained on x < -0.15, tested on x > 0.15), logs 3DGS vs. Mini-NeRF to rows 5-6, and saves results/figures/pair3_w4_outputs.png.
   - Cell 11: Heading for Pair 4. Cell 12 runs src/w5_text3d.py on held-out Class 9 ankle boot views, logs DreamFields vs. DreamFusion SDS to rows 7-8, and saves results/figures/pair4_w5_outputs.png.
   - Cell 13: Heading for Pair 5. Cell 14 runs src/w6_video_gen.py on 8-frame moving garment clips, logs Text2Video-Zero vs. AnimateDiff to rows 9-10, and saves results/figures/pair5_w6_outputs.png.
   - Cell 15: Heading for Pair 6. Cell 16 runs src/w7_video_edit.py on 8-frame stripe restyling clips, logs TokenFlow vs. Tune-A-Video to rows 11-12, and saves results/figures/pair6_w7_outputs.png.
   - Cell 17: Heading for Pair 7. Cell 18 runs src/w8_agents.py on the 20 held-out store tickets and RubricJudgeMLP kappa check, logs rows 13-14, and saves results/figures/pair7_w8_outputs.png.
   - Cell 19: Heading for the visual review. Cell 20 prints the qualitative visual inspection notes from results/metrics/user_feedback_protocol.md.
