# Visual Inspection Notes and Merchandiser Review Sheet

Because this project was completed individually, I inspected the seven output comparison figures (`pair1_w2_outputs.png` to `pair7_w8_outputs.png`) directly across three practical checks:
1. **Visual Clarity (1-5):** How clean are edges, textures, and surface shading compared to the target?
2. **Silhouette Preservation (1-5):** Does the output keep the original garment or footwear outline without warping?
3. **Failure Predictability (1-5):** When the model makes an error under noise or unseen views, is the failure easy to spot and filter?

## Summary of My Visual Inspection Across Pairs 1-7
- **Pair 1 (W2):** Both Prompt-to-Prompt and StyleAligned preserve garment outlines well on ankle boots, pullovers, and trousers; StyleAligned gives slightly more even contrast across items.
- **Pair 2 (W3):** DreamBooth produces noticeably darker, sharper interior contrast on the pullover and trousers than Custom Diffusion (+0.62 dB PSNR), though both recover the silhouette cleanly.
- **Pair 3 (W4):** Both 3DGS and Mini-NeRF reconstruct the trained left half of the sphere smoothly, and both blur on the held-out right half (`x > 0.15`), with Mini-NeRF blending the unseen boundary slightly more smoothly.
- **Pair 4 (W5):** DreamFields produces a coarse, blocky boot massing from the 4x4 pooled semantic prior, while DreamFusion SDS recovers the ankle boot contour cleanly because its score network receives the 3x3-blurred view map.
- **Pair 5 (W6):** Text2Video-Zero keeps the boot crisp across frames by warping Frame 0 horizontally, whereas AnimateDiff tracks the subtle vertical sway better.
- **Pair 6 (W7):** Tune-A-Video renders sharper horizontal stripes inside the garment silhouette than TokenFlow after 25 tuning steps.
- **Pair 7 (W8):** The policy classifier blocks all 10 adversarial store tickets, while the code agent solves the 2 join tasks but lets 8/10 adversarial tickets through.
