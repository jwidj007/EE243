# EE243 Final Project — HumanWild

**Authors:** Vihan Jagasia & Jet Widjaja

This project evaluates HumanWild (TPAMI 2025), a synthetic data generation pipeline for 3D Human Pose and Shape estimation. Starting from our assigned paper Diffusion-HPC (3DV 2024), we trace the lineage to HumanWild as the most recent open-source frontier paper. The goal is to empirically explore the pipeline's weaknesses through targeted experiments, demonstrating both success cases where the model performs well and failure cases where it breaks down.

---

## Setup & Reproduction

Open `HumanWild_Generator.ipynb` in Google Colab (T4 or A100 recommended) and run cells in order.

**Compatibility patches (run before inference):**
```python
!sed -i 's/from transformers import DPTFeatureExtractor, DPTForDepthEstimation/from transformers import AutoImageProcessor as DPTFeatureExtractor, DPTForDepthEstimation/' /content/WildHuman/demo.py
!sed -i 's/from PIL import Image$/from PIL import Image\nfrom PIL.Image import Resampling/' /content/WildHuman/demo.py
!pip install -q compel
```

**Model weights (~83GB, downloaded automatically by notebook):**
- SDXL base: `stabilityai/stable-diffusion-xl-base-1.0`
- VAE: `madebyollin/sdxl-vae-fp16-fix`
- HumanWild ControlNet: `geyongtao/HumanWild`

---

## Repository Structure

```
EE243/
├── index.html                  # Project webpage
├── README.md                   
├── HumanWild_Generator.ipynb   # Inference notebook
└── images/
    ├── success1.jpg            # Football kick — pose followed correctly
    ├── success2.jpg            # Contortionist clean output (Run 1)
    ├── fail1.jpg               # Surgeon — blobby gloved hands (hand noise)
    ├── fail2.jpg               # Contortionist — missing head (Run 2)
    ├── fail3.jpg               # Contortionist — mismatched clothing (Run 3)
    ├── fail4.jpg               # Ballet — phantom duplicate leg
    └── fail5.jpg               # Sprinter — phantom limb at waist
```

---

## References

- Ge, Y. et al. "3D Human Reconstruction in the Wild with Synthetic Data Using Generative Models." TPAMI 2025.
- Weng, Z. et al. "Diffusion-HPC: Generating Synthetic Images with Realistic Humans." 3DV 2024.
