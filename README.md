# Distracted Driver Detection

State Farm distracted-driver classification using scratch CNNs, a custom residual CNN, pretrained ResNet18/VGG16, and a two-resolution ResNet18 ensemble.

## Setup

Python 3.10. Install dependencies:

```bash
pip install -r requirements.txt
```

Place images in `imgs/train/c0` through `imgs/train/c9` and keep `driver_imgs_list.csv` in the project root. Model checkpoints use Git LFS.

## Run

Open `main.ipynb` and run the desired experiment cells in order. Results are saved under `results/`.

Driver-based splits use 18 training, 4 validation, and 4 test drivers. Run final test cells only after model selection is complete.
