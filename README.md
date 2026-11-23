# Obadah scratch-CNN contribution

Use Python 3.10. Install dependencies with `python -m pip install -r requirements.txt`.
For NVIDIA acceleration install matching CUDA-enabled torch/torchvision builds for your machine (development environment: torch 2.9.1+cu128, torchvision 0.24.1+cu128).
Launch `python -m notebook main.ipynb` from this directory and run cells in order.

Obtain the State Farm dataset through its authorized source. Keep `driver_imgs_list.csv` here and images at `imgs/train/c0` through `imgs/train/c9`. Do not redistribute dataset imagery. Check licensing before publishing failure images.

The notebook reuses six CSV manifests under data/splits; if absent it creates fixed seed-42 splits once. Share these exact manifests for team comparisons. Existing partial manifest sets produce an error. The earlier 21/5 run used all labeled images during development, so the subsequently created test set is not historically untouched; disclose this limitation.

Three runs cover driver-based augmentation ON/OFF and random-image augmentation ON. Defaults: 40 epochs maximum, patience 10, batch size 32. Existing checkpoints are reused. To intentionally start fresh, use a new results directory. Interrupted runs with a checkpoint are also reused; inspect history/config before treating them as complete.

Results include checkpoints, configuration/hardware/runtime, curves, prediction CSVs, class metrics, confusion matrices, ablation comparison and unseen-driver scores. Inspect actual failures and fill in possible_cause in failure_analysis.csv. Write the report interpretation using measured results; do not invent explanations or scores.

Final test is disabled by default. Freeze decisions using validation, train the selected augmentation under both protocols if necessary, then set SELECTED_AUGMENTATION and RUN_FINAL_TEST. A marker prevents repeat test execution. Document failures rather than deleting the marker to retry.

The original notebook/output is preserved in preliminary_21_5.ipynb. Its selected epoch-28 checkpoint achieved 78.23% validation accuracy; it is not a final test result. Its recorded training time was 33.1 minutes. New runs record hardware and elapsed hours in config.json; sum CUDA runs for approximate GPU-hours.

Submission still requires your own one-page statement, report interpretation and presentation. Disclose AI assistance: OpenAI Codex assisted with code restructuring, experiment/evaluation implementation, and requirements review. Review and adapt that statement to reflect all tools actually used. Attribute any other reused code.
