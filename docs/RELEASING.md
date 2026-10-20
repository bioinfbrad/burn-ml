# Publishing a release with trained models

The trained networks are not stored in git. They are published as a file attached to a GitHub release, and archived on Zenodo, so that `scripts/download_models.py` can fetch and verify them.

1. **Export** the archival ensembles (trained by `slurm/s07_final.sh` or `run_local.sh`) on the machine where they were trained. No GPU or TensorFlow is needed:

   ```sh
   python scripts/export_models.py --results results --cv-results run1/results --version 2.0.0
   ```

   This writes `dist/burn-ml-models-v2.0.0.zip` and prints its SHA-256. The zip contains the networks, a model card per task, `MODEL_CARD.md` and checksums. It contains no images, file names, patient groups or machine paths.

2. **Check** the model card (`unzip -p dist/burn-ml-models-v2.0.0.zip models/MODEL_CARD.md`), then test the bundle:

   ```sh
   python scripts/download_models.py --url file://$PWD/dist/burn-ml-models-v2.0.0.zip --dest /tmp/check
   python scripts/predict.py --task detection --models-dir /tmp/check/models some_test_crops/
   ```

3. **Release on GitHub**: Releases → Draft a new release → tag `v2.0.0` → attach `burn-ml-models-v2.0.0.zip` → paste the SHA-256 into the release notes → Publish.

4. **Archive on Zenodo**: if the GitHub integration is enabled, the release's source code is archived automatically. Upload the model zip to Zenodo as well (in the same record or a linked one), so that the weights have a permanent DOI.

5. **Record the DOIs** in the README, `CITATION.cff` and the paper's availability statement.
