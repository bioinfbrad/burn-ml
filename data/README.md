# Data

The image dataset is **not** included in this repository. It consists of clinical photographs of burns and is held in a restricted-access record on Zenodo (doi:10.5281/zenodo.20052466). Access is available from the corresponding author on reasonable request, subject to the conditions of the original patient consent.

## Expected layout

The pipeline reads cropped images from one folder per class:

```
<raw-dir>/
    Full_thickness_burn/
    Partial_thickness_burn/
    Normal_skin/
```

Each file name must contain the accession number of the photograph the crop was taken from (a run of seven digits). `scripts/01_prepare_manifest.py` uses it to assign crops to patient episodes: by default, photographs whose accession numbers differ by 50 or less are grouped together. If you have true anonymised patient identifiers, supply them instead, because grouping by accession number is an approximation.

## Never commit data

Do not add images, crops, manifests, predictions or reports to this repository: file names and patient group names derived from accession numbers are identifiers. The `.gitignore` blocks the common cases, but it is not a substitute for care.
