# NICS Vision Geo: rocks and minerals from photos

Status (2026-11-22): **prototype**. Data pipeline, model, training, calibration and the
answer text are implemented and tested; the first training run is a short cloud trial on
museum photos (results in [`model_cards/nics-vision-geo-micro.md`](../model_cards/nics-vision-geo-micro.md)).
No service, no users, nothing charged.

Built from scratch on 2026-11-22 at the founder's request. The August 2026
`nics-vision-nano` pipeline check (Fashion-MNIST) is not reused: new code
(`src/nics/geo/`), new data, new model name.

## Goal

A person photographs a stone, a beach, the ground around a house or a farm; NICS Vision
Geo names the most likely rock or mineral with an honest probability, shows how to check
it by hand, and says what such a find **may** indicate. Later: video frames, satellite
images, Moon and Mars data.

## What a photo can and cannot tell

| Can (with probability) | Cannot |
|---|---|
| The class of a rock or mineral: quartz, pyrite, granite, malachite... | Say that a place holds gold, or how much |
| Indicator minerals and rocks worth a closer look: sulfides, copper stains, quartz veins, gossan, hot-spring sinter | Tell pyrite from gold, or mica flakes from gold, for certain: that takes a hand test or an assay |
| Which simple checks separate the top guesses (streak, magnet, knife, acid) | Detect oil or gas: they are not visible on a ground photo at all |

Every answer ends with the same limit: a photo is a hint, a lab assay is proof. Below 25 %
the model does not name anything and asks for a better photo.

## Pipeline

1. `scripts/geo_fetch.py`: metadata from each source into `data/geo/manifest.jsonl`, one row
   per image with licence, attribution, specimen record and find location.
2. `scripts/geo_build.py`: class mapping, caps per class and per specimen, download, resize
   to 192 px (re-encoded, no EXIF), near-duplicate removal, split by specimen
   (80 / 10 / 10, a hash of the specimen id, so new data never moves old specimens).
3. `scripts/train_geo.py`: GeoNet from random weights, class-balanced sampling, colour-honest
   augmentation (no hue shift), checkpoint chosen on validation, temperature calibration on
   validation, test evaluated once. Snapshot to the CEO desk as model `nics-vision`.
4. `scripts/geo_identify.py photo.jpg`: one photo, answer in Russian or English.

On the laptop: `scripts\windows\geo_train.bat` (add `--max-minutes 60 --gentle` by day).

## Model

GeoNet (`src/nics/geo/model.py`): inverted-residual convolutional blocks with
squeeze-excitation, two heads (class and coarse group), about 0.98 M parameters at
160 x 160. Chosen for a CPU without a GPU.

## Data

Per-image licence gate (`src/nics/geo/licence.py`): CC0, public domain, CC BY, CC BY-SA;
NC, ND and unknown refused. Sources and counts: [`docs/dataset-register.md`](dataset-register.md).

Known weakness: museum photos show clean specimens on plain backgrounds; field photos
show dusty, wet, weathered stone. Accuracy on museum photos will overstate accuracy in the
field until field photos with confirmed labels exist.

## Stages

| Stage | Content | Gate to the next one |
|---|---|---|
| 0 | Data: classes, licences, sources, splits | No specimen in two splits (tested) |
| 1 | Model, training, calibration, answer | Test accuracy well above chance, calibrated probabilities |
| 2 | Wikimedia Commons photos from the laptop; identification service; location layer from public deposit databases; user photos for training only with opt-in consent | Accuracy on the first field photos with confirmed labels |
| 3 | Satellite: alteration mapping from multispectral bands; own model on image patches labelled from deposit databases | Needs a GPU or rented compute |
| 4 | Video frames; Moon and Mars from public NASA data | |

## Rules kept

* No pretrained weights, no external model in the core.
* Texts about indicators and tests are written by people (`taxonomy.py`), the model only
  gives probabilities. No claim of a deposit, no money value, no oil or gas from a rock photo.
* User photos never enter training without explicit consent; EXIF (GPS, camera) is stripped.
