# Model Card: `nics-vision-geo-micro`

Status: **prototype / experiment**. One short trial run in the cloud sandbox on museum
photos. Not deployed, not offered to anyone. See [`docs/vision-geo.md`](../docs/vision-geo.md).

## Summary

NICS Vision Geo names the most likely rock or mineral on a photo, with a calibrated
probability, alternatives and simple hand checks. Built from scratch: own architecture
(GeoNet), random initialisation, no pretrained weights, no external model.

## Provenance

| Field | Value |
|---|---|
| Model ID | `nics-vision-geo-micro` |
| Architecture | GeoNet: inverted-residual CNN with squeeze-excitation, class + group heads (`src/nics/geo/model.py`) |
| Parameters | 975,946 (39 classes, 11 groups) |
| Pretrained weights | **None** |
| Input | 160 x 160 RGB, from a 192 px cache (random crop in training, full view in evaluation) |
| Config | `configs/vision/geo_micro.yaml` (trial: 20 epochs instead of 40) |
| Data | `smithsonian-nmnh-minsci-v1` (CC0, 4,367 photos) + `nhm-london-mineralogy-v1` (CC BY 4.0, 2,316 photos), manifest sha256 `9abd38a2...aff10` |
| Split | by specimen, 80/10/10: train 5,405 / val 653 / test 625 photos; 473 near-duplicates removed, 496 downloads failed |
| Hardware | cloud sandbox, 4 CPU threads, 27.1 minutes |
| Seed | 42 |

## Trial result (2026-11-22, run exp-20261122T124155Z)

Checkpoint chosen on validation (epoch 19 of 20); test evaluated once.

| Split | Top-1 | Top-3 | Balanced accuracy | ECE |
|---|---|---|---|---|
| Validation (calibrated) | 69.4 % | 83.2 % | 60.6 % | 4.3 % |
| Test (raw) | 61.4 % | 81.4 % | 51.2 % | 5.1 % |
| Test (calibrated, T = 0.89) | 61.4 % | 81.4 % | 51.2 % | 6.0 % |

Chance for 39 classes: 2.6 %. Validation accuracy was still rising at the last epoch:
the run is under-trained, the laptop config (40 epochs, more data) is the real first run.

Good on the test split: meteorite 95 % (86 photos), siliceous sinter 87 %, sandstone 84 %,
zeolites 80 %, galena 76 %, granite 75 %. Weak: corundum, fluorite, garnet, tourmaline 0 %,
quartz and chalcopyrite 25 %, marble 43 % (confused with limestone 19 times in both
directions). Many test classes have under 10 photos, so per-class numbers are noisy.

Calibration: temperature fitted on validation improved validation ECE (6.6 -> 4.3 %) but
not test ECE (5.1 -> 6.0 %). Probabilities are roughly honest (within ~5-6 points) on
museum photos; not measured on field photos at all.

## Limitations

* **Museum photos only.** Clean specimens on plain backgrounds. Expect clearly lower
  accuracy on phone photos from a beach, garden or farm until field photos with confirmed
  labels are added. Do not publish these numbers without "on museum photos".
* Classes missing for lack of data in this run: pyrite (23 photos), hematite, magnetite,
  cassiterite, gypsum, volcanic glass and others (11 of the 50 taxonomy classes).
* Does not detect deposits, gold content, oil or gas. The answer text says so every time.
* No "not a rock" class yet: a photo of something else still gets a rock name, only the
  25 % abstain threshold protects against nonsense.

## Next

Wikimedia Commons photos from the laptop, 40 epochs on the laptop, then the first field
photos with confirmed labels (stage 2).
