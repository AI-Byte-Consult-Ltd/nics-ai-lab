"""NICS Vision Geo: rock and mineral recognition from photos, built from scratch.

New code, new data, new model name (`nics-vision-geo-*`). Nothing here reuses the
2026-08 `nics-vision-nano` pipeline check (src/nics/models/vision, Fashion-MNIST).

Modules:
  taxonomy  -- the classes, how raw museum names map to them, what each may indicate
  licence   -- which image licences may be trained on (gate, per image)
  sources   -- metadata readers for each image source (manifest rows, no pixels)
  dataset   -- download, resize, de-duplicate, split by specimen, cache as uint8 arrays
  model     -- GeoNet, our own small convolutional network, random initialisation
  train     -- training loop, augmentation, temperature calibration, metrics
  advisor   -- turns probabilities into an honest answer (ru/en): top-3, hand tests,
               what it may indicate, and what a photo can NOT tell
"""
