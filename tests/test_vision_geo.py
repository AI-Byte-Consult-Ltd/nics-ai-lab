"""NICS Vision Geo: labels, licences, source readers, splits, model and honest answers.
Synthetic data only, no network."""
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.geo import advisor, dataset, licence, sources, taxonomy, train  # noqa: E402
from nics.geo.model import GeoNet, GeoNetConfig, count_params  # noqa: E402


def test_names_map_to_classes_and_junk_is_dropped():
    m = taxonomy.map_name
    assert m("Gold on Quartz") == "gold"
    assert m("Pyrite (var. X), Peru") == "pyrite"
    assert m("Porphyritic granite") == "granite"
    assert m("Malachite with Azurite") == "copper_carbonate"
    assert m("Siliceous sinter") == "sinter"
    assert m("Anything", ["Meteorites"]) == "meteorite"
    assert m("Plastic box, small") is None
    assert m("Shark tooth sampled for Sr analysis") is None
    assert m("Unidentified rock") is None
    assert m("") is None


def test_every_class_has_texts_in_both_languages():
    for c in taxonomy.CLASSES:
        assert c.ru and c.en and c.group in taxonomy.GROUPS
        for k in c.indicates:
            assert taxonomy.INDICATORS[k]["ru"] and taxonomy.INDICATORS[k]["en"]


def test_licence_gate():
    n = licence.normalize
    assert n("CC0") == "CC0"
    assert n("http://creativecommons.org/licenses/by/4.0/") == "CC-BY-4.0"
    assert n("CC BY-SA 3.0") == "CC-BY-SA-3.0" and licence.share_alike("CC-BY-SA-3.0")
    assert n("Public domain") == "PD"
    assert n("https://creativecommons.org/publicdomain/zero/1.0/") == "CC0"
    for bad in ("CC BY-NC 4.0", "CC BY-ND 2.0", "CC BY-NC-SA 4.0", "All rights reserved", "Fair use", "", None,
                "https://creativecommons.org/licenses/by-nc/4.0/"):
        assert n(bad) is None, bad


def test_smithsonian_reader_keeps_only_cc0_images():
    rec = {"id": "edanmdm:nmnhminsci_1", "title": "Galena", "content": {
        "descriptiveNonRepeating": {"record_link": "http://n2t.net/x", "online_media": {"media": [
            {"id": "m1", "type": "Images", "content": "https://ids.si.edu/a", "usage": {"access": "CC0"}},
            {"id": "m2", "type": "Images", "content": "https://ids.si.edu/b", "usage": {"access": "Usage conditions apply"}},
        ]}},
        "indexedStructured": {"topic": ["Mineralogy"], "place": ["Peru"],
                              "geoLocation": [{"points": {"point": {"latitude": {"content": "-9.5"}, "longitude": {"content": "-77.1"}}}}]}}}
    rows = sources.smithsonian_rows_from_record(rec)
    assert len(rows) == 1 and rows[0]["class"] == "galena" and rows[0]["licence"] == "CC0"
    assert rows[0]["specimen_id"] == "si:edanmdm:nmnhminsci_1" and rows[0]["lat"] == -9.5


def test_nhm_reader_and_commons_reader():
    r = {"_id": 1, "occurrenceID": "abc", "scientificName": "Pink granite", "associatedMedia": [
        {"assetID": "x1", "type": "StillImage", "identifier": "https://data.nhm.ac.uk/media/x1",
         "license": "http://creativecommons.org/licenses/by/4.0/", "creator": "A. Person"},
        {"assetID": "x2", "type": "StillImage", "identifier": "https://data.nhm.ac.uk/media/x2",
         "license": "http://creativecommons.org/licenses/by-nc/4.0/"}]}
    rows = sources.nhm_rows_from_record(r)
    assert [x["image_id"] for x in rows] == ["nhm:x1"] and rows[0]["url"].endswith("/x1/thumbnail")
    assert rows[0]["class"] == "granite" and "A. Person" in rows[0]["attribution"]
    pages = [
        {"title": "File:Pyrite.jpg", "imageinfo": [{"mime": "image/jpeg", "thumburl": "https://u/t.jpg", "descriptionurl": "https://c/d",
                                                   "extmetadata": {"LicenseShortName": {"value": "CC BY-SA 4.0"},
                                                                   "Artist": {"value": "<a href='x'>Rock Fan</a>"}}}]},
        {"title": "File:NC.jpg", "imageinfo": [{"mime": "image/jpeg", "thumburl": "https://u/n.jpg",
                                               "extmetadata": {"LicenseShortName": {"value": "CC BY-NC 2.0"}}}]},
        {"title": "File:Logo.svg", "imageinfo": [{"mime": "image/svg+xml", "url": "https://u/l.svg",
                                                 "extmetadata": {"LicenseShortName": {"value": "CC0"}}}]},
    ]
    rows = sources.commons_rows_from_pages(pages, "pyrite", "Pyrite")
    assert len(rows) == 1 and rows[0]["licence"] == "CC-BY-SA-4.0" and rows[0]["attribution"].startswith("Rock Fan")


def _rows(n_spec=30, per=3, classes=("quartz", "pyrite", "granite")):
    rows = []
    for c in classes:
        for s in range(n_spec):
            for k in range(per):
                rows.append({"image_id": f"t:{c}:{s}:{k}", "specimen_id": f"t:{c}:{s}", "class": c,
                             "source": "test", "licence": "CC0", "url": f"mem://{c}/{s}/{k}"})
    return rows


def test_split_is_by_specimen_and_stable():
    rows = _rows()
    chosen = dataset.select(rows, min_images=10, max_per_class=1000, max_per_specimen=2)
    assert all(sum(1 for r in chosen if r["specimen_id"] == sp) <= 2 for sp in {r["specimen_id"] for r in chosen})
    by_spec = {}
    for r in chosen:
        by_spec.setdefault(r["specimen_id"], set()).add(dataset.split_of(r["specimen_id"]))
    assert all(len(v) == 1 for v in by_spec.values())          # one specimen never spans two splits
    assert dataset.split_of("t:quartz:1") == dataset.split_of("t:quartz:1")
    assert dataset.select(rows, min_images=1000, max_per_class=10, max_per_specimen=1) == []   # too-small classes dropped
    assert len(dataset.select(rows, min_images=1, max_per_class=10, max_per_specimen=3)) == 30


def _fake_image(cls: str, s: int, k: int, size: int = 64) -> bytes:
    rng = np.random.default_rng(abs(hash((cls, s, k))) % (2 ** 32))
    base = {"quartz": (220, 220, 230), "pyrite": (190, 160, 60), "granite": (150, 120, 120)}[cls]
    arr = np.clip(np.array(base)[None, None, :] + rng.normal(0, 40, (size, size + 16, 3)), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def test_build_removes_near_duplicates_and_strips_metadata():
    rows = _rows(n_spec=8, per=2)
    data = {r["url"]: _fake_image(r["class"], int(r["url"].split("/")[3]), int(r["url"].split("/")[4])) for r in rows}
    data[rows[1]["url"]] = data[rows[0]["url"]]                 # the same picture uploaded twice
    with tempfile.TemporaryDirectory() as td:
        s = dataset.build(rows, Path(td) / "built", Path(td) / "cache", 48, lambda u: data[u], workers=2, log=lambda m: None)
        assert s["near_duplicates_dropped"] >= 1 and s["images"] == len(rows) - s["near_duplicates_dropped"]
        images, meta, summary = dataset.load_built(Path(td) / "built")
        assert images.shape[1:] == (48, 48, 3) and len(meta) == len(images)
        png = next((Path(td) / "cache").rglob("*.png"))
        assert not Image.open(png).info.get("exif")


def test_geonet_is_small_and_starts_from_random_weights():
    m = GeoNet(GeoNetConfig(num_classes=40, num_groups=11))
    n = count_params(m)
    assert 500_000 < n < 2_000_000, n
    c, g = m(torch.zeros(2, 3, 160, 160))
    assert c.shape == (2, 40) and g.shape == (2, 11)
    m2 = GeoNet(GeoNetConfig(num_classes=40, num_groups=11))
    assert not torch.equal(m.stem[0].weight, m2.stem[0].weight)  # random init, nothing loaded


def test_training_learns_colour_classes_and_calibrates():
    rng = np.random.default_rng(0)
    classes = ["pyrite", "quartz", "granite"]
    base = np.array([(190, 160, 60), (225, 225, 235), (120, 80, 80)])
    imgs, meta = [], []
    for i in range(180):
        c = i % 3
        imgs.append(np.clip(base[c][None, None, :] + rng.normal(0, 25, (40, 40, 3)), 0, 255).astype(np.uint8))
        meta.append({"class": classes[c], "split": "train" if i < 120 else "val" if i < 150 else "test"})
    summary = {"classes": sorted(classes), "images": 180, "per_split": {}, "sources": {}, "licences": {}}
    cfg = {"seed": 1, "image_size": 32, "epochs": 3, "batch_size": 32, "lr": 0.005,
           "model": {"stem": 8, "stages": [[1, 8, 1, 1], [2, 16, 1, 2]], "head": 32}}
    with tempfile.TemporaryDirectory() as td:
        r = train.train(np.stack(imgs), meta, summary, cfg, Path(td), log=lambda m: None)
        assert r["test"]["calibrated"]["top1"] > 0.8, r["test"]
        assert 0.05 <= r["temperature"] <= 20
        model, mmeta = train.load_model(Path(td))
        p = train.classify(model, mmeta, imgs[0])
        assert abs(sum(p.values()) - 1) < 1e-4 and max(p, key=p.get) == "pyrite"
        assert json.loads((Path(td) / "report.json").read_text())["chance_top1"] == round(1 / 3, 4)


def test_answer_is_honest():
    sure = advisor.answer({"gold": 0.8, "pyrite": 0.15, "mica": 0.05}, "ru")
    assert sure["level"] == "sure" and "лаборатории" in sure["text"] and sure["may_indicate"]
    assert any("Пирит" in t for t in sure["tests"])
    low = advisor.answer({"gold": 0.2, "pyrite": 0.19, "quartz": 0.18}, "en")
    assert low["level"] == "abstain" and low["may_indicate"] == [] and "cannot identify" in low["text"]
    serp = advisor.answer({"ultramafic": 0.6, "basalt": 0.4}, "en")
    assert serp["cautions"] and "asbestos" in serp["text"]
    for res in (sure, low, serp):
        assert "deposit here" not in res["text"].lower() and "€" not in res["text"]
