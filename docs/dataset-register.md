# Dataset Register

Human-readable view of [`data_registry/datasets.json`](../data_registry/datasets.json),
the machine-readable source of truth. Datasets themselves are **not**
stored in this repository (`.gitignore`: `data/`, `datasets/`) -- only
this provenance record is.

## Eligibility rules for adding a dataset

A dataset may only be registered and used for training if all of the
following hold:

1. **License is unambiguous** -- public domain, a permissive open license,
   or data AI Byte Consult Ltd has explicit rights to (owned, licensed, or
   properly consented). "Scraped, license unclear" is not eligible.
2. **Commercial use is permitted** -- NICS AI Trader and the wider NICS AI
   Ecosystem are commercial products; a dataset restricted to
   non-commercial/research-only use is not eligible for models that ship.
3. **No personal data risk**, or personal data has been properly consented
   / anonymised and reviewed.
4. **Checksum recorded at acquisition time** (`checksum_sha256`,
   `size_bytes`) so a later re-download can be verified to be byte-identical
   to what was actually used for training.

Every entry in `data_registry/datasets.json` must carry: `dataset_id`,
`name`, `source`, `source_url`, `version`, `acquisition_date`, `license`,
`permitted_use`, `commercial_use`, `personal_data_risk`, `copyright_risk`,
`checksum_sha256`, `size_bytes`, `notes`.

## Registered datasets

### `tinyshakespeare-v1`

| Field | Value |
|---|---|
| Name | Tiny Shakespeare |
| Source | `karpathy/char-rnn` GitHub mirror of Shakespeare's complete works |
| Source URL | `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt` |
| Acquired | 2026-08-05 |
| License | Public domain (William Shakespeare, d. 1616) |
| Commercial use | Yes |
| Personal data risk | None |
| Copyright risk | None -- public domain source text |
| Size | 1,115,394 bytes |
| SHA-256 | `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed` |
| Used by | `nics-text-nano` / `exp-0001` |

Purpose: smallest possible unambiguously-licensed corpus to validate the
from-scratch tokenizer + training pipeline end to end. Not intended, on
its own, to produce a broadly capable language model -- see
[`model_cards/nics-text-nano.md`](../model_cards/nics-text-nano.md) for
honest limitations.

### `fashion-mnist-v1`

| Field | Value |
|---|---|
| Name | Fashion-MNIST |
| Source | `zalandoresearch/fashion-mnist` GitHub repo (Zalando Research) |
| Source URL | `https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/{train,t10k}-{images,labels}-idx{3,1}-ubyte.gz` |
| Acquired | 2026-08-05 |
| License | MIT (Zalando SE, 2017) |
| Commercial use | Yes |
| Personal data risk | None |
| Copyright risk | None -- Zalando's own product catalog photographs, MIT-licensed by the copyright holder (not a third-party scrape) |
| Size | 30,878,645 bytes (4 files combined) |
| SHA-256 | per-file, see `data_registry/datasets.json` |
| Used by | `nics-vision-nano` / `exp-0001` |

Purpose: 10-class grayscale clothing image classification, chosen over
Imagenette/Imagewoof after checking licenses directly rather than trusting
the repo-level license badge -- Imagenette/Imagewoof's Apache-2.0 license
covers the `fastai/imagenette` code only, and neither its README nor
LICENSE make any explicit commercial-use grant for the underlying
ImageNet-derived images, which fails Section 1's "license is unambiguous"
requirement. Fashion-MNIST's MIT license, by contrast, is issued by
Zalando SE, the actual copyright holder of the source photographs, and
covers the dataset itself, not just surrounding code.

### `nics-market-v2` (added 2026-11-18, replaces v1 for training)

All our trading data from the private, read-only n8n workflow
`[41 LAB] Market Data Export`: Vantage MT5 bridge candles on every timeframe
(1min, 5min, 15min, 1h, 4h; 27 instruments), every scanner signal (all
modes, strong and weak) with levels and recorded result, signal lifecycle
events, shadow strategies, EA manual signals, economic calendar, trade
journal, instrument specs. Weak and on-request signals are used for
internal training only and never enter public statistics. The calendar
starts 2026-10-20, so it is exported but not yet a model feature.

### `nics-public-history-v1` (added 2026-11-18)

Public D1/W1 history from Kraken (BTC, ETH) and Yahoo Finance (indices,
gold/silver/Brent futures, US stocks, FX), fetched by
`scripts/fetch_public_history.py`. Not our broker's prices; every row keeps
its source. **License caveat:** third-party terms, internal research only,
never republished. Commercial use is unclear (eligibility rule 2), so a
model meant to ship is either retrained with `--no-public` or the terms are
confirmed first. Added on Alexander's explicit request.

### `nics-own-trades-v1` (added 2026-11-18)

Our closed positions from the MT5 terminal (prop, Vantage, algo accounts),
read locally by `scripts/fetch_mt5_history.py`. The account number is not
stored.

## Planned

No further datasets are registered yet.

### `nics-market-v1` (added 2026-11-18)

Our own market data for NICS Trading: Vantage MT5 bridge candles and the
NICS scanner's signals with their outcomes, exported by the private n8n
workflow `[41 LAB] Market Data Export`. Snapshots live only on the training
machine under `data/market/<date>/`, each with a `manifest.json` holding row
counts and sha256; the training manifest copies those hashes. No user data.
