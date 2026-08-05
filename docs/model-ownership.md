# Model &amp; Weight Ownership

## Ownership

All model weights, checkpoints and exported artifacts produced by this
repository's code are the proprietary property of **AI Byte Consult Ltd**
(registered in Bulgaria), regardless of which machine trained them or
which storage tier currently holds them. Same terms as
[`LICENSE`](../LICENSE) -- not open source, no redistribution without
written authorization.

## Why weights aren't in this git repository

Binary weight files are large, non-diffable, and unsuited to git history.
Instead this repository tracks the *code that produces them* and the
*provenance record* (`manifest.json`, `events.jsonl`, checksums) --
anyone with repository access can reproduce or verify a run without the
repo itself carrying gigabytes of binary history. See
[`docs/architecture.md`](architecture.md) Section 4 for the exact file
list.

## Current storage (as of `nics-text-nano` / `exp-0001`)

| Artifact | Location today |
|---|---|
| Resume checkpoints (`trainer-state.pt`) | Local disk of the training machine only (laptop / sandbox `experiments/.../checkpoints/`) |
| Release export (`model.safetensors` + manifest + checksums) | Local disk only (`experiments/.../release/`) |
| Provenance record (`manifest.json`, `events.jsonl`) | Committed to this git repository |

## Planned storage (Stage 5, not yet started)

**Cloudflare R2** is the designated primary store for release-grade
weights (`model.safetensors` + `training-manifest.json` +
`sha256sums.txt`), one object prefix per `model_id`/`experiment_id`. This
requires an R2 account, bucket, and API token that have not yet been
provisioned -- see the open item in the project's working summary. Until
R2 is configured, released weights exist only on the training machine's
local disk and are not backed up off that machine.

## Access control

- git repository access: standard GitHub repository permissions on
  `AI-Byte-Consult-Ltd/nics-ai-lab` (public read, write restricted to
  authorized collaborators/tokens).
- Weight storage (R2, once provisioned): access limited to API tokens
  issued by AI Byte Consult Ltd; tokens are never committed to this
  repository (`.env`, `.env.*` are gitignored).
- No weight file, credential, or API token is ever to be pasted into
  commit messages, code comments, or chat transcripts.

## Verifying a weight file's provenance

Every `release/` directory ships `sha256sums.txt`. To verify a weight file
you were handed actually matches a specific training run:

1. Locate the matching `manifest.json` / `training-manifest.json` (git
   commit, seed, config, dataset, parameter count).
2. Recompute the SHA-256 of the weight file and compare against
   `sha256sums.txt`.
3. Cross-check the `dataset_id`(s) referenced in the manifest against
   [`docs/dataset-register.md`](dataset-register.md) to confirm licensed,
   registered data was used.

A weight file with no matching manifest, or a checksum mismatch, should be
treated as untrusted and not deployed.
