# fastMRI `.env` on WCSS (#77)

`FASTMRI_FULL_URLS` in `/lustre/pd03/hpc-danbor2008-1756464546/CyFM/.env` holds 6
signed NYU links: knee multicoil train batches 0-4, plus val. Signatures expire
2026-12-11.

**Verified live**, 6/6, via:

```bash
uv run python scripts/data/fastmri_urls.py --check
```

Note: this checks with a ranged GET, not a plain HEAD. NYU's presigned links are
signed for GET specifically, so a real HEAD returns 403 regardless of validity.

Unblocks #78 (train-store build) and #76 (the gate).
