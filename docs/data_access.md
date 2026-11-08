# Complete artifact access

The current code is version 2.0.0. The 3.05 GB data/model ZIP has been prepared and verified. Its new Zenodo record has not yet been published as of the 2026-11-08 repository update. Download links must refer to the actual new version once it is available.

The earlier record https://zenodo.org/records/20839925 identifies v1.0.0, with 160 support combinations. The project concept DOI is https://doi.org/10.5281/zenodo.20839924 . Neither link is asserted to contain the current three-module archive until the new version is published.

## Import a local ZIP

```bash
python scripts/prepare_archive.py --archive /path/to/longwall-surrogate-v2.0.0_20261108.zip
```

This command uses the ZIP name and SHA256 in `metadata/archive.json`. It checks archive identity, extracts the archive, verifies its deposited file checksums and installs it at `artifacts/longwall-surrogate-v2.0.0/`. It does not overwrite the GitHub source tree. Model weights and large arrays are ignored by Git.

## Use an already extracted archive

Set `LONGWALL_SURROGATE_ARTIFACTS` to the directory containing `data/`, `models/` and `SHA256SUMS.txt`. For PowerShell:

```powershell
$env:LONGWALL_SURROGATE_ARTIFACTS = 'D:\artifacts\longwall-surrogate-v2.0.0'
python predict_support.py --dataset archived --output outputs/support.csv
```

For a POSIX shell:

```bash
export LONGWALL_SURROGATE_ARTIFACTS=/path/to/longwall-surrogate-v2.0.0
python predict_fields.py --output outputs/fields.npz
```

The complete archive contains 175 files. Its checksum list covers 174 files; the checksum file itself is protected by the outer ZIP identity. The original 25 model artifacts are retained. Three normal stress bundles account for approximately 2.73 GB.

## Publication metadata

Once Zenodo publication is complete, update `metadata/archive.json` with the real version-specific DOI, landing page, download URL and status. Update the citation guidance using the actual creator metadata. Preserve the ZIP identity when it is the same file. Use a new artifact identity for changed files, with the corresponding validation record.
