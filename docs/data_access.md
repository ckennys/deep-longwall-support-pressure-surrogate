# Complete artifact access

The complete **version 2.0.0** data/model archive was published on 2026-11-08.

* Version DOI: [10.5281/zenodo.24049496](https://doi.org/10.5281/zenodo.24049496)
* [Zenodo record](https://zenodo.org/records/24049496)
* [Download the complete ZIP](https://zenodo.org/api/records/24049496/files/longwall-surrogate-v2.0.0_20261108.zip/content)
* File: `longwall-surrogate-v2.0.0_20261108.zip`
* Size: 3,048,576,124 bytes, approximately 3.05 GB
* SHA256: `b56acdc8f97d8e6e97a3813dab3b4a800ac34ce872946a6fcdb9b32c14dfb5e7`
* Zenodo MD5: `c0ac51b24118d44014f5f9f203df31b3`

The published filename, byte count and MD5 match the locally verified complete ZIP. The public file endpoint responds successfully. See `metadata/published_archive_verification.json`.

## Import the downloaded ZIP

```bash
python scripts/prepare_archive.py --archive /path/to/longwall-surrogate-v2.0.0_20261108.zip
python verify_files.py
```

The installer verifies the ZIP SHA256, extracts its files, checks the deposited payload checksums and installs them at `artifacts/longwall-surrogate-v2.0.0/`. Source files in the GitHub working tree remain separate. Large artifacts are ignored by Git.

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

The complete archive has 175 files. Its checksum list covers 174 files; the outer ZIP identity protects the checksum list itself. It contains 25 frozen model files, including three normal stress bundles totaling approximately 2.73 GB.

## Version-specific citation

Use `10.5281/zenodo.24049496` to identify these files. The historical [v1.0.0 record](https://zenodo.org/records/20839925) identifies the earlier support-only study. The [concept DOI](https://doi.org/10.5281/zenodo.20839924) connects all project versions and can resolve to a later version in future. Cite the fixed DOI supporting the results being reused.
