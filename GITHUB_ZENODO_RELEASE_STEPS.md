# GitHub and Zenodo release steps

This repository is prepared as a supplementary data and source-code package for the manuscript on boundary-aware surrogate modelling of minimum equivalent support pressure in deep longwall mining.

## 1. Before upload

1. Replace placeholder author information in `CITATION.cff`.
2. Replace placeholder author information and repository links in `.zenodo.json`.
3. Confirm that anonymised field data do not contain confidential mine records.
4. Confirm that FLAC3D files are example scripts and do not include proprietary Itasca software files.

## 2. Create a GitHub repository

1. Create a public GitHub repository, for example:

   `deep-longwall-support-pressure-surrogate`

2. Add the remote repository locally:

   ```bash
   git remote add origin https://github.com/USERNAME/deep-longwall-support-pressure-surrogate.git
   git branch -M main
   git push -u origin main
   ```

## 3. Connect GitHub to Zenodo

1. Log in to Zenodo.
2. Open the GitHub integration page.
3. Enable the GitHub repository in Zenodo.
4. Create a GitHub release, for example `v1.0.0`.
5. Zenodo will archive the release and generate a DOI.

## 4. Update the manuscript

After Zenodo generates the DOI, update the Data and code availability statement with:

1. GitHub repository URL.
2. Zenodo DOI.
3. Version tag, for example `v1.0.0`.

Recommended statement:

The anonymised FLAC3D-derived parameter tables, support-pressure labels, model-evaluation tables, learning-curve data, field pressure conversion summary, and source scripts are available at the GitHub repository and archived on Zenodo under DOI: TO_BE_ADDED.
