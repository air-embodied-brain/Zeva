# Zeva Project Page

This folder is a ready-to-publish GitHub Pages site for the Zeva paper.

## To deploy on GitHub Pages

1. Push this `zeva-web` folder (or its contents) to a GitHub repository.
2. In the repository **Settings → Pages**, choose **Deploy from a branch**, and select the branch and `/` (root) or `/docs`.
3. If you only want to publish this folder as the site root, copy the files into the repository root (or keep `zeva-web` as the root if the repo is named `zeva-web.github.io`).

## Before publishing

Replace the placeholder items in `index.html`:

- Authors and affiliations
- `arXiv`, `Code`, `PDF`, `Model`, and `BibTeX` links
- The `Model` button currently uses `href="#"`; replace it with the Hugging Face model URL
- Any placeholder project page URL

The `assets/` directory contains the latest figures copied from `@zeva_paper/` (scaling curves, qualitative robot images, t-SNE plots, human warm-up images, and the architecture figure), and `assets/videos/` contains the demo videos copied from `@zeva_video/` (excluding `rollout.mp4`), compressed to 720p H.264 so they can be hosted on GitHub Pages.

## Files

- `index.html` — single-page project site
- `assets/` — images used by the page
