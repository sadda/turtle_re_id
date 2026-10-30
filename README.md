<p align="center">
  <a href="https://github.com/sadda/turtle_re_id/actions/workflows/ci.yml"><img src="https://github.com/sadda/turtle_re_id/actions/workflows/ci.yml/badge.svg" alt="sadda/turtle_re_id CI"></a>
  <a href="https://colab.research.google.com/github/sadda/turtle_re_id/blob/master/examples/introduction.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
  <a href="https://github.com/sadda/turtle_re_id/blob/master/LICENSE"><img src="https://img.shields.io/github/license/sadda/turtle_re_id" alt="License"></a>
</p>

<!-- TODO: add the sketch
<p align="center">
<img src="docs/resources/sketch.png" alt="Workflow" width="700">
</p>
-->

# Multi-body-part animal re-identification

This repository helps ecologists recognise individual animals in new photos by comparing them with photos of already known individuals. It was developed for sea turtles, but the approach works for any species with distinctive body parts. Instead of comparing whole images, each animal is split into body parts (such as head and flippers), which are compared separately. The computer does the time-consuming work of searching through all photos and proposes the most likely matches, while the final decision stays with a human expert. The researcher keeps full control over every identity assigned, and the results stay reliable even for difficult photos. No GPU is needed, because all computed results are stored and adding new photos only processes the new ones.

## How it works

1. **Detection** (automated) – the [TurtleDetector](https://huggingface.co/BVRA/TurtleDetector) finds the body parts in every photo.
2. **Re-identification** (automated) – every body part of the new photos is compared with the same body part of known individuals.
3. **Verification** (manual) – the most promising candidate pairs are checked by a human in the [Image Comparator](https://github.com/sadda/image-comparator) app.
4. **Propagation** (automated) – the verified answers are used to assign identities to the new photos.

## Getting started

See the [introduction notebook in Google Colab](https://colab.research.google.com/github/sadda/turtle_re_id/blob/master/examples/introduction.ipynb). It is not necessary to install anything to run it.

## Installation

The simplest way is to run the [introduction notebook in Google Colab](https://colab.research.google.com/github/sadda/turtle_re_id/blob/master/examples/introduction.ipynb), which needs no installation.

For a local installation, install Git and [uv](https://docs.astral.sh/uv/) as described in [Image Comparator](https://github.com/sadda/image-comparator#step-1-install-git-and-uv). Then run

```
git clone https://github.com/sadda/turtle_re_id
cd turtle_re_id
uv venv --python 3.14
uv pip install -e ".[models]" jupyter
```

This creates a virtual environment in the `.venv` folder and installs the package into it. Before using it, activate the environment by `source .venv/bin/activate` (Linux, Mac) or `.venv\Scripts\activate` (Windows). Then the notebook can be started by

```
jupyter notebook examples/introduction.ipynb
```

If DINOv3 is used for feature extraction, you will need to log in to HuggingFace with access to DINOv3 (`huggingface-cli login`).


## Contact

In case of problems, contact us by <a href="mailto:wilddatasets@gmail.com">:mailbox_with_mail:Email</a>.
