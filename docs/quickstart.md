# Quick start

The fastest way to see the whole workflow is the introduction notebook opened in [Google Colab](https://colab.research.google.com/github/sadda/turtle_re_id/blob/master/examples/introduction.ipynb). The first run is slow, because all database photos have to be processed. After that, adding new photos only processes the new ones.

| Part | When is it downloaded/computed |
|------|--------------------------------|
| Installing the packages | once |
| Downloading the models | once |
| Processing database photos | once |
| Processing query photos | only for new photos |

If you are satisfied with the functionality of the package and want to try more, the next step is to [install](installation.md) the package locally and use it with your own data. This also solves the issue with Google Colab, which deletes the run results and everything needs to be computed from scratch.
 