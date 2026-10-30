# Quick start

The fastest way to see the whole workflow is to open one of the notebooks in Google Colab:

- [Identification](https://colab.research.google.com/github/sadda/turtle_re_id/blob/master/examples/introduction.ipynb): new photos are compared with a database of known individuals.
- [Discovery](https://colab.research.google.com/github/sadda/turtle_re_id/blob/master/examples/discovery.ipynb): photos are compared with each other to find the same individuals.

The first run is slow, because all photos have to be processed. After that, adding new photos only processes the new ones.

| Part | When is it downloaded/computed |
|------|--------------------------------|
| Installing the packages | once |
| Downloading the models | once |
| Processing database photos | once |
| Processing query photos | only for new photos |

If you are satisfied with the functionality of the package and want to try more, the next step is to [install](installation.md) the package locally and use it with your own data. This also avoids the limitation of Google Colab, which deletes all results when the session ends, so everything has to be computed again.
