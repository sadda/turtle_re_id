# Local installation

The package can be tried without any installation in the [introduction notebook in Google Colab](https://colab.research.google.com/github/sadda/turtle_re_id/blob/master/examples/introduction.ipynb). The first code cell installs everything that is needed.

All commands are run in a terminal. If you have not used it before, the [Image Comparator installation guide](https://sadda.github.io/image-comparator/installation/) explains how to open it.

**Step 1.** Install Git and [uv](https://docs.astral.sh/uv/) as described in [Image Comparator](https://sadda.github.io/image-comparator/installation/#step-1-install-git-and-uv).

**Step 2.** Download this repository and install it:

```
git clone https://github.com/sadda/turtle_re_id
cd turtle_re_id
uv venv --python 3.14
uv pip install -e ".[models]" jupyter
```

This creates a virtual environment in the `.venv` folder and installs the package into it.

**Step 3.** Activate the environment and start the notebook:

=== "Linux, Mac"

    ```
    source .venv/bin/activate
    jupyter notebook examples/introduction.ipynb
    ```

=== "Windows"

    ```
    .venv\Scripts\activate
    jupyter notebook examples/introduction.ipynb
    ```

The environment must be activated every time you open a new terminal.

## Installing Image Comparator

The verification step uses the [Image Comparator](https://github.com/sadda/image-comparator) app. Install it by following its [installation guide](https://sadda.github.io/image-comparator/installation/). It is installed separately and runs on your computer, not in Colab.

## Updating

To get the latest version, run in the `turtle_re_id` folder:

```
git pull
uv pip install -e ".[models]"
```
