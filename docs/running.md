# Running the pipeline

The notebook runs the detection, the re-identification and the export of the pairs for verification. This page describes what to change for your own data, what is stored between runs, and how to verify the pairs.

## Changes in the notebook

The notebooks are set up for the example data. For your own data, change only three things:

1. **Folders:** set the paths to your photos and skip the cell that downloads the example data, see [Preparing photos](data.md#setting-the-folders-in-the-notebook).
2. **Answers:** the cell with `true_identities` only simulates the verification for the example data. Replace it by loading your answers, see [Updating the database](updating.md#loading-the-answers).
3. **GPU (optional):** if you have an NVIDIA GPU, set `device = "cuda"`. The results are the same, only faster.

## What is stored

All computed results are stored, so the next run processes only new photos.

| File or folder | Content | Notes |
|----------------|---------|-------|
| `segmentation.csv` (next to the photos) | Detected body parts | The detection runs only for photos which are not in this file. Photos with no detection are stored too, so they are not processed again. |
| `root_cache` | Features and keypoint matches | Use a different `root_cache` for every project. Changing `img_load` creates a new cache. |
| `output/verification_data.csv` | Pairs for verification | Created again in every run. |

Deleting anything in the `output` folder is safe, only the next run takes longer.

!!! warning
    The results are stored under the photo's path. If you replace a photo by another one with the same name, the old results are kept. Delete `segmentation.csv` and `root_cache` in that case.

## Verification

The pairs for verification are checked in [Image Comparator](https://sadda.github.io/image-comparator/). After [installing it](https://sadda.github.io/image-comparator/installation/):

=== "Identification"

    1. Create a folder for the verification and copy `verification_data.csv` there.
    2. Create a config file `my_project.json` in that folder with the path to the folder which contains `query` and `database`:

        ```json
        {
          "root": "/path/to/my_project"
        }
        ```

    3. Open a terminal in that folder and run

        ```
        image-comparator identification --config my_project.json
        ```

=== "Discovery"

    1. Create a folder for the verification and copy `verification_data.csv` there.
    2. Create a config file `my_project.json` in that folder with the path to the folder with the photos (`root` in the notebook):

        ```json
        {
          "root": "/path/to/my_project/query"
        }
        ```

    3. Open a terminal in that folder and run

        ```
        image-comparator discovery --config my_project.json
        ```

Answer every pair by **Same**, **Different** or **Unknown**. The answers are saved to `answers.csv` in the same folder. See the [Image Comparator documentation](https://sadda.github.io/image-comparator/custom/) for sharing the verification with other annotators.

!!! warning
    Do not change the photos, `metadata.csv` or `segmentation.csv` between running the notebook and the verification. Image Comparator finds the photos of every pair through them.
