# Troubleshooting

??? question "`Images must be exactly N folder(s) deep to match levels`"
    Some photos are at a different folder depth than expected. Check the folder structure in [Preparing photos](data.md). The error message lists some of the affected photos.

??? question "`image_id must be strings with at least 10 characters`"
    The `image_id` column in your `metadata.csv` contains numbers or short names. Create the ids from the paths with `get_persistent_id`, see [Using custom dataset loaders](data.md#using-custom-dataset-loaders).

??? question "`Datasets 0 and 1 share image_id`"
    A photo appears in both the query and the database, or two `metadata.csv` files use the same ids. Remove the photo from one of the folders.

??? question "All files are gone in Google Colab"
    Colab deletes all files when the session ends. For regular work, use the [local installation](installation.md).

??? question "The notebook is very slow"
    The first run processes all database photos and can take a long time on a CPU. The next runs process only new photos. If you have an NVIDIA GPU, set `device = "cuda"`.

??? question "A turtle in the database was marked as `new`"
    None of its photos were among the proposed candidates. Check that the detected crops of this turtle look correct, see [Updating the database](updating.md#results).

??? question "Image Comparator says that `answers.csv` has wrong format"
    The `answers.csv` in the project folder belongs to an older `verification_data.csv`. Move it away and start the app again, see [Updating the database](updating.md#routine-for-new-photos).

??? question "Image Comparator says that files are missing"
    Run it from the folder that contains `verification_data.csv` and check `root` in the config file, see [Verification](running.md#verification).

## Contact

In case of other problems, contact us by [email](mailto:wilddatasets@gmail.com).
