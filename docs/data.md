# Preparing photos

The photos are sorted into folders, and the folder structure tells the pipeline which photos belong to which individual or encounter. What is needed depends on the type of the project:

- **Identification** needs two folders: the **database** with photos of known individuals and the **query** with new photos.
- **Discovery** needs only the **query** with new photos.

!!! info "Species"
    The body parts are found by the [TurtleDetector](https://huggingface.co/BVRA/TurtleDetector), which is trained on sea turtles. Other species need a different detector.

## Folder structure

!!! note
    The structure below is only an example. Any other structure can be used, as described below.

=== "Identification"

    ```
    my_project/
    ├── database/
    │   ├── t001/
    │   │   ├── IMG_0001.jpg
    │   │   └── IMG_0002.jpg
    │   └── t002/
    │       └── IMG_0103.jpg
    └── query/
        └── 2024-06-12/
            ├── encounter_01/
            │   ├── IMG_2001.jpg
            │   └── IMG_2002.jpg
            └── encounter_02/
                └── IMG_2010.jpg
    ```

    - **Database**: `<identity>/<photo>`. The folder name is the identity of the individual.
    - **Query**: `<date>/<encounter>/<photo>`. All photos in one encounter folder must show the same animal. The folder names can be anything, they only need to be different for different encounters on the same date.

=== "Discovery"

    ```
    my_project/
    └── query/
        ├── 2024-06-12/
        │   ├── encounter_01/
        │   │   ├── IMG_2001.jpg
        │   │   └── IMG_2002.jpg
        │   └── encounter_02/
        │       └── IMG_2010.jpg
        └── 2024-06-13/
            └── encounter_01/
                └── IMG_2105.jpg
    ```

    - **Query**: `<date>/<encounter>/<photo>`. All photos in one encounter folder must show the same animal. The folder names can be anything, they only need to be different for different encounters on the same date.

In the notebook, the folder levels are set by `levels` in `Dataset_FolderStructured`:

=== "Identification"

    ```python
    data = [
        (root_query, False, ("date", "encounter_id")),
        (root_database, True, ("identity",)),
    ]
    ```

=== "Discovery"

    ```python
    data = [
        (root_query, False, ("date", "encounter_id")),
    ]
    ```

If your folders have a different number of levels, change `levels` accordingly, while keeping these rules:

- The database should always specify `identity`.
- All photos must be exactly at the depth described above. Photos at another depth cause an error.
- Supported formats are `.jpg`, `.jpeg`, `.png`, `.tiff` and `.bmp`.
- Every photo should show only one animal. Photos with several animals of the same species can produce wrong matches.

!!! tip "Why encounters"
    Different photos of one encounter often show different body parts or different sides of the animal. Grouping them into an encounter combines the evidence of all photos into one prediction.

## Using custom dataset loaders

If the folder structure does not fit, or the identities and encounters are stored in a table, load the photos by any loader from [wildlife-datasets](https://wildlifedatasets.github.io/wildlife-datasets/adding/) and save its metadata as `metadata.csv` next to the photos. It must contain the columns `image_id`, `path`, `identity` and optionally `encounter_id`.

!!! warning
    `image_id` must be text with at least 10 characters, and the query and database must not share any `image_id`. All computed results are stored under it. Create it from the paths by `get_persistent_id(metadata["path"])` from `wildlife_datasets.datasets.utils`.

## Setting the folders in the notebook

In section 0 of the notebook, set the paths to your folders and skip the cell that downloads the example data:

=== "Identification"

    ```python
    root_query = "path/to/my_project/query"
    root_database = "path/to/my_project/database"
    root_cache = "output/cache_my_project"
    ```

=== "Discovery"

    ```python
    root_query = "path/to/my_project/query"
    root_cache = "output/cache_my_project"
    ```

Use a different `root_cache` for every project.
