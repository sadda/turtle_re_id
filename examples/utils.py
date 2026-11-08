import os
import urllib.request
import zipfile

URL_EXAMPLES = "https://raw.githubusercontent.com/sadda/wildlife-labels/refs/heads/main/Example"


def download_example(name: str, root: str, roots_clean: list[str]) -> None:
    if os.path.exists(root):
        return

    url = f"{URL_EXAMPLES}/{name}.zip"
    file_name = f"{name}.zip"
    urllib.request.urlretrieve(url, file_name)
    with zipfile.ZipFile(file_name) as f:
        f.extractall(root)
    os.remove(file_name)

    for root_clean in roots_clean:
        os.remove(f"{root_clean}/metadata.csv")
        os.remove(f"{root_clean}/segmentation.csv")
