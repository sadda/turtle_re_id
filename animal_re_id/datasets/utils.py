from datetime import datetime


def add_date_exif(dataset, i, file_name):
    date = datetime.fromisoformat(dataset.df.loc[i, "date"])
    date = date.strftime("%Y:%m:%d %H:%M:%S")

    img = dataset[i]
    exif = img.getexif()

    for tag in [306, 36867, 36868]:
        exif[tag] = date
    img.save(file_name, exif=exif)


def get_wildlife_dataset(matching_dataset, matching_part):
    labels = matching_dataset.matching_parts[matching_part]["labels"]
    mask = matching_dataset.wildlife_dataset.metadata["label"].isin(labels)
    return matching_dataset.wildlife_dataset.get_subset(mask)
