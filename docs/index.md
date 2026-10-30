# Animal re-identification

This tool helps ecologists recognise individual animals in new photos by comparing them with photos of already known individuals. It was developed for sea turtles. Instead of comparing whole images, each animal is split into body parts (head, front flippers, hind flippers), and each body part is compared separately.

The computer does the time-consuming work of searching through all photos and proposes the most likely matches. The final decision always stays with a human expert.

## How it works

1. **Detection** (automatic) – the [TurtleDetector](https://huggingface.co/BVRA/TurtleDetector) finds the body parts in every photo.
2. **Re-identification** (automatic) – every body part in the new photos is compared with the same body part of the known individuals.
3. **Verification** (manual) – you check the most promising candidate pairs in the [Image Comparator](https://github.com/sadda/image-comparator) app.
4. **Assigning identities** (automatic) – your answers are turned into one identity for every new animal.

## Terms used in this documentation

| Term | Meaning |
|------|---------|
| **Database** | Photos of individuals whose identity is already known. |
| **Query** | New photos whose identity you want to find. |
| **Encounter** | All photos of one animal taken at one occasion, for example on one day. Identities are assigned to whole encounters, not to single photos. |
| **Body part** | A part of the animal that is compared on its own: `heads`, `front flippers` or `hind flippers`. |

## Identification and discovery

The workflow can be used in two ways.

| | **Identification** | **Discovery** |
|---|---|---|
| Question | Which known individual is this? | Which of these photos show the same individual? |
| Compared | Query with database | Query with itself |
| Result | Every query encounter gets an identity from the database, or is marked as new. | Query encounters are grouped into individuals, and each group gets a new identity. |

**Identification** is used when there is a database of known individuals, for example a catalogue of photo-identified turtles. **Discovery** is used when there is no such database, or when the new photos contain many individuals that are not in it. Both are verified in [Image Comparator](https://sadda.github.io/image-comparator/custom/), where the mode is set by the dataset type `Identification` or `Discovery`.

The [introduction notebook](https://colab.research.google.com/github/sadda/turtle_re_id/blob/master/examples/introduction.ipynb) shows identification.

