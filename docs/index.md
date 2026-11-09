# Animal re-identification

This tool helps ecologists recognise individual animals in new photos by comparing them with photos of already known individuals, or with each other. It was developed for sea turtles. Instead of comparing whole images, each animal is split into body parts (head, front flippers, hind flippers), and each body part is compared separately.

The computer does the time-consuming work of searching through all photos and proposes the most likely matches. The final decision always stays with a human expert.

## How it works

1. **Detection** (automatic) – the [TurtleDetector](https://huggingface.co/BVRA/TurtleDetector) finds the body parts in every photo.
2. **Re-identification** (automatic) – every body part in the new photos is compared with the same body part of the known individuals (identification), of the other new photos (discovery) or of the photos of the other individuals in the database (error detection).
3. **Verification** (manual) – you check the most promising candidate pairs in the [Image Comparator](https://github.com/sadda/image-comparator) app.
4. **Assigning identities** (automatic) – your answers are turned into one identity for every new animal (identification), into groups of photos of the same animal (discovery) or into a list of photos which are stored under a wrong identity (error detection).

## Terms used in this documentation

| Term | Meaning |
|------|---------|
| **Database** | Photos of individuals whose identity is already known. |
| **Query** | New photos whose identity you want to find. |
| **Encounter** | All photos of one animal taken at one occasion, for example on one day. Identities are assigned to whole encounters, not to single photos. |
| **Body part** | A part of the animal that is compared on its own: `heads`, `front flippers` or `hind flippers`. |

## Identification, discovery and error detection

The workflow can be used in three ways.

| | **Identification** | **Discovery** | **Error detection** |
|---|---|---|---|
| Used when | There is a database of known individuals, for example a catalogue of photo-identified turtles. | There is no such database, or the new photos contain many individuals that are not in it. | There is a database of known individuals, which may contain mistakes. |
| Question | Which known individual is this? | Which of these photos show the same individual? | Which photos of different individuals show the same individual? |
| Compared | Query with database | Query with itself | Database with itself, only photos of different individuals |
| Result | Every query encounter gets an identity from the database, or is marked as new. | Query encounters are grouped into individuals, and each group gets a new identity. | Pairs of photos with different identities which show the same individual. Such identities are merged, or such photos are moved to the correct identity. |
