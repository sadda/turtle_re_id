# Updating the database

The database grows with every round of verification: the identified animals are added to it, so that they help to identify future photos. Every round processes only the new photos.

## Loading the answers

After the verification, copy `answers.csv` from the Image Comparator folder next to the notebook, or set `file_name_answers` to its path. When this file exists, the notebook loads it instead of simulating the answers:

=== "Identification"

    ```python
    answers = pd.read_csv(file_name_answers)
    predictions_verified = answers_to_predictions(answers)
    ```

=== "Discovery"

    ```python
    answers = pd.read_csv(file_name_answers)
    groups = answers_to_groups(answers)
    ```

=== "Error detection"

    ```python
    answers = pd.read_csv(file_name_answers)
    errors = answers_to_errors(answers, query_dataset.wildlife_dataset.metadata)
    ```

## Results

=== "Identification"

    Every query encounter gets one of four results:

    | Result | Meaning | What to do |
    |--------|---------|------------|
    | identity, e.g. `t023` | One candidate was answered **Same**. | Move the photos to `database/t023/`. |
    | `new` | All candidates were answered **Different**. | Choose a new identity and move the photos to `database/<new identity>/`. |
    | `conflict` | Two different individuals were answered **Same**. | Check the pairs of this encounter again in Image Comparator. |
    | `None` | No candidate was answered **Same**, but some were not answered. | Answer the remaining pairs, or leave the encounter in the query for the next round. |

    !!! note
        `new` means that the individual is not among the checked candidates. Rarely, an individual which is in the database was not proposed at all, and it is marked `new` too. If you know the population well, check the `new` encounters before giving them a new identity.

    Remove the identified encounters from the query folder. Otherwise they are proposed for verification again.

=== "Discovery"

    `groups` contains one row for every group of encounters which were answered **Same**, directly or through other encounters. Every group is one individual: choose a new identity for it and move its photos to `database/<new identity>/`.

    !!! note
        Two encounters of one individual end up in different groups when no pair between them was proposed or answered **Same**, for example when the photos show opposite sides of the animal. Such groups are merged later, when the individual is photographed again.

    Once the groups are in the database, continue with identification for the new photos.

=== "Error detection"

    `errors` contains one row for every pair of photos with different identities which was answered **Same**. The photos of each pair of identities show the type of the mistake:

    | Mistake | How it looks | What to do |
    |---------|--------------|------------|
    | One individual stored under two identities | Photos from most encounters of both identities are matched. | Keep one identity and move the photos of the other one to its folder. |
    | Photo stored under a wrong identity | Only one or a few photos of an identity are matched with another identity. | Move these photos to the folder of the other identity. |

    Every answer is valid only for its pair of photos. A photo of one identity matched with a photo of another identity does not mean that all their photos show the same individual.

    !!! note
        Error detection is a check of the database, not a step of every round. Run it from time to time, for example before a larger round of identification, so that the mistakes do not spread to the new identities.

## Routine for new photos

1. Copy the new photos into the query folder.
2. Run the notebook. The detection and the re-identification run only for the new photos and for the photos which were moved to the database.
3. In the Image Comparator folder, move the old `answers.csv` away (for example to `answers_2024-06-12.csv`) and copy the new `verification_data.csv` there. Image Comparator refuses to start with an `answers.csv` which belongs to another `verification_data.csv`.
4. Answer the pairs, load the answers and update the database as described above.

!!! tip
    Keep the old `answers.csv` files. They document every identity decision.
