from dataclasses import dataclass


@dataclass
class SimilarityResult:
    similarity: object
    query: object
    database: object
    name: str
    B: object
    img_load: str
    matching_part: str
    match_opposite_sides: object
    split_year: str

    def to_dict(self):
        return {
            "similarity": self.similarity,
            "query": self.query,
            "database": self.database,
            "name": self.name,
            "B": self.B,
            "img_load": self.img_load,
            "matching_part": self.matching_part,
            "match_opposite_sides": self.match_opposite_sides,
            "split_year": self.split_year,
        }
