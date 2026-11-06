"""Download the EPIC-KITCHENS-100 annotation tables used by this project.

The tables are the public CSVs from epic-kitchens/epic-kitchens-100-annotations.
Video files are not downloaded.
"""

from pathlib import Path
from urllib.request import urlretrieve

SOURCE = "https://github.com/epic-kitchens/epic-kitchens-100-annotations/raw/master"
FILES = (
    "EPIC_100_train.csv",
    "EPIC_100_validation.csv",
    "EPIC_100_test_timestamps.csv",
    "EPIC_100_video_info.csv",
    "EPIC_100_verb_classes.csv",
    "EPIC_100_noun_classes.csv",
)
DEST = Path(__file__).resolve().parents[1] / "data" / "epic"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = DEST / name
        if path.is_file() and path.stat().st_size > 0:
            print(f"already present: {path}")
            continue
        print(f"downloading {name}")
        urlretrieve(f"{SOURCE}/{name}", path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
