"""Counts and checks for the EPIC-KITCHENS-100 annotation tables.

The release stores one row per action segment. Train and validation rows
include the narration and the parsed verb and noun. The test file has the
same timing columns and does not include those labels. Segment length in
this module is the difference of the published start and stop timestamps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# This file lives in src/, so the project root is one directory up.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "epic"
ANNOTATION_SOURCE = "https://github.com/epic-kitchens/epic-kitchens-100-annotations"

SPLIT_FILES = {
    "train": "EPIC_100_train.csv",
    "validation": "EPIC_100_validation.csv",
    "test": "EPIC_100_test_timestamps.csv",
}
VIDEO_INFO_FILE = "EPIC_100_video_info.csv"
VERB_CLASS_FILE = "EPIC_100_verb_classes.csv"
NOUN_CLASS_FILE = "EPIC_100_noun_classes.csv"

# Class name published with the release, then the raw string from the narration.
LABEL_COLUMNS = {
    "verb": ("verb_name", "verb"),
    "noun": ("noun_name", "noun"),
}

# "labeled" fields are columns of the train and validation files only.
COVERAGE_FIELDS = (
    ("start timestamp", "start_s", "all"),
    ("stop timestamp", "stop_s", "all"),
    ("narration timestamp", "narration_s", "all"),
    ("narration", "narration", "labeled"),
    ("verb", "verb", "labeled"),
    ("noun", "noun", "labeled"),
)

SPLIT_ORDER = ("train", "validation", "test")


def load_segments(data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Stack the three splits, parse timestamps, and attach video and class info.

    Label columns on the test split are missing because that file does not
    contain them. ``labels_in_file`` is False for those rows, so later counts
    can leave them out instead of treating an absent column as a blank note.
    """
    data_dir = Path(data_dir)
    frames = [_read_split(data_dir, split, name) for split, name in SPLIT_FILES.items()]
    segments = pd.concat(frames, ignore_index=True)
    segments["split"] = pd.Categorical(segments["split"], SPLIT_ORDER, ordered=True)

    if segments["narration_id"].duplicated().any():
        raise ValueError("narration_id is not unique across the three splits.")
    if segments["start_s"].isna().any() or segments["stop_s"].isna().any():
        raise ValueError("A segment is missing its start or stop timestamp.")

    segments["duration_s"] = segments["stop_s"] - segments["start_s"]
    segments = segments.merge(_read_video_info(data_dir), on="video_id", how="left", validate="many_to_one")
    if segments["video_duration_s"].isna().any():
        raise ValueError("At least one segment has no row in EPIC_100_video_info.csv.")

    segments = _attach_class_names(segments, data_dir)
    labeled = segments["labels_in_file"]
    if segments.loc[labeled, "verb_name"].isna().any() or segments.loc[labeled, "noun_name"].isna().any():
        raise ValueError("A labeled segment uses a verb or noun class id missing from the class tables.")

    ordered = ["split", "narration_id"]
    return segments.sort_values(ordered, kind="stable").reset_index(drop=True)


def collection_counts(segments: pd.DataFrame) -> pd.DataFrame:
    """Count participants, videos, and segments.

    The ``all`` row counts distinct people and videos. Split rows do not add
    up to that row, because the same participant can appear in more than one split.
    """
    rows = [_count_row(str(split), part) for split, part in segments.groupby("split", observed=True)]
    rows.append(_count_row("all", segments))
    return pd.DataFrame(rows)


def participant_overlap(segments: pd.DataFrame) -> pd.DataFrame:
    """Count how many splits each participant appears in."""
    n_splits = segments.groupby("participant_id", observed=True)["split"].nunique()
    counts = n_splits.value_counts().sort_index()
    return pd.DataFrame({"splits": counts.index.astype(int), "participants": counts.to_numpy()})


def participant_summary(segments: pd.DataFrame) -> pd.DataFrame:
    """Describe how unevenly segments are spread across participants."""
    counts = segments.groupby("participant_id", observed=True).size().sort_values(ascending=False)
    total = int(counts.sum())
    return pd.DataFrame(
        [
            {
                "participants": int(counts.size),
                "min_segments": int(counts.min()),
                "median_segments": float(counts.median()),
                "max_segments": int(counts.max()),
                "largest_participant": counts.index[0],
                "largest_share": float(counts.iloc[0] / total),
                "top5_share": float(counts.iloc[:5].sum() / total),
            }
        ]
    )


def coverage(segments: pd.DataFrame) -> pd.DataFrame:
    """Share of segments in which each field is present.

    For narration, verb, and noun, the test split is reported as not in the
    file. A missing share there means the column was not released. That is
    different from a blank cell in train or validation.
    """
    rows: list[dict] = []
    grouped = {str(split): part for split, part in segments.groupby("split", observed=True)}
    grouped["train+validation"] = segments[segments["labels_in_file"]]
    grouped["all"] = segments

    for field, column, scope in COVERAGE_FIELDS:
        if scope == "labeled":
            split_names = ("train", "validation", "test", "train+validation")
        else:
            split_names = ("train", "validation", "test", "all")
        for split in split_names:
            part = grouped[split]
            in_file = scope != "labeled" or bool(part["labels_in_file"].all())
            rows.append(_coverage_row(field, split, part, column, in_file))
    return pd.DataFrame(rows)


def duration_summary(segments: pd.DataFrame) -> pd.DataFrame:
    """Summarize action length in seconds, using stop time minus start time."""
    rows = [_duration_row(str(split), part) for split, part in segments.groupby("split", observed=True)]
    rows.append(_duration_row("all", segments))
    return pd.DataFrame(rows)


def class_frequencies(
    segments: pd.DataFrame,
    label: str,
    split: str = "train+validation",
    n: int = 10,
) -> pd.DataFrame:
    """Return the most common verb or noun classes and their shares.

    Classes are the 97 verb ids and 300 noun ids published with the release.
    Several raw narration strings map onto one class, so those raw strings
    are not the units counted here.
    """
    class_column, _raw_column = _label_columns(label)
    part = _labeled_subset(segments, split)
    counts = part[class_column].value_counts()
    top = counts.head(n)
    return pd.DataFrame(
        {
            "class_name": top.index,
            "segments": top.to_numpy(),
            "share": top.to_numpy() / len(part),
        }
    )


def class_concentration(segments: pd.DataFrame, label: str) -> pd.DataFrame:
    """Share of labeled segments falling in the most common classes, by split."""
    class_column, raw_column = _label_columns(label)
    rows = []
    for split in ("train", "validation", "train+validation"):
        part = _labeled_subset(segments, split)
        counts = part[class_column].value_counts(normalize=True)
        rows.append(
            {
                "split": split,
                "segments": len(part),
                "classes_used": int(part[class_column].nunique()),
                "raw_strings": int(part[raw_column].nunique()),
                "top_class": counts.index[0],
                "top_share": float(counts.iloc[0]),
                "top10_share": float(counts.iloc[:10].sum()),
            }
        )
    return pd.DataFrame(rows)


def recording_length(segments: pd.DataFrame) -> pd.DataFrame:
    """Hours of source video, and hours that fall inside action segments.

    Video hours count each video once. The proposal's "100 hours" refers to
    that total. Segment hours are shorter because the annotated actions do
    not cover every moment of every video.
    """
    videos = segments.drop_duplicates("video_id")
    return pd.DataFrame(
        [
            {
                "videos": int(len(videos)),
                "video_hours": float(videos["video_duration_s"].sum() / 3600),
                "segment_hours": float(segments["duration_s"].sum() / 3600),
            }
        ]
    )


def recording_formats(segments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count distinct videos by listed frame rate and resolution.

    Each video is counted once. Segment rows are not counted again.
    """
    videos = segments.drop_duplicates("video_id")
    fps = videos["fps"].round(2).value_counts().rename_axis("fps").reset_index(name="videos")
    resolution = videos["resolution"].value_counts().rename_axis("resolution").reset_index(name="videos")
    return fps, resolution


def frame_clock_report(segments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare clock duration with frame span divided by the listed frame rate.

    The summary covers videos whose median ratio of those two lengths is
    between 0.9 and 1.1. The second table is one row per video outside that
    band. This report does not rescale any video. Duration elsewhere in this
    module still uses the timestamps.
    """
    work = segments.copy()
    frame_span = work["stop_frame"] - work["start_frame"]
    work["frame_duration_s"] = frame_span / work["fps"]
    work["_ratio"] = work["frame_duration_s"] / work["duration_s"]
    per_video = (
        work.groupby("video_id", observed=True)
        .agg(
            segments=("narration_id", "size"),
            listed_fps=("fps", "first"),
            median_ratio=("_ratio", "median"),
        )
        .reset_index()
    )
    near = per_video["median_ratio"].between(0.9, 1.1, inclusive="neither")
    agreeing_ids = set(per_video.loc[near, "video_id"])
    agreeing = work[work["video_id"].isin(agreeing_ids)]
    abs_gap = (agreeing["frame_duration_s"] - agreeing["duration_s"]).abs()
    summary = pd.DataFrame(
        [
            {
                "videos_near_one": int(near.sum()),
                "segments_in_those_videos": int(per_video.loc[near, "segments"].sum()),
                "max_abs_seconds": float(abs_gap.max()) if len(abs_gap) else np.nan,
                "videos_outside": int((~near).sum()),
                "segments_outside": int(per_video.loc[~near, "segments"].sum()),
            }
        ]
    )
    exceptions = per_video.loc[~near, ["video_id", "listed_fps", "segments", "median_ratio"]].copy()
    exceptions = exceptions.sort_values("video_id").reset_index(drop=True)
    exceptions["listed_fps"] = exceptions["listed_fps"].round(2)
    exceptions["median_ratio"] = exceptions["median_ratio"].round(3)
    return summary, exceptions


def _read_split(data_dir: Path, split: str, filename: str) -> pd.DataFrame:
    frame = pd.read_csv(_require(data_dir, filename))
    frame["split"] = split
    frame["labels_in_file"] = split != "test"
    frame["start_s"] = _timestamp_to_seconds(frame["start_timestamp"])
    frame["stop_s"] = _timestamp_to_seconds(frame["stop_timestamp"])
    frame["narration_s"] = _timestamp_to_seconds(frame["narration_timestamp"])
    for column in ("verb_class", "noun_class"):
        if column not in frame.columns:
            frame[column] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
        else:
            frame[column] = frame[column].astype("Int64")
    for column in ("narration", "verb", "noun"):
        if column not in frame.columns:
            frame[column] = pd.Series(pd.NA, index=frame.index, dtype="string")
        else:
            frame[column] = frame[column].astype("string")
    return frame[
        [
            "split",
            "labels_in_file",
            "narration_id",
            "participant_id",
            "video_id",
            "start_s",
            "stop_s",
            "narration_s",
            "start_frame",
            "stop_frame",
            "narration",
            "verb",
            "verb_class",
            "noun",
            "noun_class",
        ]
    ]


def _read_video_info(data_dir: Path) -> pd.DataFrame:
    info = pd.read_csv(_require(data_dir, VIDEO_INFO_FILE))
    if info["video_id"].duplicated().any():
        raise ValueError("EPIC_100_video_info.csv has duplicate video ids.")
    info = info.rename(columns={"duration": "video_duration_s"})
    return info[["video_id", "video_duration_s", "fps", "resolution"]]


def _attach_class_names(segments: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    verbs = pd.read_csv(_require(data_dir, VERB_CLASS_FILE))
    nouns = pd.read_csv(_require(data_dir, NOUN_CLASS_FILE))
    verbs["id"] = verbs["id"].astype("Int64")
    nouns["id"] = nouns["id"].astype("Int64")
    segments = segments.merge(
        verbs[["id", "key"]].rename(columns={"id": "verb_class", "key": "verb_name"}),
        on="verb_class",
        how="left",
    )
    segments = segments.merge(
        nouns[["id", "key"]].rename(columns={"id": "noun_class", "key": "noun_name"}),
        on="noun_class",
        how="left",
    )
    # String dtype keeps a missing class name as a missing value on the test
    # split, rather than as a floating-point NaN from the merge.
    segments["verb_name"] = segments["verb_name"].astype("string")
    segments["noun_name"] = segments["noun_name"].astype("string")
    return segments


def _timestamp_to_seconds(values: pd.Series) -> pd.Series:
    """Parse ``HH:MM:SS.ss`` timestamps. Missing entries stay missing."""
    text = values.astype("string").str.strip()
    parts = text.str.split(":", expand=True)
    if parts.shape[1] != 3:
        raise ValueError("A timestamp is not in HH:MM:SS form.")
    seconds = (
        pd.to_numeric(parts[0], errors="coerce") * 3600
        + pd.to_numeric(parts[1], errors="coerce") * 60
        + pd.to_numeric(parts[2], errors="coerce")
    )
    bad = text.notna() & seconds.isna()
    if bad.any():
        example = text[bad].iloc[0]
        raise ValueError(f"Could not parse timestamp {example!r}.")
    return seconds


def _coverage_row(field: str, split: str, part: pd.DataFrame, column: str, in_file: bool) -> dict:
    segments_n = len(part)
    if not in_file:
        return {
            "field": field,
            "split": split,
            "in_file": False,
            "segments": segments_n,
            "present": pd.NA,
            "missing": pd.NA,
            "share_present": np.nan,
        }
    if pd.api.types.is_string_dtype(part[column]) or part[column].dtype == object:
        text = part[column].astype("string").str.strip()
        present_mask = text.notna() & text.ne("")
    else:
        present_mask = part[column].notna()
    present = int(present_mask.sum())
    return {
        "field": field,
        "split": split,
        "in_file": True,
        "segments": segments_n,
        "present": present,
        "missing": segments_n - present,
        "share_present": present / segments_n,
    }


def _duration_row(split: str, part: pd.DataFrame) -> dict:
    duration = part["duration_s"]
    return {
        "split": split,
        "segments": len(part),
        "nonpositive": int((duration <= 0).sum()),
        "min_s": float(duration.min()),
        "p05_s": float(duration.quantile(0.05)),
        "median_s": float(duration.median()),
        "mean_s": float(duration.mean()),
        "p95_s": float(duration.quantile(0.95)),
        "max_s": float(duration.max()),
        "longer_than_60s": int((duration > 60).sum()),
    }


def _count_row(split: str, part: pd.DataFrame) -> dict:
    return {
        "split": str(split),
        "participants": int(part["participant_id"].nunique()),
        "videos": int(part["video_id"].nunique()),
        "segments": int(len(part)),
    }


def _labeled_subset(segments: pd.DataFrame, split: str) -> pd.DataFrame:
    part = _subset(segments, split)
    part = part[part["labels_in_file"]]
    if part.empty:
        raise ValueError(f"Split {split!r} has no verb or noun labels.")
    return part


def _subset(segments: pd.DataFrame, split: str) -> pd.DataFrame:
    if split == "train+validation":
        return segments[segments["split"].isin(["train", "validation"])]
    if split == "all":
        return segments
    if split not in SPLIT_ORDER:
        raise ValueError(f"Unknown split {split!r}.")
    return segments[segments["split"] == split]


def _label_columns(label: str) -> tuple[str, str]:
    try:
        return LABEL_COLUMNS[label]
    except KeyError as exc:
        raise ValueError("label must be 'verb' or 'noun'.") from exc


def _require(data_dir: Path, filename: str) -> Path:
    path = data_dir / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Download the annotation tables from {ANNOTATION_SOURCE}."
        )
    return path
