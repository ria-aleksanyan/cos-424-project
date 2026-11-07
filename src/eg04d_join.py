"""Join Ego4D Forecasting Hands and Objects verbs to narration sentences.

The long-term anticipation files store a verb, a noun, and a time window.
The sentence is in ``narrations.json``. Each collapsed action is joined to
one narration whose timestamp falls inside that window. A row with two
sentences about equally close to the action is marked ambiguous and left
out of the scoring set.

The official files are released through the Ego4D CLI after the license
agreement at https://ego4d-data.org/. This module reads local paths and
can invoke that CLI when the files are missing.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Two inside-window sentences this close in distance are not one action.
AMBIGUOUS_GAP_S = 0.75

_TOKEN = re.compile(r"[a-z0-9']+")
_CAMERA_WEARER = re.compile(r"^(?:(?:#+\s*)?c\b[\s,.:;-]*)+", re.I)
_IRREGULAR = {
    "take": ("took", "taken"),
    "leave": ("left",),
    "get": ("got", "gotten"),
    "hold": ("held",),
    "throw": ("threw", "thrown"),
    "eat": ("ate", "eaten"),
    "make": ("made",),
    "give": ("gave", "given"),
    "come": ("came",),
    "go": ("went", "gone"),
    "sit": ("sat",),
    "stand": ("stood",),
    "write": ("wrote", "written"),
    "break": ("broke", "broken"),
    "speak": ("spoke", "spoken"),
    "choose": ("chose", "chosen"),
    "drive": ("drove", "driven"),
    "fall": ("fell", "fallen"),
    "feel": ("felt",),
    "keep": ("kept",),
    "lose": ("lost",),
    "run": ("ran",),
    "say": ("said",),
    "sell": ("sold",),
    "send": ("sent",),
    "sleep": ("slept",),
    "teach": ("taught",),
    "tell": ("told",),
    "think": ("thought",),
    "wear": ("wore", "worn"),
    "win": ("won",),
    "cut": ("cut",),
    "put": ("put",),
    "read": ("read",),
    "do": ("does", "did", "done"),
    "have": ("has", "had"),
}

_ACTIONS_TRAIN = "fho_lta_train.json"
_ACTIONS_VAL = "fho_lta_val.json"
_NARRATIONS = "narrations.json"
_MANIFEST = "ego4d.json"

_OFFICIAL_DOWNLOAD = (
    "python -m ego4d.cli.cli --output_directory DATA "
    "--datasets annotations --benchmarks FHO"
)


@dataclass(frozen=True)
class AnnotationPaths:
    """Local paths for the two splits, the narrations, and the video manifest."""

    actions_train: Path
    actions_val: Path
    narrations: Path
    manifest: Path


@dataclass(frozen=True)
class JoinCounts:
    """Counts the proposal uses to show the join can be scored.

    ``total_rows`` is the collapsed action list. ``rows_surviving_join``
    matched one narration and were not ambiguous. ``ambiguous_rows_dropped``
    matched more than one equally close narration and are excluded from
    scoring. ``val_participant_in_train`` counts surviving validation rows
    whose ``fb_participant_id`` also appears on a training action.
    """

    total_rows: int
    rows_surviving_join: int
    ambiguous_rows_dropped: int
    val_participant_in_train: int


def tokenize(text: object) -> list[str]:
    """Lowercase a string and split it into words."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []
    return _TOKEN.findall(str(text).lower().replace("-", " ").replace("_", " "))


def verb_surfaces(label: object) -> set[str]:
    """Split a taxonomy label into the words that should leave the sentence.

    ``take_(pick,_grab,_get)`` contributes take, pick, grab, and get, plus
    the inflected and irregular forms of each.
    """
    forms: set[str] = set()
    for lemma in tokenize(label):
        forms.add(lemma)
        forms.update(_IRREGULAR.get(lemma, ()))
        forms.add(lemma + "s")
        forms.add(lemma + "ed")
        forms.add(lemma + "ing")
        if lemma.endswith("e"):
            forms.add(lemma[:-1] + "ing")
            forms.add(lemma + "d")
        if len(lemma) >= 3:
            forms.add(lemma + lemma[-1] + "ing")
            forms.add(lemma + lemma[-1] + "ed")
    return forms


def strip_camera_wearer(text: object) -> str:
    """Remove a leading ``#C`` / ``C`` camera-wearer tag."""
    return _CAMERA_WEARER.sub("", str(text).strip())


def strip_verb(text: object, verb_label: object) -> list[str]:
    """Delete the labeled verb and its forms. Return the leftover words."""
    banned = verb_surfaces(verb_label)
    return [word for word in tokenize(strip_camera_wearer(text)) if word not in banned]


def verb_remains(leftover: object, verb_label: object) -> bool:
    """True when a form of the labeled verb is still in the leftover.

    A true value means the strip did not remove the verb.
    """
    banned = verb_surfaces(verb_label)
    return any(word in banned for word in tokenize(leftover))


def _read_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_actions(path: Path | str, split: str) -> pd.DataFrame:
    """Load one ``fho_lta_*.json`` and collapse repeated forecast rows.

    The same physical action is listed once for every future step. One row
    is kept per video, frame span, and verb.
    """
    path = Path(path)
    payload = _read_json(path)
    clips = payload["clips"] if isinstance(payload, dict) else payload
    columns = [
        "split",
        "video_uid",
        "verb",
        "verb_label",
        "noun",
        "start_sec",
        "end_sec",
        "start_frame",
        "end_frame",
    ]
    rows = [
        {
            "split": split,
            "video_uid": clip["video_uid"],
            "verb": clip["verb"],
            "verb_label": clip["verb_label"],
            "noun": clip.get("noun"),
            "start_sec": clip["action_clip_start_sec"],
            "end_sec": clip["action_clip_end_sec"],
            "start_frame": clip.get("action_clip_start_frame"),
            "end_frame": clip.get("action_clip_end_frame"),
        }
        for clip in clips
    ]
    actions = pd.DataFrame(rows, columns=columns)
    if actions.empty:
        return actions
    return (
        actions.drop_duplicates(
            ["video_uid", "start_frame", "end_frame", "verb_label", "noun"]
        )
        .sort_values(["video_uid", "start_sec", "verb_label"], kind="stable")
        .reset_index(drop=True)
    )


def load_narrations(path: Path | str) -> pd.DataFrame:
    """Load ``narrations.json``. Use the revised pass when that video has one."""
    path = Path(path)
    raw = _read_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} is not a narration dict keyed by video uid.")
    rows: list[dict] = []
    for video_uid, passes in raw.items():
        if not isinstance(passes, dict):
            continue
        revised = passes.get("narration_pass_2", {}).get("narrations") or []
        original = passes.get("narration_pass_1", {}).get("narrations") or []
        chosen = revised or original
        pass_name = "narration_pass_2" if revised else "narration_pass_1"
        for note in chosen:
            rows.append(
                {
                    "video_uid": video_uid,
                    "timestamp_sec": note["timestamp_sec"],
                    "narration_text": note.get("narration_text", ""),
                    "annotation_uid": note.get("annotation_uid"),
                    "narration_pass": pass_name,
                }
            )
    return pd.DataFrame(rows)


def load_participants(path: Path | str) -> pd.DataFrame:
    """Load ``video_uid`` and ``fb_participant_id`` from the Ego4D manifest."""
    path = Path(path)
    raw = _read_json(path)
    videos = raw["videos"] if isinstance(raw, dict) and "videos" in raw else raw
    frame = pd.DataFrame(videos)
    keep = frame[["video_uid", "fb_participant_id"]].drop_duplicates("video_uid")
    return keep.reset_index(drop=True)


def join_actions(actions: pd.DataFrame, narrations: pd.DataFrame) -> pd.DataFrame:
    """Attach the narration whose timestamp falls inside the action window.

    A sentence outside the window is not used. When the closest two sentences
    inside the window are within ``AMBIGUOUS_GAP_S`` seconds of each other,
    ``ambiguous`` is true and the row is excluded from scoring.
    """
    grouped = {
        video_uid: group.sort_values("timestamp_sec")
        for video_uid, group in narrations.groupby("video_uid", sort=False)
    }
    kept: list[dict] = []
    for action in actions.itertuples(index=False):
        notes = grouped.get(action.video_uid)
        if notes is None:
            continue
        inside = notes[
            (notes["timestamp_sec"] >= action.start_sec)
            & (notes["timestamp_sec"] <= action.end_sec)
        ]
        if inside.empty:
            continue
        midpoint = (action.start_sec + action.end_sec) / 2.0
        ranked = inside.assign(dist=(inside["timestamp_sec"] - midpoint).abs())
        ranked = ranked.sort_values(["dist", "timestamp_sec"], kind="stable")
        best = ranked.iloc[0]
        ambiguous = len(ranked) > 1 and (ranked.iloc[1]["dist"] - best["dist"]) < AMBIGUOUS_GAP_S
        sentence = strip_camera_wearer(best["narration_text"])
        leftover = strip_verb(sentence, action.verb)
        leftover_text = " ".join(leftover)
        kept.append(
            {
                "split": action.split,
                "video_uid": action.video_uid,
                "verb": action.verb,
                "verb_label": action.verb_label,
                "noun": action.noun,
                "narration": sentence,
                "leftover": leftover_text,
                "n_leftover": len(leftover),
                "ambiguous": bool(ambiguous),
                "verb_remains": verb_remains(leftover_text, action.verb),
                "n_candidates": int(len(ranked)),
                "annotation_uid": best["annotation_uid"],
                "narration_pass": best["narration_pass"],
            }
        )
    return pd.DataFrame(kept)


def attach_participants(notes: pd.DataFrame, participants: pd.DataFrame) -> pd.DataFrame:
    """Add ``fb_participant_id`` and whether that person also appears in train."""
    merged = notes.merge(participants, on="video_uid", how="left", validate="many_to_one")
    train_ids = set(
        merged.loc[merged["split"] == "train", "fb_participant_id"].dropna()
    )
    merged["participant_in_train"] = merged["fb_participant_id"].isin(train_ids)
    return merged


def scoring_rows(notes: pd.DataFrame) -> pd.DataFrame:
    """Rows that joined to one narration and were not marked ambiguous."""
    if notes.empty:
        return notes.copy()
    return notes.loc[~notes["ambiguous"]].reset_index(drop=True)


def join_counts(actions: pd.DataFrame, notes: pd.DataFrame) -> JoinCounts:
    """Count collapsed actions, surviving joins, dropped ambiguous rows, and overlap."""
    if notes.empty:
        ambiguous = 0
        surviving = notes
    else:
        ambiguous = int(notes["ambiguous"].sum())
        surviving = notes.loc[~notes["ambiguous"]]
    val = surviving.loc[surviving["split"] == "val"] if not surviving.empty else surviving
    overlap = int(val["participant_in_train"].sum()) if not val.empty else 0
    return JoinCounts(
        total_rows=int(len(actions)),
        rows_surviving_join=int(len(surviving)),
        ambiguous_rows_dropped=ambiguous,
        val_participant_in_train=overlap,
    )


def _first(root: Path, name: str) -> Path | None:
    matches = sorted(root.rglob(name))
    return matches[0] if matches else None


def resolve_annotation_paths(data_dir: Path | str) -> AnnotationPaths:
    """Find the FHO action files, narrations, and manifest under ``data_dir``."""
    root = Path(data_dir)
    train = _first(root, _ACTIONS_TRAIN)
    val = _first(root, _ACTIONS_VAL)
    narrations = _first(root, _NARRATIONS)
    manifest = _first(root, _MANIFEST)
    missing = [
        name
        for name, path in (
            (_ACTIONS_TRAIN, train),
            (_ACTIONS_VAL, val),
            (_NARRATIONS, narrations),
            (_MANIFEST, manifest),
        )
        if path is None
    ]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            f"{root} is missing {joined}. Download the official annotations with "
            f"`{_OFFICIAL_DOWNLOAD}` after accepting the license at https://ego4d-data.org/."
        )
    return AnnotationPaths(train, val, narrations, manifest)


def ensure_annotations(data_dir: Path | str, *, download: bool = False) -> AnnotationPaths:
    """Return local annotation paths, downloading with the official CLI if asked.

    ``download=True`` runs ``python -m ego4d.cli.cli`` into ``data_dir``.
    The CLI requires the Ego4D license credentials. This function does not
    fetch the files any other way.
    """
    root = Path(data_dir)
    try:
        return resolve_annotation_paths(root)
    except FileNotFoundError:
        if not download:
            raise
    root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "ego4d.cli.cli",
        f"--output_directory={root}",
        "--datasets",
        "annotations",
        "--benchmarks",
        "FHO",
    ]
    subprocess.run(command, check=True)
    return resolve_annotation_paths(root)


def build_joined_table(
    data_dir: Path | str,
    *,
    download: bool = False,
) -> tuple[pd.DataFrame, JoinCounts]:
    """Join both splits and return the table plus the four counts.

    Ambiguous rows stay in the table with ``ambiguous`` true. ``scoring_rows``
    drops them.
    """
    paths = ensure_annotations(data_dir, download=download)
    actions = pd.concat(
        [
            load_actions(paths.actions_train, "train"),
            load_actions(paths.actions_val, "val"),
        ],
        ignore_index=True,
    )
    notes = join_actions(actions, load_narrations(paths.narrations))
    notes = attach_participants(notes, load_participants(paths.manifest))
    return notes, join_counts(actions, notes)
