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
_NARRATIONS = ("narrations.json", "narration.json")
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


def _first(root: Path, *names: str) -> Path | None:
    for name in names:
        matches = sorted(root.rglob(name))
        if matches:
            return matches[0]
    return None


def resolve_annotation_paths(data_dir: Path | str) -> AnnotationPaths:
    """Find the FHO action files, narrations, and manifest under ``data_dir``."""
    root = Path(data_dir)
    train = _first(root, _ACTIONS_TRAIN)
    val = _first(root, _ACTIONS_VAL)
    narrations = _first(root, *_NARRATIONS)
    manifest = _first(root, _MANIFEST)
    missing = [
        name
        for name, path in (
            (_ACTIONS_TRAIN, train),
            (_ACTIONS_VAL, val),
            ("narration.json", narrations),
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


def descriptive_statistics(notes: pd.DataFrame) -> pd.Series:
    """Summarize the joined table: size, videos, people, verbs, leftover length."""
    if notes.empty:
        return pd.Series(
            {
                "joined_rows": 0,
                "scoring_rows": 0,
                "videos": 0,
                "participants": 0,
                "verbs": 0,
                "train_rows": 0,
                "val_rows": 0,
                "median_leftover_words": 0.0,
                "mean_leftover_words": 0.0,
            },
            name="value",
        )
    scoring = scoring_rows(notes)
    participants = (
        notes["fb_participant_id"].nunique(dropna=True)
        if "fb_participant_id" in notes
        else 0
    )
    return pd.Series(
        {
            "joined_rows": int(len(notes)),
            "scoring_rows": int(len(scoring)),
            "videos": int(notes["video_uid"].nunique()),
            "participants": int(participants),
            "verbs": int(scoring["verb"].nunique()) if not scoring.empty else 0,
            "train_rows": int((notes["split"] == "train").sum()),
            "val_rows": int((notes["split"] == "val").sum()),
            "median_leftover_words": float(scoring["n_leftover"].median())
            if not scoring.empty
            else 0.0,
            "mean_leftover_words": float(scoring["n_leftover"].mean())
            if not scoring.empty
            else 0.0,
        },
        name="value",
    )


def verb_frequency(notes: pd.DataFrame) -> pd.Series:
    """Count joined rows per verb label, most common first."""
    if notes.empty or "verb" not in notes.columns:
        return pd.Series(dtype="int64", name="actions")
    counts = notes["verb"].value_counts()
    counts.name = "actions"
    return counts


def narrations_per_video(narrations: pd.DataFrame) -> pd.Series:
    """Count narration sentences in each video."""
    if narrations.empty:
        return pd.Series(dtype="int64", name="narrations")
    counts = narrations.groupby("video_uid", sort=False).size()
    counts.name = "narrations"
    return counts.sort_values(ascending=False)


def narration_video_summary(per_video: pd.Series) -> pd.Series:
    """How many sentences the narration file has, and how they spread across videos."""
    if per_video.empty:
        return pd.Series(
            {
                "videos": 0,
                "narration_sentences": 0,
                "median_per_video": 0.0,
                "max_per_video": 0,
            },
            name="value",
        )
    return pd.Series(
        {
            "videos": int(per_video.shape[0]),
            "narration_sentences": int(per_video.sum()),
            "median_per_video": float(per_video.median()),
            "max_per_video": int(per_video.max()),
        },
        name="value",
    )


def preprocessing_summary(notes: pd.DataFrame) -> pd.Series:
    """Report the strip and the ambiguous-join filter on the joined table."""
    if notes.empty:
        return pd.Series(
            {
                "rows_with_narration": 0,
                "ambiguous_rows": 0,
                "rows_kept_for_scoring": 0,
                "verb_still_in_leftover": 0,
                "empty_leftovers": 0,
                "median_leftover_words": 0.0,
            },
            name="value",
        )
    return pd.Series(
        {
            "rows_with_narration": int(len(notes)),
            "ambiguous_rows": int(notes["ambiguous"].sum()),
            "rows_kept_for_scoring": int((~notes["ambiguous"]).sum()),
            "verb_still_in_leftover": int(notes["verb_remains"].sum()),
            "empty_leftovers": int((notes["n_leftover"] == 0).sum()),
            "median_leftover_words": float(notes["n_leftover"].median()),
        },
        name="value",
    )


def preprocessing_examples(notes: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """Show the verb, the sentence after the camera-wearer tag is removed, and the leftover."""
    columns = ["split", "verb", "narration", "leftover", "verb_remains", "ambiguous"]
    if notes.empty:
        return pd.DataFrame(columns=columns)
    return notes.loc[:, columns].head(n).reset_index(drop=True)


def counts_table(counts: JoinCounts) -> pd.DataFrame:
    """The four join counts as a one-column table."""
    return pd.DataFrame(
        {
            "count": [
                counts.total_rows,
                counts.rows_surviving_join,
                counts.ambiguous_rows_dropped,
                counts.val_participant_in_train,
            ]
        },
        index=[
            "collapsed actions",
            "rows surviving the join",
            "ambiguous rows dropped",
            "validation rows whose participant is also in training",
        ],
    )


def plot_verb_frequency(notes: pd.DataFrame, top_n: int = 25, title: str | None = None):
    """Bar chart of verbs in a joined table."""
    import matplotlib.pyplot as plt

    freq = verb_frequency(notes).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    freq.plot.barh(ax=ax)
    ax.set_xlabel("Actions")
    ax.set_ylabel("")
    ax.set_title(title or "Verbs after the narration join")
    fig.tight_layout()
    return fig


def head_verb(label: object) -> str:
    """The taxonomy head: the word before the parenthesis."""
    words = tokenize(str(label).split("_(")[0])
    return words[0] if words else ""


def without_shared_videos(
    notes: pd.DataFrame,
    data_dir: Path | str,
) -> tuple[pd.DataFrame, JoinCounts, pd.Series]:
    """Drop every video that appears in both the train and validation files.

    Participant overlap is recomputed on the rows that remain. The returned
    counts use the collapsed actions on the videos that were kept.
    """
    paths = resolve_annotation_paths(data_dir)
    train_actions = load_actions(paths.actions_train, "train")
    val_actions = load_actions(paths.actions_val, "val")
    both = set(train_actions["video_uid"]) & set(val_actions["video_uid"])
    actions = pd.concat([train_actions, val_actions], ignore_index=True)
    actions = actions.loc[~actions["video_uid"].isin(both)].reset_index(drop=True)
    cleaned = notes.loc[~notes["video_uid"].isin(both)].copy()
    train_ids = set(cleaned.loc[cleaned["split"] == "train", "fb_participant_id"].dropna())
    cleaned["participant_in_train"] = cleaned["fb_participant_id"].isin(train_ids)
    kept_train = set(cleaned.loc[cleaned["split"] == "train", "video_uid"])
    kept_val = set(cleaned.loc[cleaned["split"] == "val", "video_uid"])
    report = pd.Series(
        {
            "videos_in_both_files": int(len(both)),
            "videos_removed": int(len(both)),
            "train_videos_remaining": int(len(kept_train)),
            "val_videos_remaining": int(len(kept_val)),
            "videos_still_in_both": int(len(kept_train & kept_val)),
        },
        name="value",
    )
    return cleaned.reset_index(drop=True), join_counts(actions, cleaned), report


def shared_verb_rows(notes: pd.DataFrame, shared: list[str] | set[str]) -> pd.DataFrame:
    """Unambiguous rows whose taxonomy head is one of the shared verbs.

    The ``verb`` column is replaced with that head, so later counts use the
    37 names rather than the full taxonomy string.
    """
    shared_set = {str(verb) for verb in shared}
    scoring = scoring_rows(notes)
    if scoring.empty:
        return scoring
    heads = scoring["verb"].map(head_verb)
    kept = scoring.loc[heads.isin(shared_set)].copy()
    kept["verb"] = heads.loc[heads.isin(shared_set)].to_numpy()
    return kept.reset_index(drop=True)


def shared_verb_frequency(notes: pd.DataFrame, shared: list[str] | set[str]) -> pd.Series:
    """Count the shared verbs, including a shared verb that has no remaining rows."""
    order = [str(verb) for verb in shared]
    rows = shared_verb_rows(notes, order)
    if rows.empty:
        counts = pd.Series(0, index=order, dtype="int64")
    else:
        counts = rows["verb"].value_counts().reindex(order, fill_value=0).astype("int64")
    counts = counts.sort_values(ascending=False)
    counts.name = "actions"
    return counts


def plot_shared_verbs(notes: pd.DataFrame, shared: list[str] | set[str]):
    """Bar chart of the shared verbs only."""
    import matplotlib.pyplot as plt

    freq = shared_verb_frequency(notes, shared).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 8))
    freq.plot.barh(ax=ax)
    ax.set_xlabel("Actions")
    ax.set_ylabel("")
    ax.set_title("37 shared verbs in the leakage-free scoring set")
    fig.tight_layout()
    return fig


def _hand_label_paths(task_dir: Path) -> list[Path]:
    candidates = [
        task_dir / "hand_labels.csv",
        task_dir.parent / "hand_labels.csv",
        Path("data") / "agibot_hand_labels.csv",
    ]
    return [path for path in candidates if path.is_file()]


def hand_label_table(task_dir: Path | str) -> pd.DataFrame:
    """Commands waiting for two independent verb labels.

    No verb is guessed from the skill name or from the words in the command.
    If a hand-label file is present, its ``ria_verb`` and ``naomi_verb``
    columns are attached on ``task_id``, ``episode_id``, and ``action_text``.
    Otherwise those columns are blank. ``agreed_verb`` is filled only where
    both labels are present and equal.
    """
    commands = pd.DataFrame(_agibot_actions(task_dir))
    blank = ["ria_verb", "naomi_verb", "agreed_verb"]
    if commands.empty:
        return pd.DataFrame(columns=["task_id", "episode_id", "action_text", *blank])
    commands = commands.loc[:, ["task_id", "episode_id", "action_text"]].copy()
    for column in blank:
        commands[column] = ""
    label_files = _hand_label_paths(Path(task_dir))
    if not label_files:
        return commands
    labels = pd.read_csv(label_files[0], dtype=str).fillna("")
    needed = {"task_id", "episode_id", "action_text", "ria_verb", "naomi_verb"}
    if not needed.issubset(labels.columns):
        return commands
    labels = labels.loc[:, ["task_id", "episode_id", "action_text", "ria_verb", "naomi_verb"]]
    commands = commands.drop(columns=["ria_verb", "naomi_verb"]).merge(
        labels,
        on=["task_id", "episode_id", "action_text"],
        how="left",
    )
    commands["ria_verb"] = commands["ria_verb"].fillna("")
    commands["naomi_verb"] = commands["naomi_verb"].fillna("")
    both = commands["ria_verb"].str.len().gt(0) & commands["naomi_verb"].str.len().gt(0)
    same = commands["ria_verb"].str.lower().eq(commands["naomi_verb"].str.lower())
    commands["agreed_verb"] = ""
    commands.loc[both & same, "agreed_verb"] = commands.loc[both & same, "ria_verb"].str.lower()
    return commands


def hand_label_counts(task_dir: Path | str) -> pd.Series:
    """How many commands have real hand labels, and how many are still blank."""
    table = hand_label_table(task_dir)
    label_file = _hand_label_paths(Path(task_dir))
    if table.empty:
        both = agreed = disagree = 0
        files = 0
    else:
        both_mask = table["ria_verb"].str.len().gt(0) & table["naomi_verb"].str.len().gt(0)
        both = int(both_mask.sum())
        agreed = int(table["agreed_verb"].str.len().gt(0).sum())
        disagree = both - agreed
        files = int(table["task_id"].nunique())
    return pd.Series(
        {
            "task_files": files,
            "command_rows": int(len(table)),
            "hand_label_file": str(label_file[0]) if label_file else "none",
            "rows_labeled_by_both": both,
            "agreements": agreed,
            "disagreements": disagree,
            "rows_still_unlabeled": int(len(table) - both),
        },
        name="value",
    )


def hand_label_examples(task_dir: Path | str, n: int = 8) -> pd.DataFrame:
    """The first commands, with the hand-label columns blank when no file exists."""
    columns = ["task_id", "action_text", "ria_verb", "naomi_verb", "agreed_verb"]
    table = hand_label_table(task_dir)
    if table.empty:
        return pd.DataFrame(columns=columns)
    return table.loc[:, columns].head(n).reset_index(drop=True)


def _metrics(y_true: pd.Series, y_pred: object) -> tuple[float, float]:
    from sklearn.metrics import accuracy_score, f1_score

    present = sorted(set(y_true.astype(str)))
    if len(y_true) == 0 or not present:
        return float("nan"), float("nan")
    return (
        float(accuracy_score(y_true, y_pred)),
        float(f1_score(y_true, y_pred, average="macro", labels=present, zero_division=0)),
    )


def fit_verb_labelers(
    notes: pd.DataFrame,
    shared: list[str] | set[str],
) -> tuple[pd.Series, pd.DataFrame]:
    """Fit the three labelers on the shared-verb training rows and score validation.

    The word-count model is TF-IDF unigrams and bigrams plus logistic regression.
    The embedding model is frozen all-MiniLM-L6-v2 plus a second logistic
    regression. The rule guesses a shared verb only when exactly one of those
    verbs is still written in the leftover; otherwise it guesses the most
    common of the shared verbs in training. Put is not treated as a special
    case, because put is not in the shared list.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sentence_transformers import SentenceTransformer

    shared_list = [str(verb) for verb in shared]
    frame = shared_verb_rows(notes, shared_list)
    train = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    val = frame.loc[frame["split"] == "val"].reset_index(drop=True)
    if train.empty or val.empty:
        raise ValueError("The shared-verb split has no training or no validation rows.")
    floor = str(train["verb"].value_counts().idxmax())
    surfaces: dict[str, set[str]] = {}
    for verb in shared_list:
        for form in verb_surfaces(verb):
            surfaces.setdefault(form, set()).add(verb)

    def rule_one(text: object) -> str:
        matched: set[str] = set()
        for word in tokenize(text):
            matched |= surfaces.get(word, set())
        if len(matched) == 1:
            return next(iter(matched))
        return floor

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    x_train = vectorizer.fit_transform(train["leftover"].fillna(""))
    x_val = vectorizer.transform(val["leftover"].fillna(""))
    word_model = LogisticRegression(max_iter=400, random_state=0)
    word_model.fit(x_train, train["verb"])
    word_pred = word_model.predict(x_val)

    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb_train = encoder.encode(
        train["leftover"].fillna("").tolist(),
        batch_size=128,
        show_progress_bar=False,
    )
    emb_val = encoder.encode(
        val["leftover"].fillna("").tolist(),
        batch_size=128,
        show_progress_bar=False,
    )
    embedding_model = LogisticRegression(max_iter=400, random_state=0)
    embedding_model.fit(emb_train, train["verb"])
    embedding_pred = embedding_model.predict(emb_val)
    rule_pred = val["leftover"].map(rule_one).to_numpy()

    masks = {
        "all validation rows": pd.Series(True, index=val.index),
        "participant overlap removed": ~val["participant_in_train"].astype(bool),
    }
    predictions = {
        "word count": word_pred,
        "sentence embedding": embedding_pred,
        "rule": rule_pred,
    }
    rows: list[dict] = []
    for program, pred in predictions.items():
        for subset, mask in masks.items():
            accuracy, macro_f1 = _metrics(val.loc[mask, "verb"], pred[mask.to_numpy()])
            rows.append(
                {
                    "program": program,
                    "rows": subset,
                    "n": int(mask.sum()),
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                }
            )
    summary = pd.Series(
        {
            "shared_verbs": int(len(shared_list)),
            "train_rows": int(len(train)),
            "val_rows": int(len(val)),
            "floor_verb": floor,
            "verbs_in_validation": int(val["verb"].nunique()),
        },
        name="value",
    )
    scores = pd.DataFrame(rows)
    return summary, scores


def head_lemmas(labels: object) -> set[str]:
    """The word before the parenthesis in each taxonomy label.

    ``take_(pick,_grab,_get)`` contributes take. The words in parentheses
    are synonyms, not separate classes.
    """
    heads: set[str] = set()
    for label in labels if isinstance(labels, (list, tuple)) else []:
        words = tokenize(str(label).split("_(")[0])
        if words:
            heads.add(words[0])
    return heads


def _taxonomy_verbs(data_dir: Path | str) -> list[str]:
    path = _first(Path(data_dir), "fho_lta_taxonomy.json")
    if path is None:
        raise FileNotFoundError(f"{data_dir} has no fho_lta_taxonomy.json.")
    payload = _read_json(path)
    if not isinstance(payload, dict) or "verbs" not in payload:
        raise ValueError(f"{path} has no verb list.")
    return list(payload["verbs"])


def _agibot_actions(task_dir: Path | str) -> list[dict]:
    root = Path(task_dir)
    rows: list[dict] = []
    for path in sorted(root.glob("*.json")):
        payload = _read_json(path)
        if not isinstance(payload, list):
            continue
        task_id = path.stem.removeprefix("task_")
        for episode in payload:
            if not isinstance(episode, dict):
                continue
            info = episode.get("label_info") or episode.get("lable_info") or {}
            actions = info.get("action_config") or []
            for action in actions:
                if not isinstance(action, dict):
                    continue
                rows.append(
                    {
                        "task_id": task_id,
                        "episode_id": episode.get("episode_id"),
                        "task_name": episode.get("task_name", ""),
                        "skill": str(action.get("skill") or "").strip(),
                        "action_text": str(action.get("action_text") or "").strip(),
                    }
                )
    return rows


def shared_verbs(ego4d_dir: Path | str, task_dir: Path | str) -> list[str]:
    """Ego4D head verbs whose name is also an AgiBot skill.

    This is the frozen shared list: the head word of a taxonomy label, matched
    to a skill name, with no synonym expansion.
    """
    heads = head_lemmas(_taxonomy_verbs(ego4d_dir))
    skills = {row["skill"].lower() for row in _agibot_actions(task_dir) if row["skill"]}
    return sorted(heads & skills)


def load_agibot_commands(
    task_dir: Path | str,
    shared: list[str] | set[str],
) -> pd.DataFrame:
    """One row per AgiBot action command, with the verb removed for labeling.

    ``shared_verb`` is the skill when that skill is one of the shared verbs.
    Otherwise it is the single shared verb whose form appears in the sentence.
    A sentence that matches several shared verbs, or none, is left unmapped.
    """
    shared_set = {str(verb).lower() for verb in shared}
    surfaces: dict[str, set[str]] = {}
    for verb in shared_set:
        for form in verb_surfaces(verb):
            surfaces.setdefault(form, set()).add(verb)
    rows: list[dict] = []
    for action in _agibot_actions(task_dir):
        skill = action["skill"].lower()
        tokens = tokenize(action["action_text"])
        matched: set[str] = set()
        for word in tokens:
            matched |= surfaces.get(word, set())
        if skill in shared_set:
            mapped = skill
        elif len(matched) == 1:
            mapped = next(iter(matched))
        else:
            mapped = ""
        banned = verb_surfaces(skill) if skill else set()
        for verb in matched:
            banned |= verb_surfaces(verb)
        if mapped:
            banned |= verb_surfaces(mapped)
        leftover = [word for word in tokens if word not in banned]
        leftover_text = " ".join(leftover)
        rows.append(
            {
                **action,
                "shared_verb": mapped,
                "leftover": leftover_text,
                "verb_remains": bool(mapped) and verb_remains(leftover_text, mapped),
            }
        )
    return pd.DataFrame(rows)


def agibot_counts(commands: pd.DataFrame, shared: list[str] | set[str]) -> pd.Series:
    """How many commands were extracted, and how many map onto the shared verbs."""
    if commands.empty:
        mapped = commands
        n_mapped = 0
        n_verbs = 0
        n_files = 0
        still = 0
    else:
        mapped_mask = commands["shared_verb"].astype(str).ne("")
        mapped = commands.loc[mapped_mask]
        n_mapped = int(mapped_mask.sum())
        n_verbs = int(mapped["shared_verb"].nunique())
        n_files = int(commands["task_id"].nunique())
        still = int(commands["verb_remains"].sum())
    return pd.Series(
        {
            "task_files": n_files,
            "command_rows": int(len(commands)),
            "shared_verb_list": int(len(set(shared))),
            "commands_mapped": n_mapped,
            "commands_unmapped": int(len(commands) - n_mapped),
            "verbs_with_a_command": n_verbs,
            "verb_still_in_leftover": still,
        },
        name="value",
    )


def agibot_examples(commands: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """Show the command, the leftover, and the shared verb it mapped to."""
    columns = ["task_id", "skill", "action_text", "leftover", "shared_verb", "verb_remains"]
    if commands.empty:
        return pd.DataFrame(columns=columns)
    return commands.loc[:, columns].head(n).reset_index(drop=True)


def agibot_plot_frame(commands: pd.DataFrame) -> pd.DataFrame:
    """Mapped commands in the column shape used by ``plot_verb_frequency``."""
    if commands.empty:
        return pd.DataFrame(columns=["verb"])
    mapped = commands.loc[commands["shared_verb"].astype(str).ne(""), ["shared_verb"]]
    return mapped.rename(columns={"shared_verb": "verb"}).reset_index(drop=True)


def plot_narrations_per_video(per_video: pd.Series):
    """Histogram of narration sentences per video."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.2))
    if not per_video.empty:
        bins = min(30, max(int(per_video.nunique()), 1))
        ax.hist(per_video.to_numpy(), bins=bins, edgecolor="white")
    ax.set_xlabel("Narration sentences")
    ax.set_ylabel("Videos")
    ax.set_title("Narrations per video")
    fig.tight_layout()
    return fig
