# What an Individual Kitchen Recording Contains

## Abstract

Household tasks such as pouring, opening, and moving objects supply training data for egocentric vision and robot learning, but the recordings are assembled in different ways. EPIC-KITCHENS-100 is filmed by 37 people in 45 of their own kitchens with a head-mounted camera, then narrated by those same people while they watch the footage. DROID records object manipulation on a standardized robot rig with three cameras, depth, calibration, joint and gripper readings, written instructions, and a success or failure mark. This project asks which parts of a usable clip are already present when one person records at home, which are notes a person can add, and which appear only because a lab rig is in the room. A usable clip lets someone who was not present tell the task, when it started and ended, and what happened. We will answer with one component table, built from the EPIC-KITCHENS annotation files and from DROID’s published episode specification.

## Research questions

For household tasks, which parts of a usable training clip are present when one person records at home, and which parts appear only in a robot lab rig?

Three smaller questions keep the comparison specific. Which sensors and notes does a home kitchen recording already contain, and how often are those notes filled in? Which additional sensors and notes does a robot-lab episode contain for the same kind of object task? Of the parts that appear only in the lab, which are notes a person could supply, and which require hardware a person at home does not have?

Both datasets are used as training examples of physical object tasks, and the fields they store differ. A field filled in throughout one dataset and absent from the other is a difference in what a reusable recording contains. The comparison separates three limits of individual collection: content already in the home recording, notes the person can add, and signals that exist only when the rig is present.

## Dataset

The primary dataset is EPIC-KITCHENS-100 (Damen et al., 2022): 100 hours of unscripted kitchen activity, 700 videos, and 89,977 action segments from 37 participants in 45 kitchens. Capture uses a head-mounted GoPro and includes audio. After filming, each participant narrates actions with a pause-and-talk procedure, and the release provides start and stop times for each segment.

We will analyze the public annotation tables. Each row has a participant id, video id, narration, verb, noun, narration timestamp, and start and stop timestamps and frames. From these files we will count participants, videos, and segments, measure how often narration and timestamps are present, summarize action length, and list the most common verbs and nouns. The collection paper supplies the sensor side of the same clips.

## Additional datasets

DROID (Khazatsky et al., 2024) is the lab comparison: about 76,000 teleoperated episodes, roughly 350 hours, across 564 scenes. Every episode is specified to include three synchronized stereo cameras, depth, camera calibration, joint and gripper state, up to three natural-language instructions, and a success or failure label. We will code that specification into the same component list, and check the small public metadata: the language-annotation file and the success and failure split. We will leave the video and depth unloaded.

Ego4D is optional. A third column can test whether the home pattern holds outside kitchens, and how often sensors such as motion or gaze were recorded. The question stays the same if Ego4D is left out.

## Methods

Each row of the comparison table is one part of a clip: a head-mounted camera, scene audio, a task description, a verb and noun, a start and end, a success mark, extra camera views, depth, calibration, and joint and gripper readings. For each row we record whether EPIC-KITCHENS contains it, whether DROID contains it, how complete it is where it exists, and whether producing it takes a note from the person or hardware the person does not have.

On the EPIC tables we will report counts of participants, videos, and segments, the share of empty narrations and missing timestamps, the distribution of segment duration, and verb and noun frequencies. DROID’s side of the table is presence by specification, checked against metadata. Rows filled in throughout EPIC-KITCHENS form the home set. Rows present only in DROID are then split by how they are produced: a success mark or a written task can be supplied by the person, while depth, calibration, and joint readings come from the rig. We will repeat the coverage statistics on the train, validation, and test splits and report how uneven the verbs are. Code will live in a Python module called from one Jupyter notebook, which will show the counts, the table, and one figure.

## Timeline

| Dates | Work |
| --- | --- |
| By November 7 | Freeze this proposal, load the EPIC tables, and report the first coverage counts. |
| November 8–21 | Finish the component table, code the DROID fields, and make the figure. |
| November 22–December 5 | Interpret the table, run the split checks, and decide on Ego4D. |
| December 6–14 | Polish the notebook, write the report of ~6 pages, and record each contribution. |
| December 15–16 | Prepare the eight-minute presentation. |
| December 17 | Submit the final report. |

## Team organization

Two members - Ria and Naomi. Ria 'owns' the EPIC download, preprocessing, descriptive statistics, and the figure. Naomi is in charge of the DROID field coding, the metadata check, the component table, and the report draft. We both label each lab-only row as a note the person can supply or as rig hardware, and disagreements are recorded in the repository.

## Questions for the course staff

Considering these datasets are outside the suggested list. Is the project acceptable if the graded analysis uses the EPIC annotation tables and DROID metadata, and leaves the full robot recordings unloaded?
