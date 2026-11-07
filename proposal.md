
## Abstract

Household tasks such as pouring, opening, and moving objects supply training data for egocentric vision and robot learning, but the language ( aka annotations) that names those tasks is written in different ways. Ego4D is filmed by people wearing a camera during ordinary activity, and the Forecasting Hands and Objects release stores a verb and a noun for each clip, and the sentence text lives in a separate narrations file. AgiBot World Beta records manipulation on a robot and stores a written command for each action. This project asks to what extent a verb labeler trained on the home narrations still names the verb on a robot command, and which reading of the leftover words carries that transfer. We will remove the verb from every sentence before a program sees it, so the labeler cannot copy the answer. We will answer with three programs, scored first on the Ego4D validation narrations and then on a hand-labeled sample of AgiBot commands.

## Research questions

To what extent does a verb labeler trained on home-kitchen narrations transfer to robot-lab instructions, and which text representation carries the transfer?

We narrow the down to 3 smaller questions : 
1. On Ego4D validation narrations the labeler never trained on, how often do the leftover words name the right verb? 2. How far does that score drop when the same labeler reads an AgiBot command? 3. Which reading carries more of the score across, a word count or a sentence embedding?

Both collections are treated as records of a physical object task. A home narration and a robot command are both supposed to say what the action is. Training and testing on home narrations only would show that leftover home words can name a home verb. Scoring the same frozen labelers on the robot commands measures whether a reader that learned the first can still read the second. If the score holds with the word count, shared objects are doing the labeling. If it holds only with the embedding, the actions match and the wording does not. If both fall to a rule that ignores the leftover words, the home narration is not a drop-in verb name for a robot command.

## Dataset

The primary dataset is Ego4D (Grauman et al., 2022): 3,670 hours of egocentric video from 931 participants. We will use the Forecasting Hands and Objects annotations. A narration and a verb are stored in separate files. The public split assigns five-minute clips to train, validation, and test, and the test labels are withheld. We will train on the training clips and score the validation clips. We will not score the test split.

We will join each verb to one sentence from the narrations file on video id and timestamp. The action window is several seconds and can hold more than one sentence. When two sentences are about equally close to the action, we will mark the row ambiguous and leave it out of the score. The split holds out videos rather than people, so the same participant can appear in training and in validation. We will report two validation scores, one on every unambiguous row and one with rows whose participant also appears in training removed.

## Additional datasets

AgiBot World Beta (Bu et al., 2025) is the lab comparison: about one million trajectories from more than 100 robots, with a written command on each action. We will use that text, not the video or the robot traces. The command has no Ego4D verb until we assign one.

An AgiBot command that names two actions is labeled with the first action only, and we will use that rule on every row. Both of us assign the verb independently, before either of us looks at a model guess. Where we agree, that verb is the test label. Where we disagree, we keep both marks and report the count.

## Methods

The unit is one leftover sentence. The leftover is the sentence after the verb word is removed. The answer is one verb. Three programs guess that verb. All three see the verb word removed. None of them is shown the verb at test time. We will fit every weight on the Ego4D training narrations only. We will train and score only the 37 verbs shared between Ego4D and AgiBot, and we will freeze that list before any model is fit.

The first program is a word count. We will build unigram and bigram weights from the training leftovers only, so a word and a pair of words can each count, and words that appear in almost every note are turned down. A logistic regression learns which patterns of those weights go with which verb.

The second program is a sentence embedding. We will pass each leftover sentence through all-MiniLM-L6-v2, a frozen model from the sentence-transformers project, not trained on Ego4D or AgiBot. A second logistic regression, trained separately on the training vectors only, turns those numbers into one verb. We will not fine-tune the embedding.

The third program is a rule, not a trained model. If a verb word is still in the sentence, it guesses that verb. Otherwise it guesses put if put is among the 37 shared verbs, and if put is not among them it guesses the most common verb among the 37 in the training narrations. The rule is the floor. A trained program that cannot beat it did not learn the leftover words.

We will score all three programs on the unambiguous Ego4D validation narrations, with the two participant-overlap scores reported beside that result. We will then run the same frozen programs on the agreed AgiBot labels. A verb that never appears in those labels will not enter the average. For each program on each pile we will report accuracy and macro-F1. Macro-F1 is the average of the per-verb scores, so a frequent verb cannot hide a program that misses the others. The drop from the home macro-F1 to the robot macro-F1 is how far that reading travels. The highest robot macro-F1 that still beats the rule is the reading that carries the transfer.

## Timeline

| Dates | Work |
| --- | --- |
| By November 7 | Freeze this proposal, load the Ego4D tables, and freeze the narration join. |
| November 8–21 | Fit the word-count model and both logistic regressions, embed the home narrations, and score the validation set. Draw the AgiBot sample and write the labeling rules. |
| November 22–December 5 | we both label the AgiBot sample, score the three frozen programs on it, and fill the metric table. |
| December 6–19 | Interpret the drops, polish the notebook, write the report of ~6 pages, and record each contribution. |
| December 20–21 | Prepare the eight-minute presentation. |
| December 22 | Submit the final report. |

## Team organization

Ria -  is in charge of the Ego4D download, the verb stripping, the word-count logistic regression, and the report draft. 
Naomi - is in charge of the narration join, the frozen embedding, the second logistic regression, and the AgiBot sample. We both* label every robot command, and disagreements are recorded in the repository. We both read the metric table before the interpretation is written.

## Questions for the course staff

These datasets are outside the suggested list. Is the project acceptable if the graded analysis uses the Ego4D Forecasting Hands and Objects annotation tables and a hand-labeled sample of AgiBot World Beta commands, and leaves the robot video unloaded?

AgiBot commands have no verb until we assign one, so the robot score depends on our agreement. Is a hand-labeled sample of AgiBot World Beta commands acceptable as the robot-side test? Would a semi-automated pass also be acceptable, in which a model proposes the verb and we review only the uncertain commands?

The shared list is 37 verbs. Is macro-F1 on that list stable enough, or should rare verbs be filtered before scoring?

The Ego4D validation split holds out videos, not people. Is reporting two scores, one with same-participant rows removed, sufficient to address that leakage?

The rule guesses put when no verb word remains, and put may dominate the 37-verb list. Is that a fair floor?

After the verb is removed, a home narration and a robot command are still different kinds of text. Is the remaining style gap a serious threat to the transfer measurement, or does removing the verb neutralize it?

Ambiguous rows, where two sentences are about equally close to an action, are left out of the score. Could excluding them bias the test toward easier or harder examples?

If a model proposes the verb and we review only the uncertain commands, is there a risk that the proposal becomes the ground truth rather than an independent judgment?

## Use of AI

For feedback, we used chatgpt to check our proposal and look for potential problems and inconsistencies within the methods, the validation split, and the choice of datasets. Furthermore, chatgpt was used to help talk through potential solutions to said problems, everything from brainstorming how to refine the main question to whether or not the pipeline is feasible. 


