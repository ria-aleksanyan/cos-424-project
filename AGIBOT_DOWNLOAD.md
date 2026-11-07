# Download AgiBot Command Text

## Install

```bash
/opt/miniconda3/bin/python3 -m pip install huggingface_hub
```

## Login

```bash
/opt/miniconda3/bin/python3 -m huggingface_hub.cli.hf auth login
```

Paste your token when prompted. Get a token at huggingface.co/settings/tokens if you do not have one.

## Download

```bash
/opt/miniconda3/bin/python3 -m huggingface_hub.cli.hf download agibot-world/AgiBotWorld-Beta --include "task_info/*.json" --repo-type dataset --local-dir data/agibot
```

## Check

Open any downloaded JSON file and confirm the action_text field contains command sentences.

## Where the files are

The 217 task-info JSON files were downloaded separately with the commands above. They are not in this repository. On this machine they live at `/Users/riaaleksanyan/data/agibot/task_info`.
