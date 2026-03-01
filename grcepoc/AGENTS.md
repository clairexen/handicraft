## Dev notes & agent cheat sheet

(Pretty much all code in this repo is AI-generated—but under strong human supervision. ~Claire)

High-level specs, defaults, and workflow reminders now live exclusively in `notes.txt`; AGENTS.md just points visiting helpers there so we avoid duplicating configuration tables that drift from the source of truth.

The main code file is `grce.py`. The implementation in `grce.py` and the spec in `notes.txt` should stay in sync. However, agents should not edit `notes.txt` unless explicitly asked—recommend spec changes to the user instead.

`corpus.py` is an auxiliary script for generating tokenizers and tokenizing training/test corpus files.

`plot.py` is an auxiliary script for displaying loss graphs and other statistics collected during training.

`pod.sh` is a helper script for running training on runpod.io more easily.
