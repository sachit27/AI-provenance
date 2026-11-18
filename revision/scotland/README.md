# Scottish inputs

`code/prepare.py` selects the official qualitative paragraphs and prepares the six written questions from the downloaded response archive. Run it after the download, parsing and verification steps in [the revision instructions](../README.md).

The committed dataset files contain question wording, selected government report passages, target definitions and counts. `records.jsonl`, exclusions and embeddings are created locally. The primary analysis then applies the relevance screen through `scotland_original/pipeline/`.

Targets cover the qualitative findings paired with each question. The full report also uses unpublished responses and wider engagement. Low similarity to a selected passage does not establish that a view is absent from the whole report.
