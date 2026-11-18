# Scottish source archive

`fetch_public.py` downloads the published Citizen Space response pages. `parse_public.py` extracts free-text and simple closed answers, and `verify_audit.py` checks the text independently with lxml. `fetch_sources.py` retrieves the report, consultation background and methodological sources.

`official_findings_sections.json` records government report prose grouped by heading. `verify_sources.py` extracts headings and paragraphs from the downloaded findings page and compares them with this index. The question-to-paragraph map is in `analyses/scotland/code/prepare.py`.

Run the commands in [the analysis workflow](../../README.md). Raw pages, response text, respondent identifiers and download manifests are generated locally. The original archive contained 1,809 published responses out of 2,207 received by the Scottish Government.
