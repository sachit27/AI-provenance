# Stance-group coverage analysis (Scotland) - reviewer question: are minority opinions lost?

Groups are defined by participants' OWN closed answers in the same consultation (no human labelling, no embedding clusters).

**Mapping chronology.** `MAPPING.json` (free-text question -> closed question(s), collapse rules, primary/secondary choice, minority definition, test families) was written from question wording alone, after parsing the closed answers and before any coverage file was read or analysed. `analyze.py` reads the mapping; it does not choose groups.

## Files
- `parse_closed.py` (stdlib) -> `closed_answers.csv` (long: respondent_id, q_num, q_text, sub_item, option, ticked_n; one row per ticked option, blank option = shown but unanswered, no row = question absent), `parse_qa.json`.
- `build_groups.py` (stdlib) -> `respondent_groups.csv` (stance/type labels per respondent; blank = missing).
- `MAPPING.json` - the mapping and rules (rationale below).
- `analyze.py` (numpy, pandas; scipy optional) -> `RESULTS.md`, `results_all.csv`, `results_primary_official.csv`, `group_sizes.csv`.

## Commands (from repository root)
```
python analyses/stance_groups/parse_closed.py
python analyses/stance_groups/build_groups.py
python analyses/stance_groups/analyze.py     # needs coverage CSVs
```
Inputs: `analyses/scotland_analysis/analysis/<topic>/participant_coverage.csv` (official; required) and `analyses/llm_summaries/scores/<topic>/participant_coverage.csv` (generated; optional; seeds averaged per respondent). Options: `--perms 10000 --boot 2000`. Dependencies: see `analyses/requirements.txt` (numpy, pandas, scipy; bs4/lxml/statsmodels not needed here; the HC3 regression is implemented in numpy).

## Mapping rationale (wording only)
| Free text | Linked closed question(s) | Stance variable |
|---|---|---|
| q4 grouping/level of regulation | Q1-Q3 (procedure -> regulation group) | Primary: choices vs the group the consultation proposed for each procedure; support = >=50% of answered procedures as proposed, oppose = otherwise (direction stricter/looser descriptive only). |
| q7 comments on Q5 and Q6 | Q5, Q6 (explicit) | Primary Q5 (first referenced); secondary Q6 (opposite polarity: "should NOT require a licence"). |
| q20 protected characteristics, q21 financial situation, q22 island communities | none (impact questions) | Respondent type only. |
| q23 UNCRC and "potential of age restrictions" | Q19, Q17 | Primary Q19 (agree/disagree 18+ for intimate areas; Q17 has no support/oppose scale); secondary Q17. |

Rule: primary = first linked closed question that has a support/oppose or agree/disagree scale. Collapse: (strongly/somewhat) support or agree = support; neutral/neither/don't know = neutral; oppose/disagree = oppose. Minority = smaller of support/oppose among respondents of that topic in the public data; neutral excluded from tests. Respondent type = Q25 individual vs organisation (the only About-you category present; no practitioner/business/public field exists). Families for Holm: primary stance (topic x embedding, official), secondary stance, respondent type; generated summaries Holm within summarizer x family and treated as exploratory.

## Statistics
Mean difference (minority - majority), 10,000-permutation respondent-level test (fixed seed), 2,000 bootstrap CI, share in overall bottom decile, low-coverage rate (official only), and OLS `coverage ~ group + log(word_count)` with HC3 SE.

## Caveats
Coverage differences are semantic similarity, not proof of misrepresentation. Public subset is 1,809 of 2,207 responses. Q18 is absent from the public pages. Parsed proportions match official findings closely (see `parse_qa.json`).
