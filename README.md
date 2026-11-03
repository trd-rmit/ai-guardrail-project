# ai-guardrail-project

Experiments demonstrating how LLM guardrails can be implemented across different frameworks.

The first experiment is **Clinical Guardrail Lab**. Guardrails built for general text can misread normal medical language, so the lab measures two things for each guardrail: false alarms on clinical text, and attacks or identifiers it misses.

All test data is made up by hand. Never add real patient data to this repo.

![Pipeline from a message through input checks, across a trust boundary to an LLM, then through output checks to a reply. Only the three input checks have been tested so far.](docs/figures/pipeline.svg)

Identifiers have to be removed before text crosses the boundary, so detection comes first. Only the solid boxes exist today, and the lab detects identifiers without replacing them yet. Miss counts are from held-out and fresh test cases.

## Results so far

### Finding identifiers

These 167 fresh cases were written before any guardrail ran on them, and nothing was tuned afterwards:

| Guardrail | False alarms on clinical notes | Identifier notes missed | Name sentences missed | Time per text |
|---|---|---|---|---|
| Presidio default | 26 of 59 | 13 of 24 | 15 of 108 | about 8 ms |
| Presidio tuned | 11 of 59 | 5 of 24 | 12 of 108 | about 14 ms |
| Clinical RoBERTa | 11 of 59 | 7 of 24 | 0 of 108 | about 180 ms |
| **Clinical RoBERTa + patterns** | **12 of 59** | **0 of 24** | **0 of 108** | about 180 ms |

![Scatter chart of false alarms against identifiers missed. Presidio default: 26 of 59 and 28 of 132. Presidio tuned: 11 of 59 and 17 of 132. Clinical model: 11 of 59 and 7 of 132. Clinical model plus patterns: 12 of 59 and 0 of 132.](docs/figures/results.svg)

The combined setup uses [`obi/deid_roberta_i2b2`](https://huggingface.co/obi/deid_roberta_i2b2), a model trained on clinical notes, plus Presidio's fixed patterns for emails, phone numbers and SSNs and a custom record number pattern. A note counts as missed if any identifier in it isn't fully masked, so partly masking an email or a record number counts as a miss.

![A test note goes to the clinical model and to fixed patterns. The model flags the name and only the first five digits of the record number. Patterns flag the whole number. Together they cover both identifiers and leave the clinical text unflagged.](docs/figures/combined-detection.svg)

This is the real output for one fresh test case, with flagged text underlined. Presidio's default setup flagged only the name on this note.

Missed names on the same cases, out of 18 sentences per group:

| Name group | Presidio default | Presidio tuned | Clinical RoBERTa, alone or combined |
|---|---|---|---|
| South Asian | 1 | 1 | 0 |
| East and Southeast Asian | 4 | 4 | 0 |
| African | 7 | 4 | 0 |
| Middle Eastern | 2 | 2 | 0 |
| European and Anglo | 1 | 1 | 0 |
| Latin American | 0 | 0 | 0 |

Groups are a rough label for where a made-up name comes from, and each has only 6 names, so this is a reason to test further, not proof of bias.

What stands out:

- **The two approaches cover each other's gaps.** The clinical model finds names that Presidio misses. The patterns finish numbers and emails that the model only partly masks.
- **Zero misses here doesn't mean zero in practice.** On earlier development cases the combined setup still missed parts of the name "Siti Rahmawati" in 2 of 5 sentences.
- **Its 12 false alarms aren't equally serious.** The scoring counts them all the same:
  - 6 are times and ages such as "Tuesday", "39 weeks" and "Sixty-four". Whether these need masking depends on your privacy rules.
  - 3 are labels such as "Dr" and "SSN", which are harmless to mask.
  - 2 are medical names, "McBurney" and part of "Hartmann", which can change clinical meaning.
  - 1 is part of a drug code.
- **Presidio alone misreads clinical text.** It flags "2 weeks" and "Monday" as dates, and "Guillain-Barre" as a person. spaCy also sometimes labels a person as an organisation, which Presidio then ignores.

### Attacks and harmful messages

These numbers come from the 38 held-out cases, whose mistakes have been reviewed:

| Guardrail | False alarms on clinical notes | Missed |
|---|---|---|
| Injection: ProtectAI DeBERTa | 2 of 28 | 2 of 7 |
| Content: toxic-bert | 0 of 28 | 1 of 3 |

- **A threshold can't fix the injection model.** Across the development and held-out cases it makes 5 false alarms and 3 misses at every threshold tested from 0.05 to 0.95. "Disregard the previous dose" scores about 1.0, and a request for the names and addresses of every patient with HIV scores about 0.
- **Asking for someone else's records isn't an injection.** These requests need access controls in the application, not a text classifier.

![Two line charts against threshold, with false alarms in blue and attacks missed in orange. The injection model stays flat at 5 of 68 false alarms and 3 of 17 missed. The content model moves from 6 of 68 false alarms and 0 of 7 missed to 0 of 68 and 3 of 7.](docs/figures/thresholds.svg)

Blue lines are false alarms on clinical notes and orange lines are attacks missed, across development and held-out cases.

### How far to trust this

These are 739 cases written by one author, mostly short sentences, with few examples per category. Read them as signs of where to look, not as benchmarks. The real test is the organisation's own data, inside an environment approved for it.

## Run it

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m spacy download en_core_web_lg
.venv/Scripts/python -m streamlit run app.py
```

The saved results load straight away. "Try it" downloads the three models on first use, about 2.5 GB.

To re-run the guardrails on the test set, or run the self-checks:

```bash
.venv/Scripts/python evaluate.py
```

```bash
.venv/Scripts/python scoring.py
```

```bash
.venv/Scripts/python guardrails.py
```

## What's where

- `data/cases.jsonl`: labelled test cases, split into development, held-out and fresh
- `guardrails.py`: the checks and the tuned rules
- `scoring.py`: how false alarms and misses are counted
- `evaluate.py`: runs every check and saves `results/scores.json`
- `app.py`: the Streamlit lab
- `docs/figures/`: the diagrams in this README, which switch between light and dark themes

## Limits

- Prompt Guard and Llama Guard need gated Hugging Face access, so open models stand in for them for now.
- The assistant itself isn't built yet, because it needs an LLM. This lab tests the guardrails on their own.
- The injection and content checks only read the first 512 tokens, so a long note needs splitting first. The PII checks read the whole text.
- Timings are from one run on a laptop CPU.
