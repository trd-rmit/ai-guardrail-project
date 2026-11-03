"""The guardrail checks. Each flag keeps its raw score so thresholds can change without re-running models."""
import re

# Prompt Guard and Llama Guard need gated Hugging Face access, so these open models stand in for now.
INJECTION_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
CONTENT_MODEL = "unitary/toxic-bert"
CLINICAL_MODEL = "obi/deid_roberta_i2b2"

GUARDRAILS = {
    "pii": "PII: Presidio default",
    "pii_tuned": "PII: Presidio tuned",
    "pii_clinical": "PII: clinical RoBERTa",
    "pii_combined": "PII: RoBERTa + patterns",
    "injection": "Injection: ProtectAI DeBERTa",
    "content": "Content: toxic-bert",
}
# Presidio scores phone numbers 0.4 without nearby context, so 0.5 would let them through.
THRESHOLDS = {"pii": 0.35, "pii_tuned": 0.35, "pii_clinical": 0.5, "pii_combined": 0.35, "injection": 0.5, "content": 0.5}
PII_GUARDRAILS = [name for name in GUARDRAILS if name.startswith("pii")]
# Presidio's fixed-format detectors, without its language model.
PATTERN_ENTITIES = ["EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "MEDICAL_RECORD_NUMBER"]

# Only terms from the dev split, so the test split shows how well this holds up on unseen names.
MEDICAL_NAME = re.compile(
    r"\b[\w'-]+(?: [\w'-]+)? (?:disease|lymphoma|thyroiditis|sign|procedure|catheter|scale|wort)\b",
    re.IGNORECASE,
)
FULL_DATE = re.compile(r"\d{1,4}[/-]\d{1,2}[/-]\d{1,4}|[A-Za-z]{3,9} \d{1,2},? \d{4}")
NAME_SHAPED = re.compile(r"[A-Z][\w'-]*[a-z][\w'-]*(?: [A-Z][\w'-]*[a-z][\w'-]*)+")

_analyzer = _mrn_recognizer = _injection = _content = _clinical = None


def _load():
    # Imported here because these take seconds to load and the scoreboard doesn't need them.
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
    from transformers import pipeline

    global _analyzer, _mrn_recognizer, _injection, _content, _clinical
    if _analyzer is None:
        # stride makes it read the whole text in overlapping windows, not just the first 512 tokens.
        _clinical = pipeline("token-classification", model=CLINICAL_MODEL, aggregation_strategy="simple", stride=64)
        _analyzer = AnalyzerEngine()
        _mrn_recognizer = PatternRecognizer(
            supported_entity="MEDICAL_RECORD_NUMBER",
            patterns=[Pattern("mrn", r"\b\d{6,10}\b", 0.3)],
            context=["mrn", "record"],  # Presidio matches single words here
        )
        _injection = pipeline("text-classification", model=INJECTION_MODEL, top_k=None)
        _content = pipeline("text-classification", model=CONTENT_MODEL, top_k=None)


def check_pii(text):
    _load()
    return [
        {"guardrail": "pii", "label": r.entity_type, "score": r.score, "start": r.start, "end": r.end}
        for r in _analyzer.analyze(text=text, language="en")
    ]


def check_pii_tuned(text):
    """Presidio plus a record number recognizer, without names of diseases or relative times."""
    _load()
    medical = [m.span() for m in MEDICAL_NAME.finditer(text)]
    flags = []
    for r in _analyzer.analyze(text=text, language="en", ad_hoc_recognizers=[_mrn_recognizer]):
        if r.entity_type in ("PERSON", "LOCATION") and any(r.start < e and s < r.end for s, e in medical):
            continue
        if r.entity_type == "DATE_TIME" and not FULL_DATE.search(text[r.start:r.end]):
            continue
        flags.append({"guardrail": "pii_tuned", "label": r.entity_type, "score": r.score, "start": r.start, "end": r.end})
    # spaCy sometimes labels a person's name as an organisation, and Presidio ignores those.
    for ent in _analyzer.nlp_engine.nlp["en"](text).ents:
        if ent.label_ == "ORG" and NAME_SHAPED.fullmatch(ent.text) and not any(ent.start_char < e and s < ent.end_char for s, e in medical):
            flags.append({"guardrail": "pii_tuned", "label": "POSSIBLE_NAME", "score": 0.85, "start": ent.start_char, "end": ent.end_char})
    return flags


def check_pii_clinical(text):
    """A model trained to find identifiers in clinical notes. It tags word pieces, so join them into spans."""
    _load()
    flags = []
    for token in _clinical(text):
        label = token["entity_group"].split("-", 1)[-1]
        last = flags[-1] if flags else None
        gap = text[last["end"]:token["start"]] if last else None
        if last and (gap == "" or (last["label"] == label and not gap.strip())):  # gap == "": pieces of one word
            last["end"] = token["end"]
            # The span's most confident piece decides, so one unsure piece can't unmask part of a name.
            last["score"] = max(last["score"], float(token["score"]))
        else:
            flags.append({"guardrail": "pii_clinical", "label": label, "score": float(token["score"]), "start": token["start"], "end": token["end"]})
    return flags


def check_pii_combined(text):
    """The clinical model finds names and places, and patterns back it up on fixed formats like record numbers."""
    flags = [{**flag, "guardrail": "pii_combined"} for flag in check_pii_clinical(text)]
    for r in _analyzer.analyze(text=text, language="en", entities=PATTERN_ENTITIES, ad_hoc_recognizers=[_mrn_recognizer]):
        flags.append({"guardrail": "pii_combined", "label": r.entity_type, "score": r.score, "start": r.start, "end": r.end})
    return flags


def _classify(model, guardrail, text, ignored_labels):
    # shortcut: only the first 512 tokens are checked, split long notes into chunks before real use
    scores = model(text, truncation=True)[0]
    top = max((s for s in scores if s["label"] not in ignored_labels), key=lambda s: s["score"])
    return [{"guardrail": guardrail, "label": top["label"], "score": top["score"], "start": 0, "end": len(text)}]


def check_injection(text):
    _load()
    return _classify(_injection, "injection", text, {"SAFE"})


def check_content(text):
    _load()
    return _classify(_content, "content", text, set())


CHECKS = {
    "pii": check_pii,
    "pii_tuned": check_pii_tuned,
    "pii_clinical": check_pii_clinical,
    "pii_combined": check_pii_combined,
    "injection": check_injection,
    "content": check_content,
}


if __name__ == "__main__":
    # A name past the model's 512-token limit must still be found.
    long_note = "Margaret Okafor was admitted. " + "Patient remains stable on the ward overnight. " * 200 + "Summary sent to Daniel Reyes."
    found = [long_note[f["start"]:f["end"]] for f in check_pii_clinical(long_note)]
    assert any("Margaret" in name for name in found) and any("Reyes" in name for name in found), found
    # Word pieces must join into one span, or the highlights and the scoring would see fragments.
    short = "Soo-ah Kim, MRN 4839201, denies suicidal ideation."
    assert "Soo-ah Kim" in [short[f["start"]:f["end"]] for f in check_pii_clinical(short)]
    # The clinical model only masks part of this number, and the patterns should cover the rest.
    assert "4839201" in [short[f["start"]:f["end"]] for f in check_pii_combined(short) if f["score"] >= THRESHOLDS["pii_combined"]]
    print("guardrail checks passed")
