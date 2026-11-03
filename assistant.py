"""The clinical assistant, runnable with or without guardrails around Claude."""
import re

import anthropic
from dotenv import load_dotenv

from guardrails import THRESHOLDS, check_content, check_injection, check_pii_combined
from hospital import ASSIGNED_MRNS, CLINICIAN, PATIENTS, chart_text, find_patients, identifiers_in

load_dotenv()

MODEL = "claude-opus-5-5"
MAX_TURNS = 6
# Dollars per million tokens, for the cost shown in the app.
INPUT_PRICE, OUTPUT_PRICE = 4.00, 20.00

SYSTEM = (
    f"You are a clinical notes assistant at a made-up teaching hospital, helping {CLINICIAN}. "
    "Use find_patient to read charts and draft_order to draft medication orders. Keep answers brief."
)
GUARDED_SYSTEM = SYSTEM + (
    " Identifiers in this conversation are replaced by placeholders like [ID_1]. "
    "Use them exactly as written, including in tool inputs, and never guess what they stand for."
)

TOOLS = [
    {
        "name": "find_patient",
        "description": "Search patient charts by name or medical record number. An empty query returns every chart you may see.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Name, record number or placeholder to search for."}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "draft_order",
        "description": "Draft a medication order for a patient.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "mrn": {"type": "string", "description": "The patient's record number or its placeholder."},
                "drug": {"type": "string"},
                "dose": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["mrn", "drug", "dose", "reason"],
            "additionalProperties": False,
        },
    },
]


class Masker:
    """Swaps identifiers for placeholders before text leaves, and back again for the clinician."""

    def __init__(self):
        self.tokens = {}
        self.real = {}

    def mask(self, text):
        flagged = sorted(
            (f["start"], f["end"]) for f in check_pii_combined(text) if f["score"] >= THRESHOLDS["pii_combined"]
        )
        spans = []
        for start, end in flagged:
            if spans and start <= spans[-1][1]:
                spans[-1][1] = max(spans[-1][1], end)
            else:
                spans.append([start, end])
        parts, position = [], 0
        for start, end in spans:
            parts += [text[position:start], self._token(text[start:end])]
            position = end
        return "".join(parts) + text[position:]

    def _token(self, value):
        if value not in self.tokens:
            self.tokens[value] = f"[ID_{len(self.tokens) + 1}]"
            self.real[self.tokens[value]] = value
        return self.tokens[value]

    def unmask(self, text):
        return re.sub(r"\[ID_\d+\]", lambda m: self.real.get(m.group(0), m.group(0)), text)


def _find_patient(query, guarded, result):
    if not guarded:
        patients = find_patients(query)
        return "\n\n".join(chart_text(p) for p in patients) or "No chart matches."

    patients = find_patients(query, ASSIGNED_MRNS)
    blocked = len(find_patients(query)) - len(patients)
    if blocked:
        _trace(result, "Access control", "BLOCKED", f"{blocked} chart(s) outside {CLINICIAN}'s patients")
    charts = []
    for patient in patients:
        # Charts are untrusted too: an attacker can write to a note without ever messaging the assistant.
        score = check_injection(patient["note"])[0]["score"]
        if score >= THRESHOLDS["injection"]:
            _trace(result, "Injection check on chart", "BLOCKED", f"Note withheld, score {score:.2f}")
            charts.append(chart_text(patient, note="[Withheld because it may contain instructions aimed at the assistant.]"))
        else:
            charts.append(chart_text(patient))
    return "\n\n".join(charts) or "No chart you may see matches."


def _draft_order(args, guarded, result):
    if args["mrn"] not in {p["mrn"] for p in PATIENTS}:
        return f"No patient has record number {args['mrn']}.", True
    if guarded and args["mrn"] not in ASSIGNED_MRNS:
        _trace(result, "Access control", "BLOCKED", f"Order for a patient outside {CLINICIAN}'s list")
        return "You may not draft orders for this patient.", True
    status = "Held for clinician approval" if guarded else "Placed automatically"
    result["orders"].append({**args, "status": status})
    if guarded:
        _trace(result, "Human approval", "HELD", f"{args['drug']} {args['dose']} waits for {CLINICIAN}")
    return f"Order status: {status}.", False


def _run_tool(block, guarded, masker, result):
    args = {key: masker.unmask(value) for key, value in block.input.items()} if guarded else dict(block.input)
    result["tool_calls"].append({"tool": block.name, **args})
    if block.name == "find_patient":
        text, is_error = _find_patient(args["query"], guarded, result), False
    else:
        text, is_error = _draft_order(args, guarded, result)
    return {"type": "tool_result", "tool_use_id": block.id, "content": masker.mask(text) if guarded else text, "is_error": is_error}


def _trace(result, stage, outcome, detail=""):
    result["trace"].append({"Stage": stage, "Result": outcome, "Detail": detail})


def _sent(result, text):
    result["sent_to_claude"].append(text)
    result["identifiers_sent"] += len(identifiers_in(text))


def run(question, guarded, client=None):
    """Answer one question. Returns everything the app needs to show what happened."""
    result = {
        "reply": "", "blocked": False, "tool_calls": [], "orders": [], "trace": [], "sent_to_claude": [],
        "identifiers_sent": 0, "input_tokens": 0, "output_tokens": 0,
    }
    masker = Masker() if guarded else None
    outgoing = question

    if guarded:
        for stage, check, name in (("Injection check", check_injection, "injection"), ("Content check", check_content, "content")):
            score = check(question)[0]["score"]
            flagged = score >= THRESHOLDS[name]
            _trace(result, stage, "BLOCKED" if flagged else "Passed", f"Score {score:.2f}")
            if flagged:
                result["blocked"] = True
                result["reply"] = f"Refused before reaching Claude: the {stage.lower()} flagged this message."
                return result
        outgoing = masker.mask(question)
        _trace(result, "Identifier masking", "Passed", f"{len(masker.tokens)} value(s) replaced")

    client = client or anthropic.Anthropic()
    messages = [{"role": "user", "content": outgoing}]
    _sent(result, outgoing)

    # A manual loop rather than the tool runner, because every byte sent needs recording and every tool call gating.
    for _ in range(MAX_TURNS):
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=GUARDED_SYSTEM if guarded else SYSTEM,
            tools=TOOLS,
            messages=messages,
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        result["input_tokens"] += response.usage.input_tokens
        result["output_tokens"] += response.usage.output_tokens
        text = "".join(b.text for b in response.content if b.type == "text")

        if response.stop_reason == "refusal":
            result["reply"] = "Claude declined this request."
            _trace(result, "Claude", "Declined", getattr(response.stop_details, "category", None) or "")
            return result
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if response.stop_reason != "tool_use" or not tool_uses:
            if response.stop_reason == "max_tokens":
                text += "\n\n[Reply cut off at the length limit.]"
            result["reply"] = masker.unmask(text) if guarded else text
            return result

        messages.append({"role": "assistant", "content": response.content})
        tool_results = [_run_tool(block, guarded, masker, result) for block in tool_uses]
        for tool_result in tool_results:
            _sent(result, tool_result["content"])
        messages.append({"role": "user", "content": tool_results})

    result["reply"] = f"Stopped after {MAX_TURNS} turns without a final answer."
    return result


def cost(result):
    return (result["input_tokens"] * INPUT_PRICE + result["output_tokens"] * OUTPUT_PRICE) / 1_000_000
