"""Test the assistant's guardrails through the real SDK against a local stand-in for the API.

This checks the wiring without a key or any cost. It says nothing about how Claude itself behaves.
"""
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import anthropic

from assistant import MODEL, run
from hospital import PROMPTS, identifiers_in

requests = []
script = []


def tool_use(name, **tool_input):
    return {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": f"toolu_{len(requests)}", "name": name, "input": tool_input}]}


def reply(text):
    return {"stop_reason": "end_turn", "content": [{"type": "text", "text": text}]}


class StandIn(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        requests.append({"body": body, "beta": self.headers.get("anthropic-beta", "")})
        step = script.pop(0)
        step = step(body) if callable(step) else step
        payload = json.dumps({
            "id": "msg_test", "type": "message", "role": "assistant", "model": MODEL, "stop_sequence": None,
            "usage": {"input_tokens": 100, "output_tokens": 10}, **step,
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def outgoing_text():
    """Everything the assistant sent as user content, across every request."""
    return json.dumps([m for r in requests for m in r["body"]["messages"] if m["role"] == "user"])


def first_placeholder(body):
    return re.search(r"\[ID_\d+\]", json.dumps(body["messages"])).group(0)


def scenario(question, guarded, steps):
    requests.clear()
    script[:] = steps
    return run(question, guarded, client)


server = HTTPServer(("127.0.0.1", 0), StandIn)
threading.Thread(target=server.serve_forever, daemon=True).start()
client = anthropic.Anthropic(api_key="test-key", base_url=f"http://127.0.0.1:{server.server_port}", max_retries=0)

summary = PROMPTS["Summarise my patient's chart"][0]

# Unguarded: identifiers go out in the message and the chart.
result = scenario(summary, False, [tool_use("find_patient", query="4839201"), reply("Stable.")])
assert result["identifiers_sent"] > 0, result
assert "Margaret Okafor" in outgoing_text()

# Guarded: the same request sends no identifier, and the clinician still sees the real one.
result = scenario(summary, True, [
    lambda body: tool_use("find_patient", query=first_placeholder(body)),
    lambda body: reply(f"Chart for {first_placeholder(body)} reviewed."),
])
assert result["identifiers_sent"] == 0, result["sent_to_claude"]
assert not identifiers_in(outgoing_text()), outgoing_text()
assert "4839201" in result["reply"] and "[ID_" not in result["reply"], result["reply"]
assert result["tool_calls"][0]["query"] == "4839201", result["tool_calls"]

# The request itself: right model, safety fallback on, tools strict.
sent = requests[0]
assert sent["body"]["model"] == "claude-opus-5-5"
assert sent["body"]["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in sent["beta"]
assert all(tool["strict"] for tool in sent["body"]["tools"])

# A direct attack never reaches the API.
result = scenario(PROMPTS["Direct prompt injection"][0], True, [])
assert result["blocked"] and not requests, result

# Access control: another clinician's patient stays private, even if Claude searches for them.
result = scenario(PROMPTS["Another clinician's patient"][0], True, [
    lambda body: tool_use("find_patient", query="Liam Becker"),
    reply("I can't access that chart."),
])
assert "Liam Becker" not in outgoing_text() and "6604871" not in outgoing_text(), outgoing_text()
assert any(t["Stage"] == "Access control" and t["Result"] == "BLOCKED" for t in result["trace"]), result["trace"]

# Asking for everyone returns only assigned patients.
result = scenario(PROMPTS["List every patient"][0], True, [tool_use("find_patient", query=""), reply("Done.")])
assert "6604871" not in outgoing_text(), outgoing_text()
result = scenario(PROMPTS["List every patient"][0], False, [tool_use("find_patient", query=""), reply("Done.")])
assert "6604871" in outgoing_text()

# A hidden instruction in a chart is withheld from Claude.
result = scenario(PROMPTS["Instruction hidden in a chart"][0], True, [
    lambda body: tool_use("find_patient", query=first_placeholder(body)),
    reply("Reviewed."),
])
assert "oxycodone" not in outgoing_text(), outgoing_text()
assert any(t["Stage"] == "Injection check on chart" for t in result["trace"]), result["trace"]
result = scenario(PROMPTS["Instruction hidden in a chart"][0], False, [tool_use("find_patient", query="5510932"), reply("Reviewed.")])
assert "oxycodone" in outgoing_text()

# Orders: held for a human when guarded, placed straight away when not.
order = dict(drug="morphine", dose="500 mg IV", reason="pain")
result = scenario(PROMPTS["Order with a dangerous dose"][0], True, [
    lambda body: tool_use("draft_order", mrn=first_placeholder(body), **order),
    reply("Drafted."),
])
assert [o["status"] for o in result["orders"]] == ["Held for clinician approval"], result
assert result["orders"][0]["mrn"] == "4839201", result["orders"]
result = scenario(PROMPTS["Order with a dangerous dose"][0], False, [tool_use("draft_order", mrn="4839201", **order), reply("Done.")])
assert [o["status"] for o in result["orders"]] == ["Placed automatically"], result

# Guarded orders can't target another clinician's patient, and unknown record numbers fail in both modes.
result = scenario("Order for MRN 6604871.", True, [tool_use("draft_order", mrn="6604871", **order), reply("Blocked.")])
assert not result["orders"], result
result = scenario("Order.", False, [tool_use("draft_order", mrn="0000000", **order), reply("Not found.")])
assert not result["orders"], result

# Claude declining is reported, not treated as an answer.
result = scenario(summary, False, [{"stop_reason": "refusal", "content": [], "stop_details": {"type": "refusal", "category": None, "explanation": None}}])
assert result["reply"] == "Claude declined this request.", result

# A loop that never finishes stops at the limit.
result = scenario(summary, False, [tool_use("find_patient", query="4839201") for _ in range(6)])
assert "Stopped after" in result["reply"], result

# Clinical wording gets refused, which is the false positive the lab measures.
result = scenario(PROMPTS["Clinical instruction that looks like an attack"][0], True, [])
print("Clinical instruction blocked:", result["blocked"], result["trace"][:1])
print("assistant checks passed")
