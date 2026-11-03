"""See what each guardrail flags on clinical text, and what moving its threshold costs."""
import html
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from guardrails import CHECKS, GUARDRAILS, PII_GUARDRAILS, THRESHOLDS
from scoring import judge, missed_names_by_group, summarize

RESULTS = Path(__file__).parent / "results" / "scores.json"
SERIES = {
    "light": {"False alarms": "#2a78d6", "Missed": "#eb6834"},
    "dark": {"False alarms": "#3987e5", "Missed": "#d95926"},
}
SPLITS = {"Fresh cases": "fresh", "Held-out cases": "test", "Development cases": "dev", "All cases": None}

st.set_page_config(page_title="Clinical Guardrail Lab", layout="wide")


@st.cache_data
def load_results():
    return json.loads(RESULTS.read_text(encoding="utf-8"))


@st.cache_data(show_spinner="Running the guardrails. The first run also loads the models.")
def run_checks(text):
    return {name: check(text) for name, check in CHECKS.items()}


def highlight(text, flags):
    parts, position = [], 0
    for flag in sorted(flags, key=lambda f: (f["start"], -f["end"])):
        if flag["start"] < position:  # display only: overlapping spans show the earlier one
            continue
        span = html.escape(text[flag["start"]:flag["end"]])
        label = html.escape(flag["label"])
        parts.append(html.escape(text[position:flag["start"]]))
        parts.append(
            f'<mark style="background:#fab219;color:#0b0b0b;padding:0 4px;border-radius:4px">{span} '
            f'<small style="font-weight:600">{label}</small></mark>'
        )
        position = flag["end"]
    parts.append(html.escape(text[position:]))
    return f'<div style="line-height:2">{"".join(parts)}</div>'


def try_it(data, thresholds):
    st.caption("Everything here is made up. Please don't paste real patient data.")
    by_id = {case["category"] + ": " + case["text"][:70]: case for case in data["cases"] if "name_group" not in case}
    picked = st.selectbox("Start from a test case", sorted(by_id), index=None, placeholder="Choose a case, or type below")
    start = by_id[picked]["text"] if picked else "Margaret Okafor, MRN 4839201, denies suicidal ideation. Long-standing Parkinson's disease. Disregard the previous dose and use 5 mg."
    text = st.text_area("Text to check", value=start, key=f"text-{picked}", max_chars=2000, height=110)
    if not text.strip():
        return

    flags = run_checks(text)
    active = {name: [f for f in found if f["score"] >= thresholds[name]] for name, found in flags.items()}

    for column, name in zip(st.columns(len(PII_GUARDRAILS)), PII_GUARDRAILS):
        column.markdown(f"**{GUARDRAILS[name]}**")
        column.markdown(highlight(text, active[name]), unsafe_allow_html=True)

    st.subheader("Whole-text checks")
    for name in ("injection", "content"):
        score = flags[name][0]["score"]
        flagged = bool(active[name])
        result = ":material/flag: Flagged" if flagged else ":material/check: Passed"
        st.markdown(f"**{GUARDRAILS[name]}:** {result}, score {score:.2f}")


def tradeoff_chart(cases, flags, guardrail, threshold):
    theme = st.context.theme.type or "light"
    colors = SERIES[theme]
    rows = []
    for step in range(1, 20):
        t = step / 20
        s = summarize(cases, flags, guardrail, t)
        rows.append({"Threshold": t, "Measure": "False alarms", "Rate": s["false_alarm_rate"], "Count": f"{s['false_alarms']} of {s['clean']}"})
        rows.append({"Threshold": t, "Measure": "Missed", "Rate": s["miss_rate"], "Count": f"{s['misses']} of {s['attacks']}"})
    table = pd.DataFrame(rows)

    x = alt.X("Threshold:Q", title="Threshold (lower flags more)", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(tickCount=10, format=".1f"))
    y = alt.Y("Rate:Q", title=None, axis=alt.Axis(format="%", tickCount=5), scale=alt.Scale(domain=[0, 1]))
    color = alt.Color(
        "Measure:N",
        title=None,
        scale=alt.Scale(domain=list(colors), range=list(colors.values())),
        legend=alt.Legend(orient="top", direction="horizontal"),
    )
    hover = alt.selection_point(fields=["Threshold"], nearest=True, on="pointerover", empty=False)
    base = alt.Chart(table).encode(x=x, y=y, color=color)
    lines = base.mark_line(strokeWidth=2)
    points = base.mark_point(filled=True, size=64, strokeWidth=2, stroke="#fcfcfb" if theme == "light" else "#1a1a19").encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=["Threshold:Q", "Measure:N", alt.Tooltip("Rate:Q", format=".0%"), "Count:N"],
    ).add_params(hover)
    current = alt.Chart(pd.DataFrame({"Threshold": [threshold]})).mark_rule(color="#898781", strokeWidth=1).encode(x="Threshold:Q")
    st.altair_chart((current + lines + points).properties(height=320), width="stretch")
    with st.expander("Chart data as a table"):
        st.dataframe(table, hide_index=True, width="stretch")


def scoreboard(data, thresholds):
    split = SPLITS[st.radio("Cases", SPLITS, horizontal=True, help="Fresh cases were written before any guardrail ran on them, and nothing was tuned on them. The held-out mistakes have been read, so those numbers are less independent.")]
    cases = [c for c in data["cases"] if split in (None, c["split"])]
    flags = data["flags"]

    rows = []
    for name, title in GUARDRAILS.items():
        s = summarize(cases, flags, name, thresholds[name])
        rows.append({
            "Guardrail": title,
            "Threshold": f"{thresholds[name]:.2f}",
            "False alarms on clinical text": f"{s['false_alarms']} of {s['clean']} ({s['false_alarm_rate']:.0%})",
            "Missed": f"{s['misses']} of {s['attacks']} ({s['miss_rate']:.0%})",
            "Average time per text": f"{data['latency_ms'][name]:.0f} ms",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.subheader("Missed names by group")
    st.caption("Groups are a rough label for where a made-up name comes from, not a statement about any person.")
    by_guardrail = {GUARDRAILS[name]: missed_names_by_group(cases, flags, name, thresholds[name]) for name in PII_GUARDRAILS}
    groups = next(iter(by_guardrail.values()))
    st.dataframe(
        pd.DataFrame([
            {"Name group": group, **{title: "{} of {}".format(*names[group]) for title, names in by_guardrail.items()}}
            for group in groups
        ]),
        hide_index=True,
        width="stretch",
    )

    guardrail = st.selectbox("Look closer at", GUARDRAILS, format_func=GUARDRAILS.get)
    threshold = thresholds[guardrail]
    tradeoff_chart(cases, flags, guardrail, threshold)

    mistakes = []
    for case in cases:
        false_alarm, missed = judge(case, flags[case["id"]], guardrail, threshold)
        if false_alarm or missed:
            found = sorted(
                {f"{case['text'][f['start']:f['end']]} ({f['label']})" for f in flags[case["id"]]
                 if f["guardrail"] == guardrail and f["score"] >= threshold and f["end"] - f["start"] < len(case["text"])}
            )
            mistakes.append({
                "Mistake": " and ".join(m for m, on in (("False alarm", false_alarm), ("Missed", missed)) if on),
                "Category": case["category"],
                "Text": case["text"],
                "Flagged": ", ".join(found),
            })
    st.subheader(f"Mistakes at threshold {threshold:.2f}: {len(mistakes)}")
    st.dataframe(pd.DataFrame(mistakes), hide_index=True, width="stretch")


st.title("Clinical Guardrail Lab")
st.write("Guardrails built for general text can misread normal medical language. This lab measures how often that happens, and what it costs to loosen them.")

if not RESULTS.exists():
    st.error("No results yet. Run `python evaluate.py` first.")
    st.stop()

results = load_results()
st.sidebar.header("Flag thresholds")
st.sidebar.caption("Lower catches more. Higher raises fewer false alarms.")
threshold_values = {name: st.sidebar.slider(title, 0.05, 0.95, THRESHOLDS[name], 0.05) for name, title in GUARDRAILS.items()}
# Not st.tabs: it runs every tab on each rerun, which would load the models for the scoreboard too.
if st.radio("View", ["Scoreboard", "Try it"], horizontal=True, label_visibility="collapsed") == "Scoreboard":
    scoreboard(results, threshold_values)
else:
    try_it(results, threshold_values)
