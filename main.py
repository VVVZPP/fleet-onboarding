import json
import os
import asyncio
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use relative paths so the app works on any server
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_FILE     = os.path.join(BASE_DIR, "submissions.json")
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")
STATIC_DIR    = os.path.join(BASE_DIR, "static")

# Admin PIN — read from environment variable if set, else fall back to default
ADMIN_PIN = os.environ.get("ADMIN_PIN", "cdgengie2025")

# OpenAI API key — read from environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ── Data helpers ──────────────────────────────────────────────

def load_data():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

def load_settings():
    defaults = {"dashboard_visible": False}
    if not os.path.exists(SETTINGS_FILE):
        return defaults
    with open(SETTINGS_FILE, "r") as f:
        s = json.load(f)
    return {**defaults, **s}

def save_settings(s):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(s, f, indent=2)

def require_admin(x_admin_pin: Optional[str]):
    if x_admin_pin != ADMIN_PIN:
        raise HTTPException(status_code=403, detail="Invalid admin PIN.")

# ── Models ────────────────────────────────────────────────────

class OptionAssessment(BaseModel):
    rec: Optional[str] = None   # "yes" | "cond" | "no"
    pros: str = ""
    cons: str = ""
    mitigations: str = ""

class Submission(BaseModel):
    name: str
    dept: str = ""
    o1: OptionAssessment
    o2: OptionAssessment
    o3: OptionAssessment
    extra: str = ""

class SettingsUpdate(BaseModel):
    dashboard_visible: Optional[bool] = None

# ── Routes ────────────────────────────────────────────────────

@app.post("/api/submit")
def submit(sub: Submission):
    data = load_data()
    entry = sub.dict()
    entry["timestamp"] = datetime.utcnow().isoformat()
    entry["id"] = len(data) + 1
    data.append(entry)
    save_data(data)
    return {"ok": True, "id": entry["id"], "total": len(data)}

@app.get("/api/submissions")
def get_submissions():
    return load_data()

@app.delete("/api/submissions")
def clear_submissions(x_admin_pin: Optional[str] = Header(default=None)):
    require_admin(x_admin_pin)
    save_data([])
    return {"ok": True}

# ── Settings ──────────────────────────────────────────────────

@app.get("/api/settings")
def get_settings():
    return load_settings()

@app.patch("/api/settings")
def update_settings(
    body: SettingsUpdate,
    x_admin_pin: Optional[str] = Header(default=None)
):
    require_admin(x_admin_pin)
    s = load_settings()
    if body.dashboard_visible is not None:
        s["dashboard_visible"] = body.dashboard_visible
    save_settings(s)
    return s

# ── Admin PIN verify ──────────────────────────────────────────

@app.post("/api/admin/verify")
def verify_admin(x_admin_pin: Optional[str] = Header(default=None)):
    if x_admin_pin != ADMIN_PIN:
        raise HTTPException(status_code=403, detail="Invalid PIN.")
    return {"ok": True}

# ── Rule-based Synthesis ─────────────────────────────────────

OPTION_NAMES = {
    "o1": "Option 1: Actual User Details",
    "o2": "Option 2: Dummy Accounts",
    "o3": "Option 3: Fleet Cards (Deprioritised)",
}

REC_LABELS = {"yes": "Yes", "cond": "Conditional", "no": "No"}

OVERALL_VERDICT = {
    "yes":  "Recommended",
    "cond": "Conditionally Recommended",
    "no":   "Not Recommended",
    None:   "No consensus",
}

def _majority(votes: list):
    """Return the most common vote value, or None if tied/empty."""
    if not votes:
        return None
    counts = {v: votes.count(v) for v in set(votes)}
    top = max(counts, key=lambda k: counts[k])
    # If there's a tie, prefer yes > cond > no
    max_count = counts[top]
    tied = [k for k, v in counts.items() if v == max_count]
    for pref in ["yes", "cond", "no"]:
        if pref in tied:
            return pref
    return top

def _bullet_list(texts: list) -> str:
    """Deduplicate and format a list of text entries as markdown bullets."""
    seen, out = set(), []
    for t in texts:
        for line in t.splitlines():
            line = line.strip().strip("-•").strip()
            if line and line.lower() not in seen:
                seen.add(line.lower())
                out.append(f"  - {line}")
    return "\n".join(out) if out else "  - None noted."

def _sentiment_label(yes_pct, cond_pct, no_pct):
    if yes_pct >= 60:
        return "Broadly positive — majority recommend."
    if no_pct >= 60:
        return "Broadly negative — majority do not recommend."
    if cond_pct >= 40:
        return "Mixed — significant conditional support with reservations."
    if yes_pct >= 40:
        return "Leaning positive with some conditions."
    return "Divided — no clear consensus."

@app.get("/api/synthesise")
def synthesise():
    """Generate a structured rule-based synthesis from all submissions."""
    data = load_data()
    if not data:
        raise HTTPException(status_code=400, detail="No submissions yet.")

    n = len(data)
    depts = sorted(set(s.get("dept", "") for s in data if s.get("dept", "")))
    dept_list = ", ".join(depts) if depts else "Various"

    # ── Per-option aggregation ────────────────────────────────
    option_stats = {}
    for opt in ["o1", "o2", "o3"]:
        votes, pros, cons, mits = [], [], [], []
        for s in data:
            o = s.get(opt, {})
            if o.get("rec"):
                votes.append(o["rec"])
            if o.get("pros", "").strip():
                pros.append(o["pros"].strip())
            if o.get("cons", "").strip():
                cons.append(o["cons"].strip())
            if o.get("mitigations", "").strip():
                mits.append(o["mitigations"].strip())
        yes_c  = votes.count("yes")
        cond_c = votes.count("cond")
        no_c   = votes.count("no")
        total_v = len(votes) or 1
        option_stats[opt] = {
            "votes": votes,
            "yes": yes_c, "cond": cond_c, "no": no_c,
            "yes_pct":  round(yes_c  / total_v * 100),
            "cond_pct": round(cond_c / total_v * 100),
            "no_pct":   round(no_c   / total_v * 100),
            "majority": _majority(votes),
            "pros": pros, "cons": cons, "mits": mits,
        }

    # ── Per-department aggregation ────────────────────────────
    dept_stats = {}
    for s in data:
        dept = s.get("dept", "Unknown")
        if dept not in dept_stats:
            dept_stats[dept] = {"count": 0, "o1": [], "o2": [], "o3": [], "extras": []}
        dept_stats[dept]["count"] += 1
        for opt in ["o1", "o2", "o3"]:
            rec = s.get(opt, {}).get("rec")
            if rec:
                dept_stats[dept][opt].append(rec)
        if s.get("extra", "").strip():
            dept_stats[dept]["extras"].append(s["extra"].strip())

    # ── Build markdown synthesis ──────────────────────────────
    lines = []
    lines.append("## Fleet Onboarding Options Review — Synthesis Report")
    lines.append(f"*Based on {n} submission{'s' if n != 1 else ''} from: {dept_list}*")
    lines.append("")
    lines.append("---")

    # Section 1: Overall Recommendation
    lines.append("### 1. Overall Recommendation")
    for opt in ["o1", "o2", "o3"]:
        st = option_stats[opt]
        verdict = OVERALL_VERDICT.get(st["majority"], "No consensus")
        lines.append(
            f"- **{OPTION_NAMES[opt]}** — {verdict} "
            f"(Yes: {st['yes']}, Conditional: {st['cond']}, No: {st['no']})"
        )
    lines.append("")
    lines.append("---")

    # Section 2: Per-Option Analysis
    lines.append("### 2. Per-Option Analysis")
    for opt in ["o1", "o2", "o3"]:
        st = option_stats[opt]
        sentiment = _sentiment_label(st["yes_pct"], st["cond_pct"], st["no_pct"])
        lines.append(f"")
        lines.append(f"- **{OPTION_NAMES[opt]}**")
        lines.append(f"  - *Sentiment:* {sentiment}")
        lines.append(f"  - *Vote breakdown:* Yes {st['yes_pct']}% · Conditional {st['cond_pct']}% · No {st['no_pct']}%")
        lines.append(f"  - **Key Pros:**")
        lines.append(_bullet_list(st["pros"]))
        lines.append(f"  - **Key Cons:**")
        lines.append(_bullet_list(st["cons"]))
        lines.append(f"  - **Mitigations Suggested:**")
        lines.append(_bullet_list(st["mits"]))
    lines.append("")
    lines.append("---")

    # Section 3: Per-Department Perspectives
    lines.append("### 3. Per-Department Perspectives")
    for dept, ds in sorted(dept_stats.items()):
        lines.append(f"")
        lines.append(f"- **{dept}** ({ds['count']} response{'s' if ds['count'] != 1 else ''})")
        for opt, label in [("o1", "Option 1"), ("o2", "Option 2"), ("o3", "Option 3")]:
            votes = ds[opt]
            if votes:
                tally = ", ".join(f"{REC_LABELS.get(v, v)} ×{votes.count(v)}" for v in sorted(set(votes), key=lambda x: ["yes","cond","no"].index(x) if x in ["yes","cond","no"] else 9))
                lines.append(f"  - {label}: {tally}")
            else:
                lines.append(f"  - {label}: No vote recorded")
        if ds["extras"]:
            lines.append(f"  - *Additional notes:*")
            for note in ds["extras"]:
                lines.append(f"    - {note}")
    lines.append("")
    lines.append("---")

    # Section 4: Key Risks & Open Questions
    lines.append("### 4. Key Risks & Open Questions")
    all_extras = [s.get("extra", "").strip() for s in data if s.get("extra", "").strip()]
    o3_cons = option_stats["o3"]["cons"]
    o2_cons = option_stats["o2"]["cons"]
    risks = []
    if option_stats["o3"]["no"] > 0:
        risks.append("Option 3 (Fleet Cards) faces charger incompatibility issues — past manual resolutions were required.")
    if option_stats["o2"]["cond"] + option_stats["o2"]["no"] > option_stats["o2"]["yes"]:
        risks.append("Option 2 (Dummy Accounts) raises concerns around manual credential management and lack of direct driver engagement.")
    if option_stats["o1"]["cond"] > 0:
        risks.append("Option 1 (Actual User Details) has conditional supporters — slower registration and duplicate account risk need to be addressed.")
    if all_extras:
        risks.append("Additional open questions raised by respondents:")
        for note in all_extras:
            risks.append(f"  - {note}")
    if not risks:
        risks.append("No specific risks or open questions were flagged beyond standard operational considerations.")
    for r in risks:
        lines.append(f"- {r}")
    lines.append("")
    lines.append("---")

    # Section 5: Suggested Next Steps
    lines.append("### 5. Suggested Next Steps")
    majority_o1 = option_stats["o1"]["majority"]
    majority_o2 = option_stats["o2"]["majority"]
    majority_o3 = option_stats["o3"]["majority"]

    if majority_o1 == "yes":
        lines.append("- **Proceed with Option 1** as the primary onboarding method — address duplicate account risk with a deduplication check at registration.")
    elif majority_o1 == "cond":
        lines.append("- **Explore Option 1** further — resolve conditional concerns (e.g. registration speed, duplicate accounts) before full rollout.")
    else:
        lines.append("- **Reconsider Option 1** — address the concerns raised before recommending it as the primary path.")

    if majority_o2 in ["yes", "cond"]:
        lines.append("- **Keep Option 2 as a fallback** for partners who cannot collect individual driver data upfront — document the escalation process clearly.")
    else:
        lines.append("- **Deprioritise Option 2** unless a specific partner use case requires it — ensure manual intervention processes are documented if used.")

    lines.append("- **Formally deprioritise Option 3** — resolve charger compatibility issues before reintroducing fleet cards as an option.")
    lines.append("- **Share this synthesis** with all participating departments for sign-off and alignment.")
    lines.append("- **Schedule a follow-up** to address open questions and finalise the recommended onboarding SOP.")
    lines.append("")
    lines.append("---")
    lines.append(f"*Synthesis generated automatically from {n} workshop submission{'s' if n != 1 else ''}.*")

    return {"synthesis": "\n".join(lines)}

# ── Static files ──────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
