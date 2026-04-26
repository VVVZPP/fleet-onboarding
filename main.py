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

# ── AI Synthesis ──────────────────────────────────────────────

@app.get("/api/synthesise")
async def synthesise():
    """Generate an AI synthesis using OpenAI (non-streaming)."""
    data = load_data()
    if not data:
        raise HTTPException(status_code=400, detail="No submissions yet.")

    lines = []
    for s in data:
        lines.append(
            f"- {s['name']} ({s.get('dept','—')}): "
            f"O1={s['o1']['rec'] or '?'} pros=\"{s['o1']['pros'][:80]}\" cons=\"{s['o1']['cons'][:80]}\" mit=\"{s['o1']['mitigations'][:80]}\"; "
            f"O2={s['o2']['rec'] or '?'} pros=\"{s['o2']['pros'][:80]}\" cons=\"{s['o2']['cons'][:80]}\" mit=\"{s['o2']['mitigations'][:80]}\"; "
            f"O3={s['o3']['rec'] or '?'} pros=\"{s['o3']['pros'][:80]}\" cons=\"{s['o3']['cons'][:80]}\" mit=\"{s['o3']['mitigations'][:80]}\"; "
            f"extra=\"{s.get('extra','')[:100]}\""
        )
    summary_text = "\n".join(lines)

    depts = list(set(s.get("dept","") for s in data if s.get("dept","")))
    dept_list = ", ".join(depts) if depts else "various"

    prompt = f"""You are an analyst synthesising workshop feedback for CDG ENGIE's fleet onboarding options review.

There are {len(data)} submissions from departments: {dept_list}.

The three options are:
- Option 1: Actual user details (Preferred) — drivers register with real name, phone, email.
- Option 2: Dummy accounts — mass accounts created upfront without individual driver details.
- Option 3: Fleet cards (Deprioritised) — physical cards issued per driver.

Raw submissions:
{summary_text}

Write a structured synthesis with the following sections:
1. **Overall Recommendation** — which option(s) are favoured and why.
2. **Per-Option Analysis** — for each option: aggregate sentiment, key pros, key cons, key mitigations suggested.
3. **Per-Department Perspectives** — for each department that submitted, summarise their stance and any unique concerns.
4. **Key Risks & Open Questions** — cross-cutting risks and unresolved questions raised.
5. **Suggested Next Steps** — concrete actions to move forward.

Be concise, professional, and actionable. Use bullet points within sections."""

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else AsyncOpenAI()

    try:
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1400,
        )
        text = response.choices[0].message.content
        return {"synthesis": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Static files ──────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
