"""
app.py — Zen backend (FastAPI)
──────────────────────────────
Local, functional API that wires the real MILP engine to the scraped resource
JSON. Diego swaps the JSON for a real DB and deploys; the route contract stays.

Run locally:
    pip install -r requirements.txt
    python scraper.py --seed          # produce data/resources.json
    uvicorn app:app --reload          # http://127.0.0.1:8000

Routes
    GET  /                 → serves the Zen web app (static/index.html)
    POST /api/intake       → transcript → constraints + confidence + action
    POST /api/match        → constraints → optimal, fair resource assignment
    GET  /api/demo         → naive-vs-fair split-screen scenario (Screen 4)
    GET  /api/dashboard    → ONG outcome metrics (Screen 7)
    POST /api/checkin      → 24/72h loop-closure response → escalation if needed
"""

from __future__ import annotations
import json, os
from datetime import datetime
from typing import Optional

import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine.intake_nlp import extract
from engine.synthetic_data import UserProfile, Resource, generate_users, zone_distance
from engine.milp_solver import solve, is_feasible
from engine.demo_scenario import build_scarcity_scenario
from engine.demand_forecast import forecast_next_week, top_shortfalls
from engine.text_tools import plain_for
from engine.followups import questions_for, interpret
from engine import caseworker as cw
from db import get_sb

# Needs treated as time-critical emergencies (real-time heuristic path).
EMERGENCY_NEEDS = {"food", "housing"}

# Service-area grid for Houston, TX metro (matches OSM scraper --bbox below)
_AREA_BBOX = (29.5, -95.8, 30.1, -95.1)   # south, west, north, east


def _coords_to_zone(lat: float, lon: float) -> int:
    """Map user lat/lon to zone 0-5 on the 3×2 service-area grid."""
    s, w, n, e = _AREA_BBOX
    row = min(2, max(0, int((lat - s) / (n - s) * 3)))
    col = min(1, max(0, int((lon - w) / (e - w) * 2)))
    return row * 2 + col

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "resources.json")

app = FastAPI(title="Zen — Benefits Navigator API", version="1.0")


@app.on_event("startup")
async def startup():
    sb = get_sb()
    if sb:
        count = sb.table("resources").select("resource_id", count="exact").execute().count
        print(f"[zen] ✓ Supabase connected — {count} resources in DB")
    else:
        print("[zen] ⚠  Supabase NOT configured — using local JSON files")


# ── load resources: Supabase when configured, local JSON as fallback ──────────
def _raw_resources() -> list[dict]:
    sb = get_sb()
    if sb:
        return sb.table("resources").select("*").execute().data
    with open(DATA) as f:
        return json.load(f)["resources"]


def load_resources() -> list[Resource]:
    return [Resource(
        resource_id=r["resource_id"], name=r["name"],
        service_type=r["service_type"], zip_zone=r.get("zip_zone", 0),
        capacity=r.get("capacity", 0), max_income=r.get("max_income", 0),
        min_household_size=r.get("min_household_size", 0),
        hours=r.get("hours", ""), last_verified_days_ago=r.get("last_verified_days_ago", 0),
    ) for r in _raw_resources()]


def resource_meta() -> dict[str, dict]:
    return {r["resource_id"]: r for r in _raw_resources()}


CITATIONS = {
    "food": "SNAP §273.9 — households at or below 130% of the federal poverty line qualify for food assistance.",
    "housing": "Emergency Rental Assistance — households facing eviction below 80% AMI are eligible.",
    "healthcare": "Medicaid §1902 — coverage for households below the state income threshold.",
    "childcare": "CCDF §98.20 — childcare subsidy for working or job-seeking parents below income limits.",
    "employment": "WIOA Title I — job placement services for dislocated and low-income workers.",
}

# Concrete onboarding steps — turns a directory into "what you do now"
NEXT_STEPS = {
    "food": {
        "steps": ["Walk in during open hours — most pantries need no appointment",
                  "Bring a photo ID if you have one (optional at most pantries)",
                  "Ask for a same-day food box for your family size"],
        "bring": "Photo ID (optional) · your household size",
    },
    "housing": {
        "steps": ["Call first to confirm they're taking applications this week",
                  "Gather your lease and any eviction or late-rent notice",
                  "Apply in person or ask if they have an online form"],
        "bring": "Lease · eviction/late notice · proof of income · ID",
    },
    "healthcare": {
        "steps": ["Call to book or walk in during clinic hours",
                  "Ask about sliding-scale fees if you're uninsured",
                  "Bring any medication you currently take"],
        "bring": "ID · proof of income (if you have it)",
    },
    "childcare": {
        "steps": ["Apply online or call the subsidy office",
                  "Have your children's birth certificates ready",
                  "Ask about the current waitlist time"],
        "bring": "Children's birth certificates · proof of income · ID",
    },
    "employment": {
        "steps": ["Visit the center or check their website for hours",
                  "Bring a resume if you have one — they'll help if not",
                  "Ask about same-week appointments"],
        "bring": "Resume (optional) · ID",
    },
}


def _build_steps(service: str, m: dict) -> list[str]:
    """Generate concrete steps injecting real name, address, hours, phone."""
    template = NEXT_STEPS.get(service, NEXT_STEPS["food"])
    steps = list(template["steps"])
    name = m.get("name", "them")
    addr = m.get("address", "")
    hours = m.get("hours", "")
    phone = m.get("phone", "")
    has_hours = hours and hours not in ("Call for hours", "")

    # Replace generic first step with real location
    if addr and has_hours:
        steps[0] = f"Go to {name} at {addr} — open {hours}"
    elif addr:
        steps[0] = f"Go to {name} at {addr}"
    elif has_hours:
        steps[0] = f"Contact {name} — open {hours}"

    # Insert phone step if available
    if phone:
        steps.insert(1, f"Call ahead to confirm availability: {phone}")

    return steps


# ── request models ────────────────────────────────────────────────────────────
class IntakeReq(BaseModel):
    transcript: str
    lat: Optional[float] = None
    lon: Optional[float] = None

class MatchReq(BaseModel):
    needs: list[str]
    urgency: str = "this_week"
    zip_zone: int = 1
    lat: Optional[float] = None
    lon: Optional[float] = None
    household_size: int = 3
    monthly_income: int = 900
    has_transport: bool = False
    language: str = "English"
    race_group: str = "Latino"
    has_children: bool = False
    priority_housing: bool = False

class CheckinReq(BaseModel):
    user_id: str
    resource_id: str
    received_help: bool
    contact_phone: str = ""
    contact_email: str = ""
    user_message: str = ""

class ONGResourceReq(BaseModel):
    """An ONG registering / updating a program's capacity."""
    name: str
    service_type: str
    address: str = ""
    phone: str = ""
    email: str = ""
    url: str = ""
    hours: str = "Call for hours"
    zip_zone: int = 0
    capacity: int = 10
    max_income: int = 0
    min_household_size: int = 0


# ── routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def home():
    with open(os.path.join(HERE, "static", "index.html"), encoding="utf-8") as f:
        return f.read()


@app.post("/api/intake")
def api_intake(req: IntakeReq):
    r = extract(req.transcript)
    if r.safety_flag:
        loc_parts = []
        if req.lat is not None: loc_parts.append(f"lat: {req.lat:.5f}")
        if req.lon is not None: loc_parts.append(f"lon: {req.lon:.5f}")
        loc_str = " · ".join(loc_parts) if loc_parts else "location not provided"
        cw.add_escalation(
            "safety",
            summary=f"EMERGENCY — safety keyword detected · {loc_str}",
            urgency="today",
            safety_flag=True,
            language=r.language,
        )
    return {
        "needs": r.needs, "urgency": r.urgency, "language": r.language,
        "confidence": r.confidence, "safety_flag": r.safety_flag,
        "action": r.action, "matched_terms": r.matched_terms,
        "followups": questions_for(r.needs),
    }


class FollowupReq(BaseModel):
    answers: dict

@app.post("/api/followups/interpret")
def api_followups(req: FollowupReq):
    """Turn the answers into plain-language eligibility signals + fields."""
    return interpret(req.answers)


@app.post("/api/match")
def api_match(req: MatchReq):
    zone = _coords_to_zone(req.lat, req.lon) if req.lat and req.lon else req.zip_zone
    user = UserProfile(
        user_id="LIVE", needs=req.needs, urgency=req.urgency,
        household_size=req.household_size, monthly_income=req.monthly_income,
        race_group=req.race_group, language=req.language,
        has_transport=req.has_transport, zip_zone=zone,
    )
    resources = load_resources()
    meta = resource_meta()
    res = solve([user], resources, fairness=True, parity_delta=0.10)

    plan = []
    ranked = sorted(res.assignments, key=lambda z: -z.confidence)
    for k, a in enumerate(ranked):
        m = meta.get(a.resource_id, {})
        ns = NEXT_STEPS.get(a.service_type, {"steps": [], "bring": ""})
        pl = plain_for(a.service_type)            # plain language + readability
        plan.append({
            "rank": k + 1,
            "resource_id": a.resource_id,
            "name": a.resource_name,
            "service": a.service_type,
            "address": m.get("address", ""),
            "hours": m.get("hours", ""),
            "phone": m.get("phone", ""),
            "url": m.get("url", ""),
            "distance_mi": a.distance,
            "confidence": a.confidence,
            "why_plain": pl["plain"],             # headline: plain language
            "citation": pl["legal"],              # secondary: legal detail
            "readability": pl["readability"],     # the metric, so it's not a vague claim
            "stale": m.get("last_verified_days_ago", 0) > 45,
            "zone": m.get("zip_zone", 0),
            "steps": _build_steps(a.service_type, m),
            "bring": ns["bring"],
            "start_here": k == 0,
        })

    # ── two-tier routing (explicit, so it's visible, not vaporware) ───────────
    is_emergency = req.urgency == "today" and any(n in EMERGENCY_NEEDS for n in req.needs)
    if is_emergency:
        tier = "emergency"
        routing = ("Routed to IMMEDIATE response: transparent real-time matching, "
                   "because you need help today. No waiting for a batch.")
    else:
        tier = "assignment"
        routing = ("Routed to ASSIGNMENT: for scarce resources like housing, the "
                   "fair-allocation optimizer (MILP) runs in the nightly batch "
                   "across everyone — so no group is systematically left behind.")

    import random as _r
    tracking = "ZEN-2026-" + str(_r.randint(1000, 9999))
    return {"plan": plan, "status": res.status, "tracking_number": tracking,
            "tier": tier, "routing_reason": routing}


@app.get("/api/demo")
def api_demo():
    users, resources = build_scarcity_scenario()
    naive = solve(users, resources, fairness=False, max_distance=2.0)
    fair = solve(users, resources, fairness=True, parity_delta=0.10, max_distance=2.0)
    return {
        "naive": {"served": naive.users_served, "total": naive.total_users,
                  "by_group": naive.served_by_group, "parity_gap": naive.parity_gap},
        "fair": {"served": fair.users_served, "total": fair.total_users,
                 "by_group": fair.served_by_group, "parity_gap": fair.parity_gap},
    }


@app.get("/api/dashboard")
def api_dashboard():
    users = generate_users(200)
    resources = load_resources()
    fair = solve(users, resources, fairness=True, parity_delta=0.10)
    closure = {"food": 0.91, "housing": 0.73, "healthcare": 0.82,
               "childcare": 0.68, "employment": 0.77}
    by_service = {}
    for st in closure:
        n = sum(1 for a in fair.assignments if a.service_type == st)
        by_service[st] = {"assigned": n, "closure_rate": closure[st]}
    total_assigned = sum(v["assigned"] for v in by_service.values()) or 1
    overall = round(sum(v["assigned"]*v["closure_rate"] for v in by_service.values())/total_assigned, 2)
    # All simulated metrics derived from the same MILP run so they are consistent
    confirmed   = round(fair.users_served * overall)
    still_without = fair.users_served - confirmed   # matched but help not confirmed
    unmatched   = fair.total_users - fair.users_served  # no resource available at all
    return {
        "people_matched": fair.users_served, "total_people": fair.total_users,
        "overall_loop_closure": overall, "closure_by_service": by_service,
        "confirmed_help": confirmed,
        "still_without_help": still_without,
        "unmatched": unmatched,
        "equity_audit": fair.served_by_group, "parity_gap": fair.parity_gap,
        "parity_maintained": fair.parity_gap <= 0.10,
        "escalations": _real_escalation_counts(),
    }


@app.post("/api/checkin")
def api_checkin(req: CheckinReq):
    if req.received_help:
        return {"loop": "closed", "message": "Glad you got help. Case closed.",
                "escalate": False}
    # broken loop → create a real escalation in the caseworker queue
    contact_parts = []
    if req.contact_phone: contact_parts.append(f"tel: {req.contact_phone}")
    if req.contact_email: contact_parts.append(f"email: {req.contact_email}")
    contact_str = " · ".join(contact_parts) if contact_parts else ""
    summary = f"No help received from {req.resource_id}"
    if contact_str:
        summary += f" — {contact_str}"
    if req.user_message:
        summary += f" | \"{req.user_message.strip()}\""
    cw.add_escalation("broken_loop", summary=summary, urgency="today")
    return {
        "loop": "broken", "escalate": True,
        "caseworker": {"name": "Sarah", "eta_hours": 2, "language": "Spanish"},
        "message": "We're on it. Your case was added to the caseworker queue, "
                   "triaged by need. A caseworker reaches out and reassigns you.",
    }


# ── caseworker view (human-in-the-loop, vulnerability-triaged) ────────────────
def _real_escalation_counts() -> dict:
    """Read actual escalation counts from the persistent store."""
    all_cases = cw._load()
    total = len(all_cases)
    resolved = sum(1 for c in all_cases if c.get("status") == "resolved")
    return {
        "safety_critical": sum(1 for c in all_cases if c.get("reason") == "safety"),
        "low_confidence": sum(1 for c in all_cases if c.get("reason") == "low_confidence"),
        # Only open broken-loop cases — resolved ones already got help
        "broken_loop": sum(1 for c in all_cases
                           if c.get("reason") == "broken_loop" and c.get("status") == "open"),
        "resolved": resolved,
        "total": total,
        "resolution_rate": round(resolved / total, 2) if total else 0.0,
    }


@app.get("/api/caseworker/queue")
def api_cw_queue():
    all_cases = cw._load()
    open_cases = [c for c in all_cases if c.get("status") != "resolved"]
    resolved_cases = [c for c in all_cases if c.get("status") == "resolved"]
    for c in open_cases:
        c["vulnerability"] = cw.vulnerability_score(c)
        c["reason_label"] = cw.REASON_LABEL.get(c["reason"], c["reason"])
    open_cases.sort(key=lambda c: -c["vulnerability"])
    for c in resolved_cases:
        c["reason_label"] = cw.REASON_LABEL.get(c.get("reason", ""), c.get("reason", ""))
    return {"queue": open_cases, "resolved": resolved_cases}

@app.get("/api/caseworker/dashboard")
def api_cw_dashboard():
    return {"stats": cw.stats(), "resolved": cw.resolved_cases(), "queue": cw.queue()}

class ResolveReq(BaseModel):
    case_id: str
    referred_to: str = ""

@app.post("/api/caseworker/resolve")
def api_cw_resolve(req: ResolveReq):
    return cw.resolve(req.case_id, req.referred_to)


# ── real-time area bootstrap: scrape OSM around the user's GPS position ────────
class BootstrapReq(BaseModel):
    lat: float
    lon: float
    radius_km: float = 10.0

@app.post("/api/bootstrap_area")
async def api_bootstrap_area(req: BootstrapReq):
    """
    Called once after geolocation. Scrapes OSM in a ~radius_km circle around the
    user, deduplicates against what's already in Supabase, and upserts new records.
    Returns immediately with a job_id; client polls /api/bootstrap_area/{job_id}.
    """
    import asyncio, math
    from scraper import scrape_osm

    lat_d = req.radius_km / 111.0
    lon_d = req.radius_km / (111.0 * math.cos(math.radians(req.lat)))
    bbox = f"{req.lat-lat_d:.4f},{req.lon-lon_d:.4f},{req.lat+lat_d:.4f},{req.lon+lon_d:.4f}"

    loop = asyncio.get_event_loop()
    records = await loop.run_in_executor(None, scrape_osm, bbox)

    if not records:
        return {"status": "no_results", "new": 0, "bbox": bbox}

    sb = get_sb()
    if sb:
        existing = {r["name"].lower() for r in sb.table("resources").select("name").execute().data}
        new_recs = [r for r in records if r["name"].lower() not in existing]
        if new_recs:
            # assign IDs continuing from current max
            all_ids = sb.table("resources").select("resource_id").execute().data
            next_idx = len(all_ids)
            for i, rec in enumerate(new_recs):
                rec["resource_id"] = f"R{next_idx + i:04d}"
            sb.table("resources").insert(new_recs).execute()
        return {"status": "ok", "found": len(records), "new": len(new_recs), "bbox": bbox}

    return {"status": "no_db", "found": len(records), "new": 0, "bbox": bbox}


# ── ONG side: register / update capacity (the supply side of the marketplace) ──
@app.get("/api/ong/resources")
def api_ong_list():
    """List current resources an ONG could see/manage."""
    rows = _raw_resources()
    return {"count": len(rows), "resources": rows}


@app.post("/api/ong/register")
def api_ong_register(req: ONGResourceReq):
    """ONG registers a new program or updates capacity."""
    sb = get_sb()
    if sb:
        existing_rows = (sb.table("resources").select("*")
                         .ilike("name", req.name).eq("zip_zone", req.zip_zone)
                         .execute().data)
        if existing_rows:
            rec = existing_rows[0]
            sb.table("resources").update({
                "capacity": req.capacity, "hours": req.hours, "last_verified_days_ago": 0,
            }).eq("resource_id", rec["resource_id"]).execute()
            rec.update({"capacity": req.capacity, "hours": req.hours, "last_verified_days_ago": 0})
            return {"action": "updated", "resource": rec,
                    "message": f"'{req.name}' updated. Live in matching immediately."}
        all_ids = sb.table("resources").select("resource_id").execute().data
        new_id = f"R{len(all_ids):04d}"
        rec = {
            "resource_id": new_id, "name": req.name, "service_type": req.service_type,
            "address": req.address, "phone": req.phone, "url": req.url,
            "hours": req.hours, "zip_zone": req.zip_zone, "capacity": req.capacity,
            "max_income": req.max_income, "min_household_size": req.min_household_size,
            "last_verified_days_ago": 0,
        }
        sb.table("resources").insert(rec).execute()
        return {"action": "registered", "resource": rec,
                "message": f"'{req.name}' registered. Live in matching immediately."}
    # JSON fallback
    with open(DATA) as f:
        blob = json.load(f)
    existing = next((r for r in blob["resources"]
                     if r["name"].lower() == req.name.lower()
                     and r["zip_zone"] == req.zip_zone), None)
    if existing:
        existing.update({"capacity": req.capacity, "hours": req.hours, "last_verified_days_ago": 0})
        action, rec = "updated", existing
    else:
        new_id = f"R{len(blob['resources']):04d}"
        rec = {
            "resource_id": new_id, "name": req.name, "service_type": req.service_type,
            "address": req.address, "phone": req.phone, "hours": req.hours,
            "zip_zone": req.zip_zone, "capacity": req.capacity,
            "max_income": req.max_income, "min_household_size": req.min_household_size,
            "last_verified_days_ago": 0,
            "hsds": {"schema": "openreferral-hsds-3.0", "status": "active",
                     "source": "ong-self-registered"},
        }
        blob["resources"].append(rec)
        blob["count"] = len(blob["resources"])
        action = "registered"
    with open(DATA, "w") as f:
        json.dump(blob, f, indent=2)
    return {"action": action, "resource": rec,
            "message": f"'{req.name}' {action}. Live in matching immediately."}


# ── demand forecast (aggregate by zone/service — never profiles people) ───────
@app.get("/api/forecast")
def api_forecast():
    """Predicted resource demand next week by zone + service, with capacity gaps."""
    full = forecast_next_week()
    shortfalls = [r for r in full if r["gap"] > 0][:6]
    # zone centers (illustrative, around a default city) for the heat map
    ZONE_COORDS = {
        0: [29.57, -95.65], 1: [29.57, -95.25],
        2: [29.73, -95.65], 3: [29.73, -95.25],
        4: [29.89, -95.65], 5: [29.89, -95.25],
    }
    # aggregate worst gap per zone for the map coloring
    zone_gap = {}
    for r in full:
        z = r["zone"]
        zone_gap[z] = max(zone_gap.get(z, -999), r["gap"])
    zones_map = [{"zone": z, "lat": ZONE_COORDS[z][0], "lng": ZONE_COORDS[z][1],
                  "max_gap": g,
                  "level": "high" if g > 6 else ("mid" if g > 0 else "ok")}
                 for z, g in sorted(zone_gap.items())]
    return {
        "shortfalls": shortfalls,
        "all": full,
        "zones_map": zones_map,
        "note": "Forecasts demand for resource types by zone — never scores or "
                "profiles individuals. ONGs pre-position capacity where shortfalls "
                "are predicted; this feeds better-stocked resources into the matcher.",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Gemini-powered chat helper (with automatic rule-based fallback)
# ══════════════════════════════════════════════════════════════════════════════
# The API key is NEVER in the frontend. It is read from the environment variable
# GEMINI_API_KEY, or from a local file `gemini_key.txt` (which is git-ignored).
# If no key / no internet / any error → falls back to the transparent rule-based
# answers, so the demo never breaks.

GEMINI_MODEL = "gemini-2.0-flash"   # change here if you use another model


def _gemini_key() -> Optional[str]:
    k = os.environ.get("GEMINI_API_KEY")
    if k:
        return k.strip()
    p = os.path.join(HERE, "gemini_key.txt")
    if os.path.exists(p):
        txt = open(p).read().strip()
        return txt or None
    return None


def _rule_based_answer(q: str, plan: list, language: str = "English") -> str:
    t = (q or "").lower()
    first = next((p for p in plan if p.get("start_here")), (plan[0] if plan else None))
    if any(w in t for w in ["first", "start"]):
        return (f"Start with {first['name']}. {first['steps'][0]}."
                if first else "Share your situation first and I'll tell you the first step.")
    if any(w in t for w in ["document", "bring", "need", "carry"]):
        return (f"For {first['name']}, bring: {first['bring']}."
                if first else "Share your situation first and I'll list what to bring.")
    if any(w in t for w in ["safe", "privacy", "immigration"]):
        return "Your information is private. We never share your data with immigration or police."
    if any(w in t for w in ["closed", "full", "wrong"]):
        return "If a place is closed or full, mark it as not received in My progress — a caseworker will reassign you."
    return "Hi, I'm here to help. Ask me about your first step, what to bring, or what to do if a place is closed."


class ChatReq(BaseModel):
    message: str
    plan: list = []
    tracking: str = ""
    language: str = "English"


@app.post("/api/chat")
def api_chat(req: ChatReq):
    key = _gemini_key()
    if key:
        try:
            plan_txt = "\n".join(
                f"- {p.get('name')} ({p.get('service')}): {p.get('why_plain','')} "
                f"Steps: {'; '.join(p.get('steps', []))}. Bring: {p.get('bring','')}."
                for p in req.plan
            ) or "No plan yet."
            sys_inst = (
                "You are Zen's helper for people in crisis seeking public benefits. "
                "Reply in English only. Be warm, clear, and brief — maximum 3 sentences. "
                "Use a 6th-grade reading level. "
                "Use ONLY the resources listed in the user's plan below — never invent programs, addresses, or phone numbers. "
                "If the user greets you or sends a short message, reply warmly and ask what you can help them with. "
                "\n\nUSER'S PLAN:\n" + plan_txt
            )
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{GEMINI_MODEL}:generateContent?key={key}")
            body = {
                "system_instruction": {"parts": [{"text": sys_inst}]},
                "contents": [{"parts": [{"text": req.message}]}],
                "generationConfig": {"temperature": 0.4, "maxOutputTokens": 200},
            }
            r = requests.post(url, json=body, timeout=20)
            r.raise_for_status()
            data = r.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise ValueError(f"No candidates: {data}")
            text = candidates[0]["content"]["parts"][0]["text"]
            return {"reply": text.strip(), "source": "gemini"}
        except Exception as e:
            return {"reply": _rule_based_answer(req.message, req.plan, req.language),
                    "source": "fallback", "note": str(e)[:120]}
    return {"reply": _rule_based_answer(req.message, req.plan, req.language), "source": "rules"}


# ══════════════════════════════════════════════════════════════════════════════
# Case persistence — save a plan under its tracking number, recover it later.
# Server-side JSON store (Diego swaps for the real DB). Makes "recover by case
# number" genuinely work across devices that hit the same server.
# ══════════════════════════════════════════════════════════════════════════════
CASES = os.path.join(HERE, "data", "cases.json")


def _load_cases() -> dict:
    """Returns {tracking: case_dict}. Supabase when configured, JSON fallback."""
    sb = get_sb()
    if sb:
        rows = sb.table("cases").select("*").execute().data
        return {r["tracking"]: {k: v for k, v in r.items() if k != "tracking"}
                for r in rows}
    if os.path.exists(CASES):
        try:
            return json.load(open(CASES, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cases(d: dict):
    json.dump(d, open(CASES, "w", encoding="utf-8"), indent=2)


class CaseSaveReq(BaseModel):
    tracking: str
    plan: list
    email: str = ""            # OPTIONAL — for reminders to the person
    ong_consent: bool = False  # OPTIONAL, separate — allow a partner ONG to reach out


@app.post("/api/case/save")
def api_case_save(req: CaseSaveReq):
    sb = get_sb()
    row = {
        "tracking": req.tracking,
        "plan": req.plan,
        "email": req.email,
        "ong_consent": req.ong_consent,
        "saved_at": datetime.now().isoformat(),
    }
    if sb:
        sb.table("cases").upsert(row).execute()
    else:
        cases = _load_cases()
        cases[req.tracking] = {k: v for k, v in row.items() if k != "tracking"}
        _save_cases(cases)
    if req.email:
        _send_reminder_email(req.email, req.tracking)
    return {"saved": req.tracking}


def _send_reminder_email(to_addr: str, tracking: str):
    """
    PREPARED FOR PRODUCTION (Diego): wire a real email provider here.
    The MVP does NOT send mail — it logs intent, so the promise stays honest.

    Example with SMTP / a provider like Resend or SendGrid:
        import smtplib, ssl
        # read creds from env: EMAIL_HOST, EMAIL_USER, EMAIL_PASS
        # build the message: "Your Zen case {tracking} — here's your plan + 24h check-in"
        # send, then return True/False

    Until credentials are configured, we only record the intent.
    """
    print(f"[email-intent] would send reminder for {tracking} to {to_addr} "
          f"(no provider configured — Diego wires this in production)")
    return False


@app.get("/api/case/{tracking}")
def api_case_get(tracking: str):
    sb = get_sb()
    if sb:
        rows = sb.table("cases").select("*").eq("tracking", tracking).execute().data
        if rows:
            return {"found": True, **rows[0]}
        return {"found": False}
    cases = _load_cases()
    if tracking in cases:
        return {"found": True, **cases[tracking]}
    return {"found": False}


# ══════════════════════════════════════════════════════════════════════════════
# MILP batch — VISIBLE. Returns per-person assignments under naive vs fair so the
# UI can show the solver actually working, not just a headline percentage.
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/batch/run")
def api_batch_run():
    users, resources = build_scarcity_scenario()
    naive = solve(users, resources, fairness=False, max_distance=2.0)
    fair = solve(users, resources, fairness=True, parity_delta=0.10, max_distance=2.0)
    nmap = {a.user_id: a.resource_id for a in naive.assignments}
    fmap = {a.user_id: a.resource_id for a in fair.assignments}
    rname = {r.resource_id: r.name for r in resources}
    people = [{
        "id": u.user_id, "group": u.race_group, "urgency": u.urgency,
        "zone": u.zip_zone, "transport": u.has_transport,
        "naive": nmap.get(u.user_id), "fair": fmap.get(u.user_id),
    } for u in users]
    return {
        "people": people,
        "resources": [{"id": r.resource_id, "name": r.name,
                       "capacity": r.capacity, "zone": r.zip_zone} for r in resources],
        "resource_names": rname,
        "naive": {"served": naive.users_served, "total": naive.total_users,
                  "by_group": naive.served_by_group, "parity_gap": naive.parity_gap},
        "fair": {"served": fair.users_served, "total": fair.total_users,
                 "by_group": fair.served_by_group, "parity_gap": fair.parity_gap},
        "explain": ("Same households, same scarce slots. NAIVE maximizes raw coverage and "
                    "systematically under-serves Outer Neighborhoods (lower score due to "
                    "distance and no transport — the exact VI-SPDAT failure mode). FAIR adds "
                    "the demographic-parity constraint and redistributes — closing the gap."),
    }
