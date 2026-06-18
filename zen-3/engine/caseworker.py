"""
caseworker.py
─────────────
The human-in-the-loop made real. The judges' feedback was blunt: a caseworker
who "reaches out in 2 hours" is a fantasy on a system whose premise is that
human time is SCARCE. So the queue is NOT first-come-first-served.

It is a VULNERABILITY-TRIAGED queue: the scarce caseworker time is spent on the
highest-need cases first. Families with children, safety flags, and same-day
urgency rise to the top; lower-stakes cases wait.

This directly answers the "infinite queue" attack: Zen does not assume infinite
caseworker time — it assumes scarce time and optimizes which cases get it.

State persists in escalations.json so "resolve" actually changes something.

Author: Steff (Data Science + Math)
"""

from __future__ import annotations
import json, os
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE = os.path.join(HERE, "data", "escalations.json")


# ── vulnerability score: what pushes a case up the scarce-time queue ──────────
def vulnerability_score(case: dict) -> int:
    s = 0
    if case.get("safety_flag"):        s += 100      # safety always first
    if case.get("has_children"):       s += 40
    if case.get("urgency") == "today": s += 30
    if case.get("reason") == "broken_loop": s += 25  # help didn't arrive
    if case.get("reason") == "low_confidence": s += 10
    if case.get("no_id"):              s += 5
    return s


REASON_LABEL = {
    "safety": "Safety keyword detected",
    "low_confidence": "Voice intake unclear after 2 tries",
    "broken_loop": "Reported they did NOT receive help",
}


def _seed():
    """A few realistic escalations so the queue isn't empty in the demo."""
    now = datetime.now()
    return [
        {"id": "C001", "user_hash": "u_8f3a", "reason": "safety", "safety_flag": True,
         "summary": "Mentioned fleeing an unsafe home tonight", "urgency": "today",
         "has_children": True, "flagged_at": (now - timedelta(minutes=6)).isoformat(),
         "status": "open", "language": "Spanish"},
        {"id": "C002", "user_hash": "u_2b91", "reason": "broken_loop", "safety_flag": False,
         "summary": "Food bank was full when they arrived", "urgency": "today",
         "has_children": True, "flagged_at": (now - timedelta(hours=3)).isoformat(),
         "status": "open", "language": "Spanish"},
        {"id": "C003", "user_hash": "u_d7c4", "reason": "low_confidence", "safety_flag": False,
         "summary": "Heavy accent, transcript unclear on housing need", "urgency": "this_week",
         "has_children": False, "no_id": True, "flagged_at": (now - timedelta(hours=1)).isoformat(),
         "status": "open", "language": "Other"},
        {"id": "C004", "user_hash": "u_5e22", "reason": "broken_loop", "safety_flag": False,
         "summary": "Clinic didn't answer; still needs care", "urgency": "this_week",
         "has_children": False, "flagged_at": (now - timedelta(hours=20)).isoformat(),
         "status": "open", "language": "English"},
    ]


def _load():
    if not os.path.exists(STORE):
        save(_seed())
    with open(STORE) as f:
        return json.load(f)


def save(cases):
    with open(STORE, "w") as f:
        json.dump(cases, f, indent=2)


def queue():
    """Open cases, triaged by vulnerability (highest need first)."""
    cases = [c for c in _load() if c["status"] == "open"]
    for c in cases:
        c["vulnerability"] = vulnerability_score(c)
        c["reason_label"] = REASON_LABEL.get(c["reason"], c["reason"])
    cases.sort(key=lambda c: -c["vulnerability"])
    return cases


def resolve(case_id: str, referred_to: str = ""):
    cases = _load()
    for c in cases:
        if c["id"] == case_id:
            c["status"] = "resolved"
            c["resolved_at"] = datetime.now().isoformat()
            c["referred_to"] = referred_to or "Direct assistance"
    save(cases)
    return {"resolved": case_id, "referred_to": referred_to or "Direct assistance"}


def resolved_cases():
    """Closed cases, most recent first, with where each was referred."""
    cases = [c for c in _load() if c["status"] == "resolved"]
    for c in cases:
        c["reason_label"] = REASON_LABEL.get(c["reason"], c["reason"])
    cases.sort(key=lambda c: c.get("resolved_at", ""), reverse=True)
    return cases


def stats():
    """Panorama metrics for the caseworker dashboard."""
    cases = _load()
    openc = [c for c in cases if c["status"] == "open"]
    closed = [c for c in cases if c["status"] == "resolved"]
    # referrals grouped by ONG
    by_ong = {}
    for c in closed:
        ong = c.get("referred_to", "Direct assistance")
        by_ong[ong] = by_ong.get(ong, 0) + 1
    return {
        "total": len(cases),
        "open": len(openc),
        "resolved": len(closed),
        "resolution_rate": round(len(closed) / len(cases), 2) if cases else 0,
        "referrals_by_ong": by_ong,
    }


def add_escalation(reason: str, summary: str, **kw):
    cases = _load()
    cid = f"C{len(cases)+1:03d}"
    cases.append({"id": cid, "user_hash": f"u_{cid}", "reason": reason,
                  "summary": summary, "flagged_at": datetime.now().isoformat(),
                  "status": "open", **kw})
    save(cases)
    return cid


if __name__ == "__main__":
    if os.path.exists(STORE):
        os.remove(STORE)
    print("Caseworker queue (vulnerability-triaged, NOT first-come-first-served):\n")
    for c in queue():
        print(f"  [{c['vulnerability']:>3}] {c['id']} · {c['reason_label']:35s} "
              f"{'👶' if c.get('has_children') else '  '} {c['urgency']:10s} {c['summary']}")
    print("\nResolving C001…"); resolve("C001")
    print("Remaining open:", [c["id"] for c in queue()])
