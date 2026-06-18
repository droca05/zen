"""
demo_scenario.py
────────────────
Builds the SCARCITY scenario that powers the split-screen demo (Screen 4).

The whole thesis only becomes visible under scarcity. With abundant capacity
everyone gets served and naive == fair. So here we deliberately create a
resource-starved scenario where a greedy (naive) allocation systematically
under-serves one demographic group, and the demographic-parity constraint
corrects it — at a small, honest cost to raw utility.

This is the number you show the judge: "naive leaves group X behind;
fair closes the gap, serving N more people from the under-served group."

Author: Steff (Data Science + Math)
"""

from __future__ import annotations
import numpy as np

try:
    from .synthetic_data import UserProfile, Resource
    from .milp_solver import solve
except ImportError:
    from synthetic_data import UserProfile, Resource
    from milp_solver import solve

RNG = np.random.default_rng(7)


def build_scarcity_scenario():
    """
    Hand-built but realistic: one resource per service, low capacity, and a
    correlation between a demographic group and a disadvantage (distance/no
    transport) that a naive optimizer will quietly punish — exactly the
    VI-SPDAT failure mode.
    """
    # Two zones. Group 'Latino' users concentrated in zone 1 (farther from
    # the single food bank in zone 0) and less likely to have transport,
    # mirroring real spatial inequity in service access.
    users: list[UserProfile] = []
    uid = 0

    def add(needs, urgency, race, zone, transport, size=3, income=900):
        nonlocal uid
        users.append(UserProfile(
            user_id=f"U{uid:03d}", needs=needs, urgency=urgency,
            household_size=size, monthly_income=income, race_group=race,
            language="Spanish" if race == "Latino" else "English",
            has_transport=transport, zip_zone=zone,
        ))
        uid += 1

    # 10 White users, mostly in zone 0 (near resources), mostly with transport
    for _ in range(10):
        add(["food"], "this_week", "White", zone=0, transport=True)
    # 10 Latino users, mostly in zone 1 (far), mostly WITHOUT transport
    for _ in range(10):
        add(["food"], "this_week", "Latino", zone=1, transport=False)

    # Scarce but distributed: a food bank in EACH zone, but the zone-0 bank
    # (near White users) has more capacity than the zone-1 bank (near Latino
    # users). A naive optimizer fills the easy/high-score matches first and
    # under-serves the far group; the parity constraint forces the solver to
    # use the zone-1 capacity for the Latino group instead of double-serving
    # easy White matches.
    resources = [
        Resource(resource_id="R000", name="Central Food Bank",
                 service_type="food", zip_zone=0, capacity=10,
                 max_income=0, min_household_size=0, hours="Mon-Fri 9-5",
                 last_verified_days_ago=2),
        Resource(resource_id="R001", name="Eastside Community Fridge",
                 service_type="food", zip_zone=1, capacity=8,
                 max_income=0, min_household_size=0, hours="Daily 8-8",
                 last_verified_days_ago=5),
    ]
    return users, resources


def run():
    users, resources = build_scarcity_scenario()
    print("SCARCITY SCENARIO")
    print(f"  20 users (10 White near + transport, 10 Latino far + no transport)")
    print(f"  2 food banks, total capacity 18 for 20 users  →  scarcity forces tradeoffs\n")

    naive = solve(users, resources, fairness=False, max_distance=2.0)
    fair  = solve(users, resources, fairness=True, parity_delta=0.10, max_distance=2.0)

    def summary(tag, res):
        w = res.served_by_group.get("White", 0)
        l = res.served_by_group.get("Latino", 0)
        print(f"{tag}")
        print(f"  served total: {res.users_served}/{res.total_users}")
        print(f"  White served rate:  {w:.0%}")
        print(f"  Latino served rate: {l:.0%}")
        print(f"  parity gap: {res.parity_gap:.0%}")
        print()

    summary("── NAIVE (utility only) ──", naive)
    summary("── FAIR (demographic parity) ──", fair)

    return naive, fair


if __name__ == "__main__":
    run()
