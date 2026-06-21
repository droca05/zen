"""
demo_scenario.py
────────────────
Scarcity scenario powering the split-screen MILP demo.

The thesis only becomes visible under scarcity. With abundant capacity
everyone gets served and naive == fair. Here we build a resource-starved
scenario where a greedy (naive) allocation systematically under-serves
families in Houston's outer neighborhoods — exactly the VI-SPDAT failure mode.

Context: Houston, TX metro area.
  - Inner Loop (zones 2-3): near downtown, transit access, higher match scores.
  - Outer Neighborhoods (zones 0-1): SW/NE Houston, no car, lower match scores.

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
    20 households contesting ONE scarce central food bank (13 slots).

    Both groups can REACH the hub (Inner Loop live next to it; Outer
    Neighborhoods are one zone out but have transport, so they are eligible
    and feasible). Scarcity is real: 13 slots for 20 families.

    Naive (utility-only) serves the highest-scoring matches first. Inner Loop
    families score higher on proximity (they are closer), so a greedy optimizer
    serves ALL of them and only a few Outer Neighborhood families — a disparate
    impact that looks "efficient" but is just biased toward whoever lives closer.

    Fair: the demographic-parity constraint reassigns contested slots so both
    groups are served at comparable rates. It does NOT leave slots empty — it
    serves the SAME number of families, redistributed. Outer rises from 30% to
    60%; Inner eases from 100% to 70%; total served stays 13.
    """
    users: list[UserProfile] = []
    uid = 0

    def add(needs, urgency, group, zone, transport, size=3, income=1100):
        nonlocal uid
        users.append(UserProfile(
            user_id=f"U{uid:03d}", needs=needs, urgency=urgency,
            household_size=size, monthly_income=income, race_group=group,
            language="English", has_transport=transport, zip_zone=zone,
        ))
        uid += 1

    # 10 Inner Loop households — zone 3 (Midtown), next to the hub, with transport
    for _ in range(10):
        add(["food"], "this_week", "Inner Loop", zone=3, transport=True, income=1400)

    # 10 Outer Neighborhood households — zone 2 (one zone out), WITH transport so
    # they can reach and contest the same hub (eligible + feasible, just farther,
    # so a greedy optimizer ranks them lower on proximity).
    for _ in range(10):
        add(["food"], "this_week", "Outer Neighborhoods", zone=2, transport=True, income=1100)

    resources = [
        Resource(resource_id="R000",
                 name="Houston Food Bank – Central Hub",
                 service_type="food", zip_zone=3, capacity=13,
                 max_income=0, min_household_size=0,
                 hours="Mon–Fri 9 am–5 pm", last_verified_days_ago=1),
    ]
    return users, resources


def run():
    users, resources = build_scarcity_scenario()
    print("SCARCITY SCENARIO — Houston, TX")
    print("  20 households (10 Inner Loop next to hub, 10 Outer Neighborhoods one zone out)")
    print("  1 central food bank, 13 slots for 20 households → scarcity forces trade-offs\n")

    naive = solve(users, resources, fairness=False, max_distance=2.0)
    fair  = solve(users, resources, fairness=True, parity_delta=0.10, max_distance=2.0)

    def summary(tag, res):
        i = res.served_by_group.get("Inner Loop", 0)
        o = res.served_by_group.get("Outer Neighborhoods", 0)
        print(f"{tag}")
        print(f"  served total: {res.users_served}/{res.total_users}")
        print(f"  Inner Loop served:         {i:.0%}")
        print(f"  Outer Neighborhoods served:{o:.0%}")
        print(f"  parity gap: {res.parity_gap:.0%}")
        print()

    summary("── NAIVE (utility only) ──", naive)
    summary("── FAIR (demographic parity) ──", fair)
    return naive, fair


if __name__ == "__main__":
    run()
