"""
scraper.py
──────────
Collects community-resource listings (food banks, rent relief, clinics,
childcare, job centers) and writes them to data/resources.json in the
Open Referral HSDS v3.0 shape that the Zen engine consumes.

TWO MODES
  1) LIVE scrape  ── `python scraper.py --live --url <listing_url>`
        Real requests + BeautifulSoup. Run this locally where you HAVE
        internet. It fetches a public resource-directory page, parses the
        listing cards, and maps each to an HSDS record. Selectors are kept
        in SELECTORS so you can retarget a new source in one place.

  2) SEED        ── `python scraper.py --seed`   (default)
        Generates a realistic, geographically-distributed HSDS dataset so
        the app runs immediately with zero network. Replace with --live
        output once you point it at a real source.

NOTE ON ETHICS/ToS: only scrape sources whose terms allow it, prefer official
open-data endpoints (many 211 / city portals publish HSDS or CSV directly),
and cache results — don't hammer a live site on every request. In production,
Diego wires this to an HSDA-compliant API instead of scraping HTML.
"""

from __future__ import annotations
import argparse, json, os, random, sys, time

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
OUT = os.path.join(DATA_DIR, "resources.json")

SERVICE_TYPES = ["food", "housing", "healthcare", "childcare", "employment"]

# ── Selectors for the LIVE scraper (retarget here for a new source) ───────────
SELECTORS = {
    "card":     "div.resource-card",       # each listing
    "name":     "h3.resource-name",
    "address":  "span.address",
    "hours":    "span.hours",
    "category": "span.category",
    "phone":    "a.phone",
}


def scrape_live(url: str, max_items: int = 100) -> list[dict]:
    """
    Real scrape. Requires `requests` and `beautifulsoup4` and internet.
    Maps each listing card to an HSDS-style record.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        sys.exit("Install deps first:  pip install requests beautifulsoup4")

    headers = {"User-Agent": "ZenBenefitsNavigator/1.0 (+hackathon MVP; contact: team)"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    records = []
    for i, card in enumerate(soup.select(SELECTORS["card"])[:max_items]):
        def grab(sel):
            el = card.select_one(SELECTORS[sel])
            return el.get_text(strip=True) if el else ""

        name = grab("name")
        if not name:
            continue
        category = grab("category").lower()
        service = next((s for s in SERVICE_TYPES if s in category), "food")
        records.append(_hsds_record(
            rid=f"R{i:04d}", name=name, service=service,
            address=grab("address"), hours=grab("hours") or "Call for hours",
            phone=grab("phone"),
        ))
        time.sleep(0.3)   # be polite
    return records


def _hsds_record(rid, name, service, address, hours, phone,
                 zone=None, capacity=None, max_income=0, min_hh=0,
                 verified_days=0) -> dict:
    """One Open Referral HSDS-shaped service record."""
    return {
        "resource_id": rid,
        "name": name,
        "service_type": service,
        "address": address,
        "phone": phone,
        "hours": hours,
        # operational fields the MILP needs
        "zip_zone": zone if zone is not None else random.randint(0, 5),
        "capacity": capacity if capacity is not None else random.randint(6, 50),
        "max_income": max_income,
        "min_household_size": min_hh,
        "last_verified_days_ago": verified_days,
        # HSDS bookkeeping
        "hsds": {
            "schema": "openreferral-hsds-3.0",
            "status": "active",
        },
    }


# ── SEED generator (realistic, no network) ────────────────────────────────────
SEED_NAMES = {
    "food": ["Northside Food Bank", "Eastside Community Fridge", "St. Mary's Pantry",
             "Harvest Hope Center", "Downtown Meal Program", "Riverside Food Shelf"],
    "housing": ["City Rent Relief Program", "Bridge Housing Services",
                "Emergency Shelter Network", "Stable Homes Initiative"],
    "healthcare": ["Community Health Clinic", "Eastside Free Clinic",
                   "Wellness Access Center", "Neighborhood Care Clinic"],
    "childcare": ["Childcare Subsidy Office", "Bright Start Daycare Assistance",
                  "Family Support Childcare"],
    "employment": ["Job Placement Center", "WorkSource Career Hub",
                   "Skills & Training Office", "Reemployment Services"],
}
STREETS = ["Main St", "Oak Ave", "5th Ave", "Elm St", "Park Blvd", "Cedar Ln",
           "Washington St", "Lincoln Ave", "Market St", "Hill Rd"]


def scrape_seed() -> list[dict]:
    random.seed(11)
    records = []
    rid = 0
    for service, names in SEED_NAMES.items():
        for name in names:
            zone = random.randint(0, 5)
            records.append(_hsds_record(
                rid=f"R{rid:04d}", name=name, service=service,
                address=f"{random.randint(100, 9999)} {random.choice(STREETS)}",
                hours=random.choice(["Mon-Fri 9-5", "Daily 8-8", "Mon-Sat 10-4",
                                     "Apply online", "Tue/Thu 9-1"]),
                phone=f"(555) {random.randint(200,999)}-{random.randint(1000,9999)}",
                zone=zone,
                capacity=random.randint(6, 45),
                max_income=random.choice([0, 1500, 2000, 2500]),
                min_hh=random.choice([0, 0, 0, 2]),
                verified_days=random.randint(0, 70),
            ))
            rid += 1
    return records


def main():
    ap = argparse.ArgumentParser(description="Zen community-resource scraper")
    ap.add_argument("--live", action="store_true", help="scrape a live URL")
    ap.add_argument("--seed", action="store_true", help="generate seed data (default)")
    ap.add_argument("--url", default="", help="listing URL for --live")
    args = ap.parse_args()

    if args.live:
        if not args.url:
            sys.exit("--live needs --url <listing_url>")
        print(f"Scraping live: {args.url}")
        records = scrape_live(args.url)
    else:
        print("Generating seed dataset (HSDS v3.0 shape)…")
        records = scrape_seed()

    with open(OUT, "w") as f:
        json.dump({"source": "live" if args.live else "seed",
                   "schema": "openreferral-hsds-3.0",
                   "count": len(records),
                   "resources": records}, f, indent=2)

    by_type = {}
    for r in records:
        by_type[r["service_type"]] = by_type.get(r["service_type"], 0) + 1
    print(f"✓ wrote {len(records)} resources → {OUT}")
    print("  by service:", by_type)
    print("  total capacity:", sum(r["capacity"] for r in records), "slots")


if __name__ == "__main__":
    main()
