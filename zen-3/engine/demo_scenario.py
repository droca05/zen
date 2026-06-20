"""
demo_scenario.py
────────────────
Escenario de escasez para la demo del MILP (pantalla split-screen).

La tesis solo se hace visible bajo escasez. Con capacidad abundante todos
son atendidos y naive == fair. Aquí construimos deliberadamente una situación
donde un algoritmo greedy (naive) perjudica sistemáticamente a las familias
de la periferia, y la restricción de paridad demográfica lo corrige.

Contexto: Área Metropolitana de Monterrey, NL.
  - Zona Centro (zonas 2-3): acceso a transporte, cerca de los recursos principales.
  - Zona Periferia (zonas 0-1): colonias alejadas, sin auto, peor puntaje de match.

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
    Escenario realista para Monterrey AMM: dos bancos de alimentos con
    capacidad insuficiente para todos, y una correlación entre zona geográfica
    y desventaja de acceso — exactamente el fallo que el VI-SPDAT reproducía.

    Naive: llena primero los matches con mejor puntaje (Centro, con transporte).
           La periferia queda sistemáticamente sin atención.
    Fair:  la restricción de paridad fuerza al solver a usar la capacidad del
           banco periférico para las familias de esa zona. Gap ≤ 10 %.
    """
    users: list[UserProfile] = []
    uid = 0

    def add(needs, urgency, group, zone, transport, size=4, income=9500):
        nonlocal uid
        users.append(UserProfile(
            user_id=f"U{uid:03d}", needs=needs, urgency=urgency,
            household_size=size, monthly_income=income, race_group=group,
            language="Spanish", has_transport=transport, zip_zone=zone,
        ))
        uid += 1

    # 10 familias Zona Centro — zona 2, con transporte, mejor puntaje de match
    for _ in range(10):
        add(["food"], "this_week", "Centro", zone=2, transport=True, income=11000)

    # 10 familias Zona Periferia — zona 0, sin transporte, peor puntaje de match
    for _ in range(10):
        add(["food"], "this_week", "Periferia", zone=0, transport=False, income=7500)

    # Dos bancos de alimentos: uno central (mayor capacidad) y uno periférico.
    # El naive llena el central primero (beneficia al grupo Centro);
    # el fair redistribuye la capacidad periférica hacia las familias periféricas.
    resources = [
        Resource(resource_id="R000",
                 name="Banco de Alimentos DIF Monterrey",
                 service_type="food", zip_zone=2, capacity=10,
                 max_income=0, min_household_size=0,
                 hours="Lun–Vie 8–17 h", last_verified_days_ago=1),
        Resource(resource_id="R001",
                 name="Despensa Comunitaria Santa Catarina",
                 service_type="food", zip_zone=0, capacity=8,
                 max_income=0, min_household_size=0,
                 hours="Lun–Sáb 9–14 h", last_verified_days_ago=3),
    ]
    return users, resources


def run():
    users, resources = build_scarcity_scenario()
    print("ESCENARIO DE ESCASEZ — Monterrey AMM")
    print("  20 familias (10 Zona Centro con transporte, 10 Zona Periferia sin transporte)")
    print("  2 bancos de alimentos, capacidad total 18 para 20 familias → escasez forzada\n")

    naive = solve(users, resources, fairness=False, max_distance=2.0)
    fair  = solve(users, resources, fairness=True, parity_delta=0.10, max_distance=2.0)

    def summary(tag, res):
        c = res.served_by_group.get("Centro", 0)
        p = res.served_by_group.get("Periferia", 0)
        print(f"{tag}")
        print(f"  atendidos total: {res.users_served}/{res.total_users}")
        print(f"  Zona Centro atendida:    {c:.0%}")
        print(f"  Zona Periferia atendida: {p:.0%}")
        print(f"  brecha de paridad: {res.parity_gap:.0%}")
        print()

    summary("── NAIVE (solo utilidad) ──", naive)
    summary("── FAIR (paridad demográfica) ──", fair)
    return naive, fair


if __name__ == "__main__":
    run()
