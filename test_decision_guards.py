"""
Unit tests for the decision-guard layer in nodes.py.

These tests call `apply_decision_guards` directly with hand-built
decision lists — no LLM call, no Mireye/EIA call, no network access.
The point is to prove that invalid or dangerous LLM output (multiple
SELECTs, a SELECT with a blocking issue, a hallucinated cost figure)
gets caught and repaired by deterministic code, not by hoping the
prompt was followed correctly.

Run with:
    python3 -m pytest test_decision_guards.py -v
or, without pytest installed:
    python3 test_decision_guards.py
"""

from nodes import apply_decision_guards


def make_decision(
    location,
    decision,
    reasons=None,
    blocking_issues=None,
    electricity_price_cents_per_kwh=None,
    assumed_load_mw=None,
    estimated_annual_electricity_cost_usd="SHOULD_BE_IGNORED",
):
    """
    Build a decision dict shaped like `SiteDecision.model_dump()`.

    `estimated_annual_electricity_cost_usd` defaults to a sentinel
    string on purpose: apply_decision_guards must overwrite it with a
    freshly computed number (or None), never trust whatever the LLM
    put there.
    """
    return {
        "location": location,
        "decision": decision,
        "reasons": reasons or [],
        "blocking_issues": blocking_issues or [],
        "electricity_price_cents_per_kwh": electricity_price_cents_per_kwh,
        "assumed_load_mw": assumed_load_mw,
        "estimated_annual_electricity_cost_usd": estimated_annual_electricity_cost_usd,
    }


# ---------------------------------------------------------------------
# 1. Normal case: exactly one valid SELECT
# ---------------------------------------------------------------------

def test_normal_selection():
    decisions = [
        make_decision("Houston", "SELECT", reasons=["Best power + water"]),
        make_decision("Dallas", "SHORTLIST", reasons=["Feasible but pricier"]),
        make_decision("Austin", "REJECT", blocking_issues=["Water insufficient"]),
    ]

    result, selected_location = apply_decision_guards(decisions, "Houston")

    assert selected_location == "Houston"
    by_location = {d["location"]: d["decision"] for d in result}
    assert by_location["Houston"] == "SELECT"
    assert by_location["Dallas"] == "SHORTLIST"
    assert by_location["Austin"] == "REJECT"


# ---------------------------------------------------------------------
# 2. SELECT with a blocking issue must be demoted to REJECT
# ---------------------------------------------------------------------

def test_blocking_issue_rejected():
    decisions = [
        make_decision(
            "Houston",
            "SELECT",
            reasons=["High score"],
            blocking_issues=["Flood zone AE — uninsurable without mitigation"],
        ),
        make_decision("Dallas", "SHORTLIST"),
    ]

    result, selected_location = apply_decision_guards(decisions, "Houston")

    houston = next(d for d in result if d["location"] == "Houston")
    assert houston["decision"] == "REJECT"
    # No other candidate was feasible enough to promote automatically.
    assert selected_location is None


# ---------------------------------------------------------------------
# 3. Multiple SELECTs: keep only the first, demote the rest
# ---------------------------------------------------------------------

def test_multiple_select_guard():
    decisions = [
        make_decision("Houston", "SELECT", reasons=["Strong"]),
        make_decision("Dallas", "SELECT", reasons=["Also strong"]),
        make_decision("Austin", "REJECT", blocking_issues=["Insufficient water"]),
    ]

    result, selected_location = apply_decision_guards(decisions, "Houston")

    select_count = sum(1 for d in result if d["decision"] == "SELECT")
    assert select_count == 1
    assert selected_location == "Houston"

    dallas = next(d for d in result if d["location"] == "Dallas")
    assert dallas["decision"] == "SHORTLIST"


# ---------------------------------------------------------------------
# 4. No feasible candidate: everything REJECT, selected_location=None
# ---------------------------------------------------------------------

def test_all_candidates_rejected():
    decisions = [
        make_decision("Houston", "REJECT", blocking_issues=["Flood risk"]),
        make_decision("Dallas", "REJECT", blocking_issues=["No water access"]),
        make_decision("Austin", "REJECT", blocking_issues=["Grid capacity exceeded"]),
    ]

    result, selected_location = apply_decision_guards(decisions, None)

    assert selected_location is None
    assert all(d["decision"] == "REJECT" for d in result)


# ---------------------------------------------------------------------
# 5. Electricity cost must be recomputed in code, never trusted from
#    the LLM. This is the most important test in this file: it proves
#    the $/year figure shown to the user is deterministic arithmetic,
#    not something the model could hallucinate.
# ---------------------------------------------------------------------

def test_electricity_cost_is_recomputed_not_trusted():
    decisions = [
        make_decision(
            "Houston",
            "SELECT",
            electricity_price_cents_per_kwh=5.5,   # $0.055/kWh
            assumed_load_mw=50,
            # LLM claims a wrong number on purpose — must be overwritten.
            estimated_annual_electricity_cost_usd=999_999_999,
        ),
    ]

    result, _ = apply_decision_guards(decisions, "Houston")

    houston = result[0]
    # 50 MW * 1000 (kW) * 8760 (hours/year) * 5.5 cents/kWh / 100
    expected = 50 * 1000 * 8760 * 5.5 / 100
    assert expected == 24_090_000.0
    assert houston["estimated_annual_electricity_cost_usd"] == 24_090_000.0
    assert houston["estimated_annual_electricity_cost_usd"] != 999_999_999


def test_electricity_cost_is_none_when_price_missing():
    decisions = [
        make_decision(
            "Somewhere",
            "SHORTLIST",
            electricity_price_cents_per_kwh=None,  # EIA lookup failed / unavailable
            assumed_load_mw=50,
        ),
    ]

    result, _ = apply_decision_guards(decisions, None)

    assert result[0]["estimated_annual_electricity_cost_usd"] is None


# ---------------------------------------------------------------------
# Minimal runner so this file works even without pytest installed.
# ---------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_normal_selection,
        test_blocking_issue_rejected,
        test_multiple_select_guard,
        test_all_candidates_rejected,
        test_electricity_cost_is_recomputed_not_trusted,
        test_electricity_cost_is_none_when_price_missing,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)
