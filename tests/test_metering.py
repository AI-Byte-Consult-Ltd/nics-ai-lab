"""Tests of token metering and credit pricing (pure arithmetic, no network).

  python tests/test_metering.py
"""
import sys
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.billing.metering import (  # noqa: E402
    TokenPrice, consumer_charge, credits_to_eur, idempotency_key, price_for, token_cost, usage_event,
)

DAY = date(2026, 12, 1)


def main() -> None:
    # A credit is the wallet unit, not a token price: 1 credit = 100,000 micro input tokens.
    assert token_cost("nics-text-micro", 100_000, 0, DAY) == Decimal("1")
    assert token_cost("nics-text-micro", 1_000_000, 0, DAY) == Decimal("10")      # 0.10 EUR per 1M in
    assert token_cost("nics-text-micro", 0, 1_000_000, DAY) == Decimal("40")      # 0.40 EUR per 1M out
    assert token_cost("nics-text-alpha", 1_000_000, 1_000_000, DAY) == Decimal("150")
    assert token_cost("nics-text-micro", 1_000_000_000, 0, DAY) == Decimal("10000")  # 1bn tokens = 100 EUR
    assert credits_to_eur(10000) == Decimal("100.0000")

    # The plan's check: one month of [01 CORE] volume on micro = 365 credits = 3.65 EUR.
    month = token_cost("nics-text-micro", 17_700_000, 4_700_000, DAY)
    assert month == Decimal("365") and credits_to_eur(month) == Decimal("3.6500")

    # End users: rounded up, at least one credit, nothing for a failed answer.
    small = token_cost("nics-text-micro", 1500, 300, DAY)
    assert small == Decimal("0.0270")
    assert consumer_charge(small, validated=True) == 1
    assert consumer_charge(Decimal("2.0001"), validated=True) == 3
    assert consumer_charge(Decimal("5"), validated=False) == 0

    # Prices are dated: a new row applies from its date, old charges stay reproducible.
    prices = (
        TokenPrice("m", Decimal("1"), Decimal("2"), date(2026, 1, 1)),
        TokenPrice("m", Decimal("3"), Decimal("4"), date(2026, 6, 1)),
    )
    assert price_for("m", date(2026, 5, 31), prices).credits_in_per_1k == Decimal("1")
    assert price_for("m", date(2026, 6, 1), prices).credits_in_per_1k == Decimal("3")
    for bad in (lambda: price_for("m", date(2025, 12, 31), prices), lambda: price_for("unknown", DAY)):
        try:
            bad()
            raise AssertionError("expected KeyError")
        except KeyError:
            pass
    try:
        token_cost("nics-text-micro", -1, 0, DAY)
        raise AssertionError("negative tokens accepted")
    except ValueError:
        pass

    # Journal rows: internal calls are metered but free; the same request has the same key.
    now = datetime(2026, 12, 1, 12, 0, tzinfo=timezone.utc)
    common = dict(product="trading.card", request_id="sig-1:ru", model="nics-text-micro", model_version="v1",
                  tokens_in=1500, tokens_out=300, duration_ms=900, now=now)
    internal = usage_event(validated=True, **common)
    assert internal.account_id is None and internal.charged_credits == 0 and internal.cost_credits == "0.0270"
    paid = usage_event(validated=True, account_id="u1", **common)
    failed = usage_event(validated=False, account_id="u1", **common)
    assert paid.charged_credits == 1 and failed.charged_credits == 0
    assert paid.idempotency_key == failed.idempotency_key == idempotency_key("trading.card", "sig-1:ru")
    assert idempotency_key("trading.card", "sig-2:ru") != paid.idempotency_key
    assert set(paid.as_row()) >= {"tokens_in", "tokens_out", "model_version", "validated", "charged_credits"}
    print("test_metering: OK")


if __name__ == "__main__":
    main()
