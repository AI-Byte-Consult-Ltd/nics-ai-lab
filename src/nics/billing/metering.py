"""Usage metering for NICS models: tokens are the meter, credits are the money.

  1 credit = 0.01 EUR. A credit is the WALLET unit, not the price of a token:
  one credit buys 100,000 input tokens of nics-text-micro.

Prices are per 1,000 tokens, separately for input and output, per model, with a
start date -- a price change is a new row, never an edit, so any past charge can
be recomputed. Tokens are counted with our own tokenizer
(nics.text.tokenizer.NicsTokenizer.count).

Status: PROPOSED prices, not approved for sale. Nothing here charges anyone; this
module only computes. The wallet and the ledger live in the production database
and are introduced separately.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal

CREDIT_EUR = Decimal("0.01")
_EXACT = Decimal("0.0001")  # API charges keep four decimals of a credit


@dataclass(frozen=True)
class TokenPrice:
    model: str
    credits_in_per_1k: Decimal
    credits_out_per_1k: Decimal
    valid_from: date


# Append only. The newest row whose valid_from is not in the future applies.
PRICE_LIST: tuple[TokenPrice, ...] = (
    TokenPrice("nics-text-micro", Decimal("0.01"), Decimal("0.04"), date(2026, 11, 22)),
    TokenPrice("nics-text-alpha", Decimal("0.03"), Decimal("0.12"), date(2026, 11, 22)),
)


def price_for(model: str, on: date | None = None, prices: tuple[TokenPrice, ...] = PRICE_LIST) -> TokenPrice:
    on = on or datetime.now(timezone.utc).date()
    rows = [p for p in prices if p.model == model and p.valid_from <= on]
    if not rows:
        raise KeyError(f"no price for model {model!r} on {on}")
    return max(rows, key=lambda p: p.valid_from)


def token_cost(model: str, tokens_in: int, tokens_out: int, on: date | None = None,
               prices: tuple[TokenPrice, ...] = PRICE_LIST) -> Decimal:
    """Exact cost in credits (four decimals). This is what an API customer pays."""
    if tokens_in < 0 or tokens_out < 0:
        raise ValueError("token counts cannot be negative")
    p = price_for(model, on, prices)
    cost = (Decimal(tokens_in) * p.credits_in_per_1k + Decimal(tokens_out) * p.credits_out_per_1k) / 1000
    return cost.quantize(_EXACT, rounding=ROUND_HALF_UP)


def consumer_charge(cost: Decimal, validated: bool) -> int:
    """Whole credits charged to an end user for one request.

    Rounded up, at least 1 credit. A request whose answer failed the format
    check (or never arrived) costs nothing.
    """
    if not validated:
        return 0
    return max(1, int(cost.to_integral_value(rounding=ROUND_CEILING)))


def credits_to_eur(credits: Decimal | int) -> Decimal:
    return (Decimal(credits) * CREDIT_EUR).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


def idempotency_key(product: str, request_id: str) -> str:
    """Stable key for one request: retrying the same request can never charge twice."""
    return hashlib.sha256(f"{product}:{request_id}".encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class UsageEvent:
    """One model call, as it is written to the usage journal."""
    idempotency_key: str
    account_id: str | None      # None = our own internal call (journal, scanner, digest)
    product: str                # e.g. "trading.card", "trading.journal", "public.digest", "language.chat"
    model: str
    model_version: str
    tokens_in: int
    tokens_out: int
    duration_ms: int
    validated: bool
    cost_credits: str           # exact, as text: never a float
    charged_credits: int
    created_at: str

    def as_row(self) -> dict:
        return asdict(self)


def usage_event(*, product: str, request_id: str, model: str, model_version: str, tokens_in: int,
                tokens_out: int, duration_ms: int, validated: bool, account_id: str | None = None,
                consumer: bool = True, now: datetime | None = None) -> UsageEvent:
    """Build the journal row for one call. Internal calls (no account) are metered at zero charge."""
    now = now or datetime.now(timezone.utc)
    cost = token_cost(model, tokens_in, tokens_out, now.date())
    if account_id is None:
        charged = 0
    elif consumer:
        charged = consumer_charge(cost, validated)
    else:
        charged = 0 if not validated else int(cost.to_integral_value(rounding=ROUND_CEILING))
    return UsageEvent(
        idempotency_key=idempotency_key(product, request_id), account_id=account_id, product=product,
        model=model, model_version=model_version, tokens_in=tokens_in, tokens_out=tokens_out,
        duration_ms=duration_ms, validated=validated, cost_credits=str(cost), charged_credits=charged,
        created_at=now.isoformat(),
    )
