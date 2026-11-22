"""Machine-readable rules registry: every rule NICS Trading has been given or will get.

The single list behind docs/rules_registry.md and the "rules" block on the CEO desk
(pushed after every nightly cycle). Neutral names only: no author, school, course or
company names (tests/test_rules_shadow.py checks this list too).

status: "training" = the model learns from it (methods.py);
        "feature"  = the model sees its state on every bar as an input (trend_context.py);
        "shadow"   = measured nightly on its own, the model does not learn from it yet (rules.py);
        "queued"   = planned, not coded yet;
        "rejected" = tested and dropped (reason in docs/rules_registry.md).
"""
from __future__ import annotations

REGISTRY = [
    # in training (src/nics/market/methods.py)
    {"id": "method_range_both_sides", "name": "Торговля в диапазоне, обе стороны",
     "status": "training", "added": "2026-11-19", "version": "0.5.0"},
    {"id": "method_breakout20", "name": "Пробой 20-свечного канала в тренде",
     "status": "training", "added": "2026-11-19", "version": "0.5.0"},
    {"id": "method_ema_pullback", "name": "Откат к EMA20 в тренде",
     "status": "training", "added": "2026-11-19", "version": "0.5.0"},
    {"id": "method_level_bounce", "name": "Отбой от уровня (два касания)",
     "status": "training", "added": "2026-11-19", "version": "0.5.0"},
    # shadow (src/nics/market/rules.py)
    {"id": "rule_strong_trend_pullback", "name": "Откат в сильном тренде (ADX > 30), вход стоп-ордером",
     "status": "shadow", "added": "2026-11-21", "version": "0.9.0"},
    {"id": "rule_level_squeeze_breakout", "name": "Уровневая методика: поджатие к уровню и пробой, 1:3",
     "status": "shadow", "added": "2026-11-21", "version": "0.9.1"},
    # daily trend context, model inputs since 0.14.0 (src/nics/market/trend_context.py);
    # research 22.11.2026, docs/methods_research_2026-11-22.md
    {"id": "trend_momentum_12m", "name": "Импульс 12 месяцев (знак доходности за 252 дня), D1",
     "status": "feature", "added": "2026-11-22", "version": "0.14.0"},
    {"id": "trend_ma_50_200", "name": "Тренд по средним: EMA50 выше/ниже EMA200, D1",
     "status": "feature", "added": "2026-11-22", "version": "0.14.0"},
    {"id": "trend_channel_55_20", "name": "Канальная система: вход на пробое 55 дней, выход по 20 дням, D1",
     "status": "feature", "added": "2026-11-22", "version": "0.14.0"},
    {"id": "trend_cloud", "name": "Тренд по облаку (линии 9/26/52), D1, пограничный результат",
     "status": "feature", "added": "2026-11-22", "version": "0.14.0"},
    {"id": "trend_macd", "name": "Тренд по MACD (линия, сигнальная и ноль согласны), D1, пограничный результат",
     "status": "feature", "added": "2026-11-22", "version": "0.14.0"},
    # tested 22.11.2026 on 1990-2026 daily history, not better than zero out of sample
    {"id": "rejected_rsi2_pullback", "name": "Короткий откат по RSI(2) в направлении SMA200",
     "status": "rejected", "added": "2026-11-22", "version": "0.14.0"},
    {"id": "rejected_band_reversion", "name": "Возврат к средней от границы полос (20, 2 сигмы)",
     "status": "rejected", "added": "2026-11-22", "version": "0.14.0"},
    {"id": "rejected_nr7_breakout", "name": "Пробой самой узкой свечи из 7 (стоп-ордера в обе стороны)",
     "status": "rejected", "added": "2026-11-22", "version": "0.14.0"},
    {"id": "rejected_atr_trailing_trend", "name": "Трендовая линия по ATR (10, 3) с переворотом",
     "status": "rejected", "added": "2026-11-22", "version": "0.14.0"},
    {"id": "rejected_volatility_squeeze", "name": "Сжатие полос волатильности и выход из него",
     "status": "rejected", "added": "2026-11-22", "version": "0.14.0"},
    # queue, one at a time through shadow
    {"id": "queued_multi_timeframe_screen", "name": "Три экрана: тренд на 4h, откат осциллятора на 1h, стоп-вход",
     "status": "queued", "added": "2026-11-21", "version": "-"},
    {"id": "queued_level_false_breakout", "name": "Уровневая методика: ложный пробой уровня, возврат внутрь",
     "status": "queued", "added": "2026-11-21", "version": "-"},
    {"id": "queued_second_entry", "name": "Второй вход в тренде (вторая попытка отката)",
     "status": "queued", "added": "2026-11-21", "version": "-"},
    {"id": "queued_accumulation_phases", "name": "Фазы накопления и распределения: ложный выход из боковика и тест",
     "status": "queued", "added": "2026-11-21", "version": "-"},
    {"id": "queued_trend_highs_lows", "name": "Структура тренда: последовательность максимумов и минимумов, смена тренда",
     "status": "queued", "added": "2026-11-21", "version": "-"},
    {"id": "queued_classic_patterns", "name": "Классические фигуры: двойная вершина/дно, флаг, треугольник",
     "status": "queued", "added": "2026-11-21", "version": "-"},
    {"id": "queued_range_edges_price_action", "name": "Свечной анализ у границ диапазона: отказ, пин-бар, поглощение",
     "status": "queued", "added": "2026-11-21", "version": "-"},
    {"id": "queued_pullback_quality", "name": "Качество отката: глубина и скорость отката против импульса",
     "status": "queued", "added": "2026-11-21", "version": "-"},
    {"id": "queued_adaptive_trend_system", "name": "Адаптивная трендовая система (адаптивная средняя, фильтр шума)",
     "status": "queued", "added": "2026-11-21", "version": "-"},
    {"id": "queued_volume_price", "name": "Объём и цена: подтверждение движения объёмом, кульминация",
     "status": "queued", "added": "2026-11-21", "version": "-"},
    {"id": "queued_r_multiple_sizing", "name": "Риск в R: размер позиции от стопа, оценка системы в R",
     "status": "queued", "added": "2026-11-21", "version": "-"},
]

STATUS_RU = {"training": "в обучении", "feature": "признак в обучении", "shadow": "в тени", "queued": "в очереди",
             "rejected": "отклонено"}


def summary() -> dict:
    """What the CEO desk shows: one line per rule."""
    return {"rules": [{**r, "status_ru": STATUS_RU[r["status"]]} for r in REGISTRY]}
