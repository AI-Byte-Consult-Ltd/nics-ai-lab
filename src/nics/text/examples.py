"""Our own training examples for NICS Language tasks: "facts in, text out".

Task "journal" (first to replace a third-party model): a short channel post about a demo
trade that was just opened. Everything here is written by us; no output of any other
model is used.

Rules built into the format:
  * The model never writes a number, a ticker or the trader's note. It writes value
    slots (<|symbol|>, <|entry|>, <|stop|>, <|v0|>...) and code puts the real values in.
  * The post never says how the trade will end and never calls anyone to action.
  * The facts line holds only closed-vocabulary fields, so free text typed by a person
    never reaches the model as instructions: the note travels only as a slot value.

Record layout (one line of the token stream):
  <bos><|in|><|task:journal|><|lang:ru|> dir=buy targets=many conf=yes note=yes<|out|>TEXT<|end|><eos>
"""
from __future__ import annotations

import random
import re
from typing import Iterator

from nics.text.tokenizer import APP_LANGUAGES

# slot name in the phrase bank -> control token the model writes
SLOTS = {
    "symbol": "<|symbol|>",
    "entry": "<|entry|>",
    "stop": "<|stop|>",
    "targets": "<|v1|>",
    "conf": "<|v2|>",
    "note": "<|v0|>",
}
NL = "\n"

# Every bank: sentence position -> variants. {a|b} inside a variant is expanded per
# direction as buy|sell. Slots are written as {symbol}, {entry}, {stop}, {targets},
# {conf}, {note}.
BANK: dict[str, dict[str, list[str]]] = {
    "ru": {
        "title": [
            "<b>Демо-сделка: {symbol}</b>",
            "<b>{symbol}: открыл демо-сделку</b>",
            "<b>Дневник. {symbol}, {покупка|продажа}</b>",
            "<b>{symbol}, {лонг|шорт} на демо-счёте</b>",
            "<b>Новая запись в дневнике: {symbol}</b>",
        ],
        "open": [
            "Сегодня открыл демо-сделку по {symbol}, направление {на покупку|на продажу}.",
            "Открыл на демо-счёте {покупку|продажу} по {symbol}.",
            "По {symbol} вошёл {в покупку|в продажу}, сделка на демо-счёте.",
            "Взял {symbol} {в лонг|в шорт} на демо, по разбору нашей системы.",
            "Сделка по {symbol}: {покупка|продажа}, счёт демонстрационный.",
            "Решил пройти по разбору системы и открыл {покупку|продажу} по {symbol} на демо.",
        ],
        "entry": [
            "Зона входа {entry}.",
            "Вход брал в зоне {entry}.",
            "Входил в диапазоне {entry}.",
            "Цена входа лежит в зоне {entry}.",
            "Зашёл в пределах {entry}.",
        ],
        "stop": [
            "Стоп стоит на {stop}, это {ниже|выше} входа.",
            "Стоп поставил на {stop}.",
            "Защитный стоп на уровне {stop}.",
            "Если цена дойдёт до {stop}, сделка закроется по стопу.",
            "Стоп {под входом|над входом}, на {stop}.",
        ],
        "targets_one": [
            "Цель одна: {targets}.",
            "Тейк стоит на {targets}.",
            "Забирать планирую на {targets}.",
            "Цель по сделке {targets}.",
        ],
        "targets_many": [
            "Цели: {targets}.",
            "Тейки расставлены на {targets}.",
            "Забирать планирую частями: {targets}.",
            "Цели по сделке идут по порядку: {targets}.",
            "Фиксировать буду по уровням {targets}.",
        ],
        "conf": [
            "Система оценила этот вход в {conf} из 100.",
            "Оценка системы по этой сделке {conf} из 100.",
            "По шкале системы это {conf} из 100.",
            "Балл системы {conf} из 100, сам я его не менял.",
        ],
        "note": [
            "Из моих заметок: {note}",
            "Что записал себе по этой сделке: {note}",
            "Моя заметка к сделке: {note}",
            "Для себя отметил: {note}",
            "Заметка на полях: {note}",
        ],
        "close": [
            "Сделка пока открыта, чем закончится, запишу отдельно.",
            "Позиция ещё в работе, итог будет в следующей записи.",
            "Результата пока нет, сделка открыта.",
            "Пока просто наблюдаю, итог напишу, когда сделка закроется.",
            "Сделка идёт, выводов пока не делаю.",
        ],
    },
    "en": {
        "title": [
            "<b>Demo trade: {symbol}</b>",
            "<b>{symbol}: demo trade opened</b>",
            "<b>Journal. {symbol}, {buy|sell}</b>",
            "<b>{symbol}, {long|short} on the demo account</b>",
            "<b>New journal entry: {symbol}</b>",
        ],
        "open": [
            "Today I opened a demo trade on {symbol}, {on the buy side|on the sell side}.",
            "I opened a {buy|sell} on {symbol} on the demo account.",
            "On {symbol} I went {long|short}; this is a demo account trade.",
            "I took {symbol} {long|short} on demo, following our system's analysis.",
            "A trade on {symbol}: {a buy|a sell}, on the demonstration account.",
            "I decided to follow the system's analysis and opened a {buy|sell} on {symbol} on demo.",
        ],
        "entry": [
            "The entry zone is {entry}.",
            "I entered in the {entry} zone.",
            "The entry was taken in the range {entry}.",
            "The entry price sits in the {entry} zone.",
            "I got in within {entry}.",
        ],
        "stop": [
            "The stop is at {stop}, {below|above} the entry.",
            "I placed the stop at {stop}.",
            "The protective stop is at {stop}.",
            "If price reaches {stop}, the trade closes at the stop.",
            "The stop sits {under the entry|over the entry}, at {stop}.",
        ],
        "targets_one": [
            "There is one target: {targets}.",
            "The take profit is at {targets}.",
            "I plan to take profit at {targets}.",
            "The target for this trade is {targets}.",
        ],
        "targets_many": [
            "Targets: {targets}.",
            "Take profits are set at {targets}.",
            "I plan to take profit in parts: {targets}.",
            "The targets go in order: {targets}.",
            "I will be closing parts at {targets}.",
        ],
        "conf": [
            "The system scored this entry {conf} out of 100.",
            "The system's score for this trade is {conf} out of 100.",
            "On the system's scale this is {conf} out of 100.",
            "The system's score is {conf} out of 100, and I did not change it.",
        ],
        "note": [
            "From my notes: {note}",
            "What I wrote down about this trade: {note}",
            "My note on the trade: {note}",
            "A note to myself: {note}",
            "A note in the margin: {note}",
        ],
        "close": [
            "The trade is still open, and I will write up how it ends separately.",
            "The position is still running, the result will be in the next entry.",
            "There is no result yet, the trade is open.",
            "For now I am just watching, and I will post the outcome when the trade closes.",
            "The trade is running, and I am not drawing conclusions yet.",
        ],
    },
    "es": {
        "title": [
            "<b>Operación demo: {symbol}</b>",
            "<b>{symbol}: operación demo abierta</b>",
            "<b>Diario. {symbol}, {compra|venta}</b>",
            "<b>{symbol}, {largo|corto} en la cuenta demo</b>",
            "<b>Nueva entrada del diario: {symbol}</b>",
        ],
        "open": [
            "Hoy abrí una operación demo en {symbol}, {de compra|de venta}.",
            "Abrí una {compra|venta} en {symbol} en la cuenta demo.",
            "En {symbol} entré {en largo|en corto}; es una operación en cuenta demo.",
            "Tomé {symbol} {en largo|en corto} en demo, siguiendo el análisis de nuestro sistema.",
            "Operación en {symbol}: {compra|venta}, en la cuenta de demostración.",
            "Decidí seguir el análisis del sistema y abrí una {compra|venta} en {symbol} en demo.",
        ],
        "entry": [
            "La zona de entrada es {entry}.",
            "Entré en la zona {entry}.",
            "La entrada se tomó en el rango {entry}.",
            "El precio de entrada está en la zona {entry}.",
            "Entré dentro de {entry}.",
        ],
        "stop": [
            "El stop está en {stop}, {por debajo|por encima} de la entrada.",
            "Puse el stop en {stop}.",
            "El stop de protección está en {stop}.",
            "Si el precio llega a {stop}, la operación se cierra por stop.",
            "El stop queda {bajo la entrada|sobre la entrada}, en {stop}.",
        ],
        "targets_one": [
            "Hay un solo objetivo: {targets}.",
            "La toma de beneficios está en {targets}.",
            "Pienso recoger beneficios en {targets}.",
            "El objetivo de la operación es {targets}.",
        ],
        "targets_many": [
            "Objetivos: {targets}.",
            "Las tomas de beneficios están en {targets}.",
            "Pienso recoger beneficios por partes: {targets}.",
            "Los objetivos van en orden: {targets}.",
            "Iré cerrando partes en {targets}.",
        ],
        "conf": [
            "El sistema valoró esta entrada con {conf} sobre 100.",
            "La puntuación del sistema para esta operación es {conf} sobre 100.",
            "En la escala del sistema esto es {conf} sobre 100.",
            "La puntuación del sistema es {conf} sobre 100 y no la cambié.",
        ],
        "note": [
            "De mis notas: {note}",
            "Lo que apunté sobre esta operación: {note}",
            "Mi nota sobre la operación: {note}",
            "Una nota para mí: {note}",
            "Nota al margen: {note}",
        ],
        "close": [
            "La operación sigue abierta y contaré aparte cómo termina.",
            "La posición sigue en marcha, el resultado irá en la próxima entrada.",
            "Todavía no hay resultado, la operación está abierta.",
            "Por ahora solo observo y publicaré el resultado cuando la operación se cierre.",
            "La operación sigue en curso y aún no saco conclusiones.",
        ],
    },
    "bg": {
        "title": [
            "<b>Демо сделка: {symbol}</b>",
            "<b>{symbol}: отворена демо сделка</b>",
            "<b>Дневник. {symbol}, {покупка|продажба}</b>",
            "<b>{symbol}, {дълга позиция|къса позиция} на демо сметка</b>",
            "<b>Нов запис в дневника: {symbol}</b>",
        ],
        "open": [
            "Днес отворих демо сделка по {symbol}, {в посока покупка|в посока продажба}.",
            "Отворих {покупка|продажба} по {symbol} на демо сметката.",
            "По {symbol} влязох {в покупка|в продажба}, сделката е на демо сметка.",
            "Взех {symbol} {в дълга позиция|в къса позиция} на демо, по анализа на нашата система.",
            "Сделка по {symbol}: {покупка|продажба}, сметката е демонстрационна.",
            "Реших да следвам анализа на системата и отворих {покупка|продажба} по {symbol} на демо.",
        ],
        "entry": [
            "Зоната за вход е {entry}.",
            "Входът беше в зоната {entry}.",
            "Влязох в диапазона {entry}.",
            "Цената на входа е в зоната {entry}.",
            "Влязох в рамките на {entry}.",
        ],
        "stop": [
            "Стопът е на {stop}, {под входа|над входа}.",
            "Сложих стопа на {stop}.",
            "Защитният стоп е на ниво {stop}.",
            "Ако цената стигне до {stop}, сделката се затваря по стоп.",
            "Стопът стои {под входа|над входа}, на {stop}.",
        ],
        "targets_one": [
            "Целта е една: {targets}.",
            "Тейкът е на {targets}.",
            "Смятам да затворя на {targets}.",
            "Целта по сделката е {targets}.",
        ],
        "targets_many": [
            "Цели: {targets}.",
            "Тейковете са на {targets}.",
            "Смятам да затварям на части: {targets}.",
            "Целите вървят по ред: {targets}.",
            "Ще фиксирам по нивата {targets}.",
        ],
        "conf": [
            "Системата оцени този вход с {conf} от 100.",
            "Оценката на системата за тази сделка е {conf} от 100.",
            "По скалата на системата това е {conf} от 100.",
            "Оценката на системата е {conf} от 100 и не съм я променял.",
        ],
        "note": [
            "От моите бележки: {note}",
            "Какво си записах за тази сделка: {note}",
            "Моята бележка към сделката: {note}",
            "Отбелязах си: {note}",
            "Бележка в полето: {note}",
        ],
        "close": [
            "Сделката още е отворена, как ще завърши, ще напиша отделно.",
            "Позицията още работи, резултатът ще бъде в следващия запис.",
            "Засега няма резултат, сделката е отворена.",
            "Засега само наблюдавам, резултата ще напиша, когато сделката се затвори.",
            "Сделката върви и още не правя изводи.",
        ],
    },
    "pt": {
        "title": [
            "<b>Operação demo: {symbol}</b>",
            "<b>{symbol}: operação demo aberta</b>",
            "<b>Diário. {symbol}, {compra|venda}</b>",
            "<b>{symbol}, {comprado|vendido} na conta demo</b>",
            "<b>Nova entrada no diário: {symbol}</b>",
        ],
        "open": [
            "Hoje abri uma operação demo em {symbol}, {de compra|de venda}.",
            "Abri uma {compra|venda} em {symbol} na conta demo.",
            "Em {symbol} entrei {comprado|vendido}; é uma operação em conta demo.",
            "Entrei {comprado|vendido} em {symbol} na demo, seguindo a análise do nosso sistema.",
            "Operação em {symbol}: {compra|venda}, na conta de demonstração.",
            "Decidi seguir a análise do sistema e abri uma {compra|venda} em {symbol} na demo.",
        ],
        "entry": [
            "A zona de entrada é {entry}.",
            "Entrei na zona {entry}.",
            "A entrada foi feita no intervalo {entry}.",
            "O preço de entrada está na zona {entry}.",
            "Entrei dentro de {entry}.",
        ],
        "stop": [
            "O stop está em {stop}, {abaixo|acima} da entrada.",
            "Coloquei o stop em {stop}.",
            "O stop de proteção está em {stop}.",
            "Se o preço chegar a {stop}, a operação fecha no stop.",
            "O stop fica {abaixo da entrada|acima da entrada}, em {stop}.",
        ],
        "targets_one": [
            "Há um único alvo: {targets}.",
            "A realização de lucro está em {targets}.",
            "Pretendo realizar lucro em {targets}.",
            "O alvo da operação é {targets}.",
        ],
        "targets_many": [
            "Alvos: {targets}.",
            "As realizações de lucro estão em {targets}.",
            "Pretendo realizar lucro por partes: {targets}.",
            "Os alvos seguem em ordem: {targets}.",
            "Vou fechando partes em {targets}.",
        ],
        "conf": [
            "O sistema avaliou esta entrada em {conf} de 100.",
            "A pontuação do sistema para esta operação é {conf} de 100.",
            "Na escala do sistema isto é {conf} de 100.",
            "A pontuação do sistema é {conf} de 100 e eu não a alterei.",
        ],
        "note": [
            "Das minhas notas: {note}",
            "O que anotei sobre esta operação: {note}",
            "A minha nota sobre a operação: {note}",
            "Uma nota para mim: {note}",
            "Nota à margem: {note}",
        ],
        "close": [
            "A operação continua aberta e vou contar à parte como termina.",
            "A posição continua em andamento, o resultado virá na próxima entrada.",
            "Ainda não há resultado, a operação está aberta.",
            "Por enquanto só observo e publico o resultado quando a operação fechar.",
            "A operação segue em curso e ainda não tiro conclusões.",
        ],
    },
}

_DIR = re.compile(r"\{([^{}|]*)\|([^{}|]*)\}")
_SLOT = re.compile(r"\{(symbol|entry|stop|targets|conf|note)\}")
_SLOT_TOKEN = re.compile(r"<\|[a-z0-9:]+\|>")
_ALLOWED_TAGS = re.compile(r"</?[bi]>")


def facts_line(direction: str, targets: int, has_conf: bool, has_note: bool) -> str:
    """The closed-vocabulary description of a demo trade the model is conditioned on."""
    if direction not in ("buy", "sell"):
        raise ValueError(f"direction must be buy or sell, got {direction!r}")
    if targets < 1:
        raise ValueError("a trade needs at least one target")
    return (f" dir={direction} targets={'one' if targets == 1 else 'many'}"
            f" conf={'yes' if has_conf else 'no'} note={'yes' if has_note else 'no'}")


def prompt(lang: str, direction: str, targets: int, has_conf: bool, has_note: bool) -> str:
    if lang not in BANK:
        raise ValueError(f"no journal phrases for language {lang!r}")
    return f"<|in|><|task:journal|><|lang:{lang}|>{facts_line(direction, targets, has_conf, has_note)}<|out|>"


def _render(variant: str, direction: str) -> str:
    text = _DIR.sub(lambda m: m.group(1) if direction == "buy" else m.group(2), variant)
    return _SLOT.sub(lambda m: SLOTS[m.group(1)], text)


def compose(lang: str, direction: str, targets: int, has_conf: bool, has_note: bool, rng: random.Random) -> str:
    """One post with slots, composed from our phrase bank."""
    bank = BANK[lang]
    pick = lambda key: _render(rng.choice(bank[key]), direction)  # noqa: E731
    body = [pick("open"), pick("entry")]
    tail = [pick("stop"), pick("targets_one" if targets == 1 else "targets_many")]
    if rng.random() < 0.3:
        tail.reverse()  # sometimes the targets come before the stop
    body += tail
    if has_conf:
        body.append(pick("conf"))
    paragraphs = [pick("title"), " ".join(body)]
    if has_note:
        paragraphs.append(pick("note"))
    paragraphs.append(pick("close"))
    return (NL + NL).join(paragraphs)


def record(lang: str, direction: str, targets: int, has_conf: bool, has_note: bool, rng: random.Random) -> str:
    """One full training record (prompt + answer + end marker)."""
    return prompt(lang, direction, targets, has_conf, has_note) + compose(lang, direction, targets, has_conf, has_note, rng) + "<|end|>"


def stream(seed: int, languages: tuple[str, ...] = APP_LANGUAGES) -> Iterator[str]:
    """Endless stream of records, all fact combinations and languages equally likely."""
    rng = random.Random(seed)
    while True:
        yield record(rng.choice(languages), rng.choice(("buy", "sell")), rng.choice((1, 2, 3, 4)),
                     rng.random() < 0.7, rng.random() < 0.7, rng)


def required_slots(targets: int, has_conf: bool, has_note: bool) -> dict[str, int]:
    """How many times each slot must appear in a valid post (the symbol is also in the title)."""
    need = {"<|symbol|>": 2, "<|entry|>": 1, "<|stop|>": 1, "<|v1|>": 1}
    if has_conf:
        need["<|v2|>"] = 1
    if has_note:
        need["<|v0|>"] = 1
    return need


_WORD = re.compile(r"[^\W\d_]+")
_VOCABULARY: dict[str, frozenset[str]] = {}


def vocabulary(lang: str) -> frozenset[str]:
    """Every word our phrase bank uses in this language (both directions), lower case."""
    if lang not in _VOCABULARY:
        words: set[str] = set()
        for variants in BANK[lang].values():
            for variant in variants:
                for direction in ("buy", "sell"):
                    plain = _ALLOWED_TAGS.sub(" ", _SLOT_TOKEN.sub(" ", _render(variant, direction)))
                    words.update(w.lower() for w in _WORD.findall(plain))
        _VOCABULARY[lang] = frozenset(words)
    return _VOCABULARY[lang]


_DIRECTION_WORDS: dict[tuple[str, str], frozenset[str]] = {}


def direction_words(lang: str, direction: str) -> frozenset[str]:
    """Words our phrases use only for this direction (buy or sell) in this language."""
    key = (lang, direction)
    if key not in _DIRECTION_WORDS:
        sides: dict[str, set[str]] = {"buy": set(), "sell": set()}
        common: set[str] = set()  # words used outside the buy|sell choices say nothing about the side
        for variants in BANK[lang].values():
            for variant in variants:
                for match in _DIR.finditer(variant):
                    sides["buy"].update(w.lower() for w in _WORD.findall(match.group(1)))
                    sides["sell"].update(w.lower() for w in _WORD.findall(match.group(2)))
                rest = _ALLOWED_TAGS.sub(" ", _SLOT.sub(" ", _DIR.sub(" ", variant)))
                common.update(w.lower() for w in _WORD.findall(rest))
        _DIRECTION_WORDS[(lang, "buy")] = frozenset(sides["buy"] - sides["sell"] - common)
        _DIRECTION_WORDS[(lang, "sell")] = frozenset(sides["sell"] - sides["buy"] - common)
    return _DIRECTION_WORDS[key]


def check(text: str, targets: int, has_conf: bool, has_note: bool, lang: str | None = None,
          direction: str | None = None) -> list[str]:
    """Why a generated post must not be published; an empty list means it may.

    Checked by code, never by the model: exact slots, no digits (every number comes from
    a slot), only <b>/<i> tags, sane length and, when lang is given, a closed vocabulary:
    every word must be one our own phrases use in that language, so a post in another
    language, a mixed one or one with an invented word is refused. With direction given,
    a post that uses a word of the opposite side (a sale described for a buy) is refused
    too. A failed post is replaced by the template."""
    problems = []
    found: dict[str, int] = {}
    for token in _SLOT_TOKEN.findall(text):
        found[token] = found.get(token, 0) + 1
    need = required_slots(targets, has_conf, has_note)
    for token, count in need.items():
        if found.get(token, 0) != count:
            problems.append(f"slot {token}: expected {count}, got {found.get(token, 0)}")
    for token in found:
        if token not in need:
            problems.append(f"unexpected control token {token}")
    plain = _SLOT_TOKEN.sub("", text)
    if re.search(r"\d", plain.replace("100", "")):
        problems.append("a number was written by the model")
    if "<" in _ALLOWED_TAGS.sub("", plain) or ">" in _ALLOWED_TAGS.sub("", plain):
        problems.append("markup other than <b> and <i>")
    if plain.count("<b>") != plain.count("</b>") or plain.count("<i>") != plain.count("</i>"):
        problems.append("unbalanced tags")
    if lang is not None:
        known = vocabulary(lang)
        strange = sorted({w.lower() for w in _WORD.findall(_ALLOWED_TAGS.sub(" ", plain))} - known)
        if strange:
            problems.append(f"words outside the {lang} vocabulary: {', '.join(strange[:5])}")
        if direction is not None:
            used = {w.lower() for w in _WORD.findall(_ALLOWED_TAGS.sub(" ", plain))}
            wrong = sorted(used & direction_words(lang, "sell" if direction == "buy" else "buy"))
            if wrong:
                problems.append(f"words of the opposite direction: {', '.join(wrong[:5])}")
            if not used & direction_words(lang, direction):
                problems.append("the direction of the trade is not stated")
    words = len(plain.split())
    if not 15 <= words <= 140:
        problems.append(f"length {words} words")
    return problems


def escape_html(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fill(text: str, values: dict[str, str]) -> str:
    """Put real values into the slots. Values are escaped, so a note cannot add markup."""
    by_token = {SLOTS[name]: escape_html(str(value)) for name, value in values.items()}
    return _SLOT_TOKEN.sub(lambda m: by_token.get(m.group(0), ""), text)
