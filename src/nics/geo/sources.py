"""Image sources for NICS Vision Geo: metadata only, one manifest row per image.

A manifest row (JSON line) holds everything needed to download, label, attribute and
split the image, so the set can be rebuilt and audited without re-reading any API:

  image_id     'si:<media id>' / 'nhm:<asset id>' / 'wc:<file title>'  (unique)
  specimen_id  one physical specimen; all its photos stay in the same split
  source       dataset_id in data_registry/datasets.json
  raw_name     the name the museum or author gave
  class        taxonomy key, or null if the name maps to no class
  url          image to download (a size we can afford on the laptop)
  licence      short id from licence.normalize (rows without an allowed licence are dropped)
  attribution  who to credit; record_link where the specimen is described
  place, lat, lon  where it was found, if the source says so (used later for maps)

Sources:
  smithsonian  NMNH Department of Mineral Sciences, Smithsonian Open Access bulk metadata
               (public S3 bucket, no key), CC0 images only; downloaded at 384 px (IDS size
               suffix), full-size files are 10x larger and the server throttles them
  nhm          Natural History Museum London Data Portal, collection specimens, MIN
               collection with images, licence per image (CC BY 4.0 in practice); the
               "thumbnail" rendition (512 px) is enough for a 192 px cache
  commons      Wikimedia Commons categories from taxonomy.CLASSES[*].commons, licence per
               file from extmetadata. Polite: one request per second, maxlag. Meant for the
               laptop (the cloud egress was rate-limited by Commons on 2026-11-22, so this
               reader has unit tests on recorded responses but no live run yet).
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from typing import Callable, Iterable, Iterator

from . import licence, taxonomy

UA = "NICS-AI-Lab/0.17 (NICS Vision Geo research; https://aibyteconsult.com)"

SI_INDEX = "https://smithsonian-open-access.s3-us-west-2.amazonaws.com/metadata/edan/nmnhminsci/index.txt"
NHM_API = "https://data.nhm.ac.uk/api/3/action/datastore_search"
NHM_SPECIMENS = "05ff2255-c38a-40c9-b657-4ccb55ab2feb"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

Fetch = Callable[[str], bytes]


def http_get(url: str, timeout: float = 120.0, tries: int = 4) -> bytes:
    last: Exception | None = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # network hiccup or rate limit: back off and retry
            last = e
            code = getattr(e, "code", None)
            if code in (400, 401, 403, 404):
                break
            time.sleep(min(60, 2 ** (i + 1)))
    raise RuntimeError(f"GET failed: {url[:120]} ({last})")


def http_get_quick(url: str) -> bytes:
    """For image downloads: a dead link should cost seconds, not minutes."""
    return http_get(url, timeout=60.0, tries=2)


# ---------------------------------------------------------------- Smithsonian

def _si_small(url: str) -> str:
    """IDS delivery URLs take a size suffix; other Smithsonian media URLs are used as they are."""
    return url + "/384" if "ids.si.edu/ids/deliveryService/id/" in url else url


def smithsonian_rows_from_record(d: dict) -> list[dict]:
    c = d.get("content", {})
    dn = c.get("descriptiveNonRepeating", {})
    ix = c.get("indexedStructured", {})
    topics = ix.get("topic", []) or []
    name = d.get("title") or ""
    key = taxonomy.map_name(name, topics)
    lat = lon = None
    for g in ix.get("geoLocation", []) or []:
        p = (g.get("points") or {}).get("point") or {}
        try:
            lat, lon = float(p["latitude"]["content"]), float(p["longitude"]["content"])
        except (KeyError, TypeError, ValueError):
            pass
    rows = []
    for m in dn.get("online_media", {}).get("media", []) or []:
        if m.get("type") != "Images" or not m.get("content"):
            continue
        lic = licence.normalize((m.get("usage") or {}).get("access"))
        if not lic:
            continue
        mid = m.get("id") or m.get("idsId")
        rows.append({
            "image_id": f"si:{mid}", "specimen_id": f"si:{d.get('id')}", "source": "smithsonian-nmnh-minsci-v1",
            "raw_name": name, "topics": topics, "class": key, "url": _si_small(m["content"]), "licence": lic,
            "attribution": "Smithsonian National Museum of Natural History, Department of Mineral Sciences",
            "record_link": dn.get("record_link"), "place": ", ".join(ix.get("place", []) or []), "lat": lat, "lon": lon,
        })
    return rows


def smithsonian(fetch: Fetch = http_get, log: Callable[[str], None] = print) -> Iterator[dict]:
    parts = [u.strip() for u in fetch(SI_INDEX).decode().splitlines() if u.strip()]
    for i, url in enumerate(parts):
        for line in fetch(url).decode("utf-8").splitlines():
            if line.strip():
                yield from smithsonian_rows_from_record(json.loads(line))
        if (i + 1) % 32 == 0:
            log(f"smithsonian: {i + 1}/{len(parts)} metadata files")


# ---------------------------------------------------------------- NHM London

def nhm_rows_from_record(r: dict) -> list[dict]:
    name = r.get("scientificName") or ""
    key = taxonomy.map_name(name)
    rows = []
    for m in r.get("associatedMedia") or []:
        lic = licence.normalize(m.get("license"))
        if not lic or m.get("type") != "StillImage" or not m.get("identifier"):
            continue
        who = ", ".join(x for x in (m.get("creator"), m.get("rightsHolder")) if x)
        try:
            lat, lon = float(r.get("decimalLatitude")), float(r.get("decimalLongitude"))
        except (TypeError, ValueError):
            lat = lon = None
        rows.append({
            "image_id": f"nhm:{m.get('assetID') or m.get('_id')}", "specimen_id": f"nhm:{r.get('occurrenceID') or r.get('_id')}",
            "source": "nhm-london-mineralogy-v1", "raw_name": name, "topics": [], "class": key,
            "url": m["identifier"].rstrip("/") + "/thumbnail", "licence": lic,
            "attribution": who or "The Trustees of the Natural History Museum, London",
            "record_link": f"https://data.nhm.ac.uk/object/{r.get('occurrenceID')}" if r.get("occurrenceID") else None,
            "place": r.get("locality") or r.get("country") or "", "lat": lat, "lon": lon,
        })
    return rows


def nhm(fetch: Fetch = http_get, log: Callable[[str], None] = print, page: int = 1000) -> Iterator[dict]:
    after = None
    while True:
        q = {"resource_id": NHM_SPECIMENS, "filters": json.dumps({"collectionCode": "MIN", "_has_image": True}),
             "limit": page}
        if after is not None:
            q["after"] = json.dumps(after)
        res = json.loads(fetch(NHM_API + "?" + urllib.parse.urlencode(q)))["result"]
        for r in res.get("records", []):
            yield from nhm_rows_from_record(r)
        after = res.get("after")
        if not res.get("records") or not after:
            return
        log(f"nhm: page done, total {res.get('total')}")


# ---------------------------------------------------------------- Wikimedia Commons

def commons_rows_from_pages(pages: Iterable[dict], key: str, category: str) -> list[dict]:
    rows = []
    for p in pages:
        info = (p.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata") or {}
        lic = licence.normalize((meta.get("LicenseShortName") or {}).get("value")) or \
            licence.normalize((meta.get("LicenseUrl") or {}).get("value"))
        mime = info.get("mime") or ""
        url = info.get("thumburl") or info.get("url")
        if not lic or not url or not mime.startswith("image/") or mime == "image/svg+xml":
            continue
        artist = (meta.get("Artist") or {}).get("value") or ""
        artist = _strip_html(artist)[:200]
        title = p.get("title", "")
        rows.append({
            "image_id": f"wc:{title}", "specimen_id": f"wc:{title}", "source": "wikimedia-commons-geo-v1",
            "raw_name": f"Category:{category}", "topics": [], "class": key, "url": url, "licence": lic,
            "attribution": f"{artist or 'Wikimedia Commons contributor'}, {lic}, via Wikimedia Commons",
            "record_link": info.get("descriptionurl"), "place": "", "lat": None, "lon": None,
        })
    return rows


def _strip_html(s: str) -> str:
    import re
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s)).strip()


def commons(fetch: Fetch = http_get, log: Callable[[str], None] = print, per_category: int = 400,
            pause: float = 1.0, thumb: int = 384) -> Iterator[dict]:
    """Files directly in each class's Commons categories (no sub-categories: they drift off-topic)."""
    for c in taxonomy.CLASSES:
        for cat in c.commons:
            got, cont = 0, None
            while got < per_category:
                q = {"action": "query", "format": "json", "generator": "categorymembers", "gcmtitle": f"Category:{cat}",
                     "gcmtype": "file", "gcmlimit": 50, "prop": "imageinfo", "iiprop": "url|mime|extmetadata",
                     "iiurlwidth": thumb, "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist", "maxlag": 5}
                if cont:
                    q.update(cont)
                d = json.loads(fetch(COMMONS_API + "?" + urllib.parse.urlencode(q)))
                rows = commons_rows_from_pages((d.get("query") or {}).get("pages", {}).values(), c.key, cat)
                for r in rows[: per_category - got]:
                    yield r
                got += len(rows)
                cont = d.get("continue")
                time.sleep(pause)
                if not cont:
                    break
            log(f"commons: {cat} -> {c.key}: {got} files with an allowed licence")


SOURCES = {"smithsonian": smithsonian, "nhm": nhm, "commons": commons}
