#!/usr/bin/env python3
"""
spx_server.py - gives the SPHEREx Blink page pictures for ANY coordinates.

The page asks:   GET /field?ra=133.795&dec=-7.245        (degrees)
This server then
  1. asks NASA/IPAC IRSA which SPHEREx spectral images cover that spot (TAP query),
  2. downloads a small cutout of each one (in parallel),
  3. re-aligns them with spx_convert.py, and
  4. sends them back in the same format as spherex-data.json.
Results are saved in ./cache, so a repeat visit is instant, and when NASA releases
new data only the NEW images are downloaded and added.

Run it:        pip install numpy        then        python spx_server.py
It listens on  http://localhost:8000   (or the PORT environment variable).
Put the server's public address into SERVER near the top of index.html.

Optional query parameters: n (grid size in pixels, 20-100, default 90),
limit (max frames, default 60, spread evenly over time), det (e.g. 5,6 = only arrays D5, D6).
Also: GET /catalog?ra=..&dec=..&r=0.12 returns Gaia DR3 stars (epoch 2016.0 + proper motion) for the overlay.
      GET /resolve?name=Barnard%27s+Star returns {ra, dec} (moved to 2026) for the search box.
"""
import csv, io, json, math, os, re, sys, threading, time
import urllib.error, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import spx_convert as spx

TAP_URL = os.environ.get("SPX_TAP", "https://irsa.ipac.caltech.edu/TAP/sync")
IRSA_ROOT = os.environ.get("SPX_ROOT", "https://irsa.ipac.caltech.edu/")
CACHE_DIR = os.environ.get("SPX_CACHE", "cache")
FRESH_SECONDS = 6 * 3600       # re-check the archive for new data after this long
DOWNLOAD_WORKERS = 6           # be gentle with the archive
MAX_JOBS = 2                   # fields being built at the same time
UA = "spherex-blink/1.0 (public sky viewer)"
GAIA_TAP = os.environ.get("SPX_GAIA", "https://gea.esac.esa.int/tap-server/tap/sync")
VIZIER_TAP = os.environ.get("SPX_VIZIER", "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync")  # Gaia DR3 mirror at CDS
CAT_TIMEOUT = float(os.environ.get("SPX_CAT_TIMEOUT", "12"))  # seconds to wait for each Gaia service
cats = {}  # Gaia cone searches, kept for a day

jobs, lock = {}, threading.Lock()
slots = threading.BoundedSemaphore(MAX_JOBS)


def http_get(url, data=None, timeout=180, tries=3):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # network hiccup: wait and retry
            last = e
            if isinstance(e, urllib.error.HTTPError):  # say exactly what the archive answered
                body = e.read()[:300].decode("utf-8", "replace").replace("\n", " ")
                print("  HTTP %s from %s: %s" % (e.code, url.split("?")[0], body), file=sys.stderr)
                if e.code in (400, 401, 403, 404):  # retrying cannot fix these
                    break
            time.sleep(1.5 * (k + 1))
    raise last


def find_frames(ra, dec):
    """Ask IRSA which SPHEREx spectral images cover this position."""
    adql = (
        "SELECT a.uri AS uri, p.time_bounds_lower AS mjd, p.energy_bandpassname AS band "
        "FROM spherex.artifact a JOIN spherex.plane p ON a.planeid = p.planeid "
        "WHERE 1 = CONTAINS(POINT('ICRS', %.6f, %.6f), p.poly) ORDER BY p.time_bounds_lower" % (ra, dec)
    )
    q = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql})
    text = http_get(TAP_URL + "?" + q, timeout=300).decode("utf-8", "replace")
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        uri = (r.get("uri") or "").strip()
        if not uri.lower().endswith(".fits"):
            continue
        band = (r.get("band") or "")
        rows.append({"url": uri if uri.startswith("http") else IRSA_ROOT + uri.lstrip("/"),
                     "mjd": float(r["mjd"]) if r.get("mjd") else 0.0,
                     "det": int(band[-1]) if band[-1:].isdigit() else 0})
    return rows


def gaia_cone(ra, dec, r):
    """Gaia DR3 stars within r degrees, brightest first: [ra, dec, pmra, pmdec, G] at epoch 2016.0.
    Tries the ESA archive first, then the CDS VizieR copy. Each gets CAT_TIMEOUT seconds, so a slow
    service can never leave the page waiting for minutes."""
    key = "%.3f_%.3f_%.3f" % (ra, dec, r)
    hit = cats.get(key)
    if hit and time.time() - hit[0] < 86400:
        return hit[1]
    cone = "1 = CONTAINS(POINT('ICRS', %s, %s), CIRCLE('ICRS', %.6f, %.6f, %.5f))"
    services = [
        (GAIA_TAP, "SELECT TOP 400 ra, dec, pmra, pmdec, phot_g_mean_mag AS g FROM gaiadr3.gaia_source WHERE "
                   + cone % ("ra", "dec", ra, dec, r) + " AND phot_g_mean_mag < 20 ORDER BY phot_g_mean_mag"),
        (VIZIER_TAP, 'SELECT TOP 400 RA_ICRS, DE_ICRS, pmRA, pmDE, Gmag FROM "I/355/gaiadr3" WHERE '
                     + cone % ("RA_ICRS", "DE_ICRS", ra, dec, r) + " AND Gmag < 20 ORDER BY Gmag"),
    ]
    last = None
    for url, adql in services:
        try:
            q = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql})
            text = http_get(url + "?" + q, timeout=CAT_TIMEOUT, tries=1).decode("utf-8", "replace")
            out = []
            for row in csv.DictReader(io.StringIO(text)):
                d = {k.lower(): v for k, v in row.items() if k}  # the two services name columns differently
                try:  # stars without a proper-motion solution are treated as stationary
                    out.append([float(d.get("ra") or d["ra_icrs"]), float(d.get("dec") or d["de_icrs"]),
                                float(d.get("pmra") or 0), float(d.get("pmdec") or d.get("pmde") or 0),
                                float(d.get("g") or d["gmag"])])
                except (KeyError, ValueError):
                    pass
            cats[key] = (time.time(), out)
            return out
        except Exception as e:
            last = e
            print("  Gaia service %s failed: %s" % (url.split("/")[2], e), file=sys.stderr)
    raise last


SESAME = [b.strip() for b in os.environ.get("SPX_SESAME", "https://cds.unistra.fr/cgi-bin/nph-sesame/,https://vizier.idia.ac.za/viz-bin/nph-sesame/,https://vizier.cfa.harvard.edu/viz-bin/nph-sesame/").split(",")]
EPOCH = 2026.0  # moving stars are placed where they are now, roughly the middle of the SPHEREx data
names = {}


def resolve(name):
    """Object name -> ICRS position (degrees) moved to EPOCH, via CDS Sesame and its mirrors."""
    name = " ".join(name.replace("’", "'").split())
    if name.lower() in names:
        return names[name.lower()]
    m = text = last = None
    for base in SESAME:  # try the next mirror if one is down or knows nothing
        try:
            text = http_get(base + "-oI/SNV?" + urllib.parse.quote(name), timeout=20, tries=1).decode("utf-8", "replace")
        except Exception as e:
            last = e
            continue
        m = re.search(r"^%J\s+([\d.]+)\s+([+-]?[\d.]+)", text, re.M)
        if m:
            break
    if not m:
        if text is None:
            raise last
        return None
    ra, dec = float(m.group(1)), float(m.group(2))
    pm = re.search(r"^%P\s+([+-]?[\d.]+)\s+([+-]?[\d.]+)", text, re.M)
    if pm:  # proper motion in mas/yr: bring the J2000 position forward to EPOCH
        yr = EPOCH - 2000.0
        dec += float(pm.group(2)) * yr / 3.6e6
        ra += float(pm.group(1)) * yr / 3.6e6 / math.cos(math.radians(dec))
    names[name.lower()] = {"ra": ra % 360, "dec": dec}
    return names[name.lower()]


def spread(rows, limit):
    """Keep at most `limit` rows, spread evenly over time."""
    if len(rows) <= limit:
        return rows
    return [rows[int(i * (len(rows) - 1) / (limit - 1))] for i in range(limit)] if limit > 1 else rows[:1]


def cache_path(key):
    return os.path.join(CACHE_DIR, key + ".json")


def load_cache(key):
    try:
        return json.load(open(cache_path(key)))
    except Exception:
        return None


def payload(c, urls):
    frames = [f for f in c["frames"] if f["url"] in urls]
    return {"state": "done", "data": {"version": 1, "pixscale": spx.PIXSCALE, "fields": [
        {"ra": c["ra"], "dec": c["dec"], "n": c["n"], "frames": frames}]}}


def wanted(rows, dets, limit):
    if dets:
        rows = [r for r in rows if r["det"] in dets]
    return spread(rows, limit)


def build(key, ra, dec, n, dets, limit, job):
    try:
        with slots:
            job.update(msg="Searching the SPHEREx archive…")
            rows = wanted(find_frames(ra, dec), dets, limit)
            if not rows:
                job.update(state="error", msg="SPHEREx has no images covering that position yet.")
                return
            c = load_cache(key) or {"ra": ra, "dec": dec, "n": n, "frames": []}
            have = {f["url"] for f in c["frames"]}
            todo = [r for r in rows if r["url"] not in have]
            job.update(done=0, total=len(todo), msg="Downloading %d new images…" % len(todo))
            size = n * spx.PIXSCALE / 3600 * 1.2  # cutout width in degrees, with some margin

            def one(r):
                url = "%s?center=%.6f,%.6fd&size=%.4f" % (r["url"], ra, dec, size)
                try:
                    fr = spx.convert(http_get(url), ra, dec, n, name=os.path.basename(r["url"]))
                    fr["url"] = r["url"]
                    return fr
                except Exception as e:
                    print("  failed", r["url"], e, file=sys.stderr)
                    return None
                finally:
                    job["done"] += 1
                    job["msg"] = "Downloaded %d of %d images…" % (job["done"], job["total"])

            with ThreadPoolExecutor(DOWNLOAD_WORKERS) as pool:
                new = [f for f in pool.map(one, todo) if f]
            if todo and not new and not c["frames"]:
                job.update(state="error", msg="The archive did not return any image files.")
                return
            c["frames"] = sorted(c["frames"] + new, key=lambda f: f["mjd"])
            c["checked"] = time.time()
            os.makedirs(CACHE_DIR, exist_ok=True)
            json.dump(c, open(cache_path(key), "w"), separators=(",", ":"))
            job.update(state="done")
    except Exception as e:
        print("job failed:", e, file=sys.stderr)
        code = getattr(e, "code", None)
        job.update(state="error", msg="Couldn’t reach the SPHEREx archive (%s)." % ("HTTP %s" % code if code else type(e).__name__)
                   + (" It may be busy, so please try again in a few minutes." if code in (429, 500, 502, 503, 504) else ""))


def field_request(ra, dec, n, limit, dets):
    ra, dec = round(ra, 3), round(dec, 3)  # nearby requests share one cache entry
    key = "%.3f_%.3f_%d_%s_%d" % (ra, dec, n, "".join(map(str, sorted(dets))) or "all", limit)  # one cache entry per view
    c = load_cache(key)
    if c and time.time() - c.get("checked", 0) < FRESH_SECONDS:
        urls = {f["url"] for f in c["frames"] if not dets or f["det"] in dets}
        keep = {r["url"] for r in spread([{"url": f["url"], "mjd": f["mjd"]} for f in c["frames"] if f["url"] in urls], limit)}
        return payload(c, keep)
    with lock:
        job = jobs.get(key)
        if job and job["state"] == "done":
            c = load_cache(key)
            if c:
                urls = {f["url"] for f in c["frames"] if not dets or f["det"] in dets}
                keep = {r["url"] for r in spread([{"url": f["url"], "mjd": f["mjd"]} for f in c["frames"] if f["url"] in urls], limit)}
                jobs.pop(key, None)
                return payload(c, keep)
        if job and job["state"] == "error":
            jobs.pop(key, None)
            return {"state": "error", "msg": job["msg"]}
        if not job:
            job = jobs[key] = {"state": "running", "done": 0, "total": 0, "msg": "Starting…"}
            threading.Thread(target=build, args=(key, ra, dec, n, dets, limit, job), daemon=True).start()
        return {"state": "running", "done": job["done"], "total": job["total"], "msg": job["msg"]}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):  # Render's health check uses HEAD
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        if u.path in ("", "/"):
            return self._send(200, {"ok": True, "message": "SPHEREx Blink server is running. Try /health or /field?ra=133.795&dec=-7.245"})
        if u.path == "/favicon.ico":
            self.send_response(204)
            return self.end_headers()
        if u.path == "/health":
            return self._send(200, {"ok": True})
        if u.path == "/catalog":
            try:
                ra, dec = float(q["ra"]) % 360, float(q["dec"])
                assert -90 <= dec <= 90
                stars = gaia_cone(round(ra, 3), round(dec, 3), min(float(q.get("r", 0.12)), 0.3))
                return self._send(200, {"state": "done", "stars": stars})
            except Exception as e:
                return self._send(200, {"state": "error", "msg": "Gaia stars are unavailable right now (%s)." % type(e).__name__})
        if u.path == "/resolve":
            name = (q.get("name") or "").strip()[:80]
            if not name:
                return self._send(400, {"state": "error", "msg": "Type a name to search for."})
            try:
                res = resolve(name)
            except Exception as e:
                return self._send(200, {"state": "error", "msg": "The name service is unavailable right now (%s). Enter coordinates instead." % type(e).__name__})
            if not res:
                return self._send(200, {"state": "error", "msg": "No object called “%s” was found. Try another name or enter coordinates." % name})
            return self._send(200, dict(res, state="done"))
        if u.path != "/field":
            return self._send(404, {"state": "error", "msg": "Use /field?ra=...&dec=..."})
        try:
            ra, dec = float(q["ra"]) % 360, float(q["dec"])
            n, limit = int(q.get("n", 90)), int(q.get("limit", 60))
            dets = {int(x) for x in q["det"].split(",")} if q.get("det") else set()
            assert -90 <= dec <= 90 and 20 <= n <= 100 and 1 <= limit <= 120
        except Exception:
            return self._send(400, {"state": "error", "msg": "Please give ra and dec in degrees."})
        self._send(200, field_request(ra, dec, n, limit, dets))

    def log_message(self, fmt, *a):
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % a))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", sys.argv[1] if len(sys.argv) > 1 else 8000))
    print("SPHEREx Blink server on port %d  (cache folder: %s)" % (port, CACHE_DIR))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
