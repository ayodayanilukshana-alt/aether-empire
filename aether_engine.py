import os
import re
import sys
import json
import math
import time
import wave
import base64
import random
import asyncio
import subprocess
import urllib.parse
from pathlib import Path

import numpy as np
import requests
import edge_tts
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright
from xgboost import XGBRegressor
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

sys.stdout.reconfigure(line_buffering=True)


def mk_url(proto, domain_parts, path=""):
    return f"{proto}://{'.'.join(domain_parts)}{path}"


def clean_secret_url(u):
    s = str(u).strip()
    if "](" in s:
        s = s.split("](")[-1].rstrip(")")
    return s.replace("[", "").replace("]", "")


# =====================================================================
# 1. ISOLATED POD CONFIGURATION & MODEL ROUTER
# =====================================================================
POD_NAME = os.environ.get("POD_NAME", "WEALTH").strip().upper()
VIDEO_MODE = os.environ.get("VIDEO_MODE", "short").strip().lower()

DEFAULT_AI_URL = mk_url("https", ["api", "relaymodels", "com"], "/v1")
RAW_AI_URL = clean_secret_url(os.environ.get("AI_BASE_URL", DEFAULT_AI_URL)).rstrip("/")
AI_API_KEY = os.environ.get("AI_API_KEY", "").strip()

POD_CREDS = {
    "WEALTH": {
        "client_id": os.environ.get("WEALTH_CLIENT_ID", "").strip(),
        "client_secret": os.environ.get("WEALTH_CLIENT_SECRET", "").strip(),
        "refresh_token": os.environ.get("WEALTH_REFRESH_TOKEN", "").strip(),
        "voice": "en-US-ChristopherNeural",
        "subreddits": ["personalfinance", "investing", "dataisbeautiful", "fluentInFinance"],
        "autocomplete_seeds": [
            "what happens if you invested 1000 in",
            "gold vs real estate vs stocks",
            "compound interest simulation",
            "how inflation destroys cash",
            "renting vs buying a house math",
            "dividend stocks vs savings account",
            "warren buffett portfolio vs",
            "if you saved 10 dollars a day"
        ],
        "yt_search_query": "investing wealth comparison simulation explained"
    },
    "PHYSICS": {
        "client_id": os.environ.get("PHYSICS_CLIENT_ID", "").strip(),
        "client_secret": os.environ.get("PHYSICS_CLIENT_SECRET", "").strip(),
        "refresh_token": os.environ.get("PHYSICS_REFRESH_TOKEN", "").strip(),
        "voice": None,
        "subreddits": ["oddlysatisfying", "Simulated", "physicsgifs"],
        "autocomplete_seeds": [
            "satisfying ball escape physics",
            "balls vs spinning rings",
            "plinko ball race simulation",
            "bouncing ball grows on every hit",
            "hexagon cage ball escape",
            "satisfying polyrhythm bounce"
        ],
        "yt_search_query": "ball escape satisfying physics rings shorts"
    },
    "SCALE3D": {
        "client_id": os.environ.get("SCALE3D_CLIENT_ID", "").strip(),
        "client_secret": os.environ.get("SCALE3D_CLIENT_SECRET", "").strip(),
        "refresh_token": os.environ.get("SCALE3D_REFRESH_TOKEN", "").strip(),
        "voice": "en-US-EricNeural",
        "subreddits": ["megalophobia", "space", "interestingasfuck", "engineeringporn"],
        "autocomplete_seeds": [
            "3d size comparison animation",
            "planets and stars size comparison 3d",
            "fastest rockets and jets speed comparison",
            "deepest ocean trenches 3d comparison",
            "famous wonders and megastructures size comparison",
            "tallest mountains in the solar system"
        ],
        "yt_search_query": "3d size comparison animation metaballstudios style"
    }
}

ACTIVE_POD = POD_CREDS.get(POD_NAME, POD_CREDS["WEALTH"])

SCRIPT_MODELS = [
    "deepseek-v4-flash",
    "qwen3.7-plus",
    "qwen3.7-max",
    "deepseek-v4.1-flash",
    "gpt-6-luna"
]

FPS = 24
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
}
WIKI_HEADERS = {
    "User-Agent": "AetherMediaBot/6.0 (https://github.com/aether-empire; contact@aether-empire.org) requests/2.31"
}


def strip_stage_prefix(text):
    s = str(text).strip()
    s = re.sub(r"^(stage|step|part|rank|scene)\s*(one|two|three|four|five|six|\d+)\s*[:\-\.,]?\s*", "", s, flags=re.IGNORECASE)
    return s.strip()


def clean_text(text):
    if not text:
        return ""
    s = strip_stage_prefix(text)
    s = s.replace("—", " ").replace("–", " ").replace("-", " ").replace("\n", " ")
    s = re.sub(r"[^\x20-\x7E]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def clean_display_text(text):
    if not text:
        return ""
    s = str(text).replace("—", "-").replace("–", "-")
    s = re.sub(r"[^\x20-\x7E]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def call_ai(models, system_prompt, user_prompt, temperature=0.78):
    headers = {"Authorization": f"Bearer {AI_API_KEY}", "Content-Type": "application/json"}
    endpoint = RAW_AI_URL + "/chat/completions"
    for model in models:
        try:
            print(f"[{POD_NAME} AI Router] Calling {model}...", flush=True)
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature
            }
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=110)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"].strip()
                if content:
                    print(f"[{POD_NAME} AI Router] Success with {model}!", flush=True)
                    return content
        except Exception as e:
            print(f"[{POD_NAME} AI Router] {model} note: {e}", flush=True)
    raise RuntimeError("All AI models failed.")


def extract_json(text):
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1:
        text = text[s:e + 1]
    return json.loads(text)


def fetch_wikipedia_image_b64(query_name):
    """Downloads real photograph from Wikipedia and encodes to Base64 Data URI."""
    if not query_name:
        return ""
    q_clean = str(query_name).strip()
    try:
        slug = urllib.parse.quote(q_clean.replace(" ", "_"))
        rest_u = mk_url("https", ["en", "wikipedia", "org"], f"/api/rest_v1/page/summary/{slug}")
        r0 = requests.get(rest_u, headers=WIKI_HEADERS, timeout=6)
        thumb_url = ""
        if r0.status_code == 200:
            thumb_url = r0.json().get("thumbnail", {}).get("source", "")

        if not thumb_url:
            search_u = mk_url("https", ["en", "wikipedia", "org"], "/w/api.php")
            params = {
                "action": "query",
                "format": "json",
                "prop": "pageimages",
                "piprop": "thumbnail",
                "pithumbsize": 500,
                "generator": "search",
                "gsrsearch": q_clean,
                "gsrlimit": 1
            }
            r = requests.get(search_u, params=params, headers=WIKI_HEADERS, timeout=6)
            if r.status_code == 200:
                pages = r.json().get("query", {}).get("pages", {})
                for _, pdata in pages.items():
                    thumb_url = pdata.get("thumbnail", {}).get("source", "")
                    if thumb_url:
                        break

        if thumb_url:
            img_r = requests.get(thumb_url, headers=WIKI_HEADERS, timeout=8)
            if img_r.status_code == 200 and len(img_r.content) > 800:
                b64 = base64.b64encode(img_r.content).decode("utf-8")
                mime = "image/png" if thumb_url.lower().endswith(".png") else "image/jpeg"
                print(f"✅ [Wikimedia Photo] Downloaded real photo for '{q_clean}'!", flush=True)
                return f"data:{mime};base64,{b64}"
    except Exception as e:
        print(f"[Wikimedia Note] {q_clean}: {e}", flush=True)
    return ""


def get_youtube_client():
    creds = Credentials(
        token=None,
        refresh_token=ACTIVE_POD["refresh_token"],
        token_uri=mk_url("https", ["oauth2", "googleapis", "com"], "/token"),
        client_id=ACTIVE_POD["client_id"],
        client_secret=ACTIVE_POD["client_secret"],
        scopes=[
            mk_url("https", ["www", "googleapis", "com"], "/auth/youtube.upload"),
            mk_url("https", ["www", "googleapis", "com"], "/auth/youtube"),
            mk_url("https", ["www", "googleapis", "com"], "/auth/youtube.force-ssl")
        ]
    )
    return build("youtube", "v3", credentials=creds)


def load_pod_memory():
    p = Path(f"history_{POD_NAME.lower()}.json")
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"seed_idx": 0, "topics": [], "playlist_id": "", "published_videos": [], "xgb_feedback_rows": []}


def save_pod_memory(mem):
    Path(f"history_{POD_NAME.lower()}.json").write_text(json.dumps(mem, indent=2), encoding="utf-8")


# =====================================================================
# 2. AGENT 1 & 4: MULTI-SOURCE HARVESTER + 18-FEATURE XGBOOST
# =====================================================================
def compute_18_xgboost_features(title, views=50000, subs=10000, likes=2000, comments=150, hours_age=48.0, source_weight=1.0):
    t_low = str(title).lower()
    words = t_low.split()
    char_len = float(len(title))
    word_cnt = float(len(words))

    view_velocity = float(views) / max(float(hours_age), 1.0)
    outlier_ratio = min(float(views) / max(float(subs), 500.0), 500.0)
    engagement_rate = min((float(likes) + float(comments) * 4.0) / max(float(views), 100.0), 0.35)

    optimal_len_score = 1.0 - min(abs(char_len - 52.0) / 52.0, 1.0)
    has_number = 1.0 if re.search(r"\d+", t_low) else 0.0
    has_dollar_or_pct = 1.0 if re.search(r"[\$%]|million|billion", t_low) else 0.0
    has_question = 1.0 if "?" in title else 0.0
    has_vs_comparison = 1.0 if (" vs " in t_low or " compared" in t_low) else 0.0

    curiosity_words = ["secret", "never", "impossible", "shocking", "truth", "actually", "happens", "survive", "escape"]
    curiosity_score = sum(1.0 for w in curiosity_words if w in t_low)
    urgency_words = ["now", "before", "warning", "crash", "final", "until", "wait"]
    urgency_score = sum(1.0 for w in urgency_words if w in t_low)
    scale_words = ["100", "1000", "deepest", "largest", "smallest", "fastest", "universe", "extreme"]
    scale_score = sum(1.0 for w in scale_words if w in t_low)
    sim_words = ["simulation", "simulated", "3d", "physics", "test", "experiment"]
    sim_score = sum(1.0 for w in sim_words if w in t_low)
    avg_word_len = sum(len(w) for w in words) / max(word_cnt, 1.0)
    caps_ratio = sum(1.0 for c in str(title) if c.isupper()) / max(char_len, 1.0)

    like_to_view = float(likes) / max(float(views), 100.0)
    comment_to_view = float(comments) / max(float(views), 100.0)
    competition_inv = 1.0 / (1.0 + math.log10(max(float(subs), 100.0)))
    src_authority = float(source_weight)

    return [
        view_velocity, outlier_ratio, engagement_rate, optimal_len_score,
        has_number, has_dollar_or_pct, has_question, has_vs_comparison,
        curiosity_score, urgency_score, scale_score, sim_score,
        avg_word_len, caps_ratio, like_to_view, comment_to_view,
        competition_inv, src_authority
    ]


def run_global_harvester_and_18_feature_xgboost():
    mem = load_pod_memory()
    s_idx = mem.get("seed_idx", 0)
    mem["seed_idx"] = s_idx + 1

    candidates = []
    X_train, y_train = [], []

    yt = None
    try:
        yt = get_youtube_client()
        pub_vids = mem.get("published_videos", [])[-15:]
        my_ids = [v["video_id"] for v in pub_vids if "video_id" in v]
        if my_ids:
            my_resp = yt.videos().list(id=",".join(my_ids), part="snippet,statistics").execute()
            for mv in my_resp.get("items", []):
                snip, stat = mv.get("snippet", {}), mv.get("statistics", {})
                v_cnt = int(stat.get("viewCount", 0))
                l_cnt = int(stat.get("likeCount", 0))
                c_cnt = int(stat.get("commentCount", 0))
                f_vec = compute_18_xgboost_features(snip.get("title", ""), views=v_cnt, subs=100, likes=l_cnt, comments=c_cnt, hours_age=24.0, source_weight=2.0)
                score = math.log10(max(v_cnt * 10 + l_cnt * 50 + 10, 10))
                X_train.append(f_vec)
                y_train.append(score)
    except Exception:
        pass

    ac_base = mk_url("http", ["suggestqueries", "google", "com"], "/complete/search?client=youtube&ds=yt&q=")
    for pref in ACTIVE_POD["autocomplete_seeds"]:
        try:
            r = requests.get(ac_base + urllib.parse.quote(pref), headers=BROWSER_HEADERS, timeout=5)
            if r.status_code == 200:
                for m in re.findall(r'\["([^"]+)"', r.text):
                    if len(m) > 10:
                        candidates.append({"title": m, "source": "youtube_autocomplete", "weight": 1.35})
        except Exception:
            pass

    reddit_base = mk_url("https", ["www", "reddit", "com"], "/r/")
    for sub in ACTIVE_POD["subreddits"]:
        try:
            r = requests.get(f"{reddit_base}{sub}/hot.json?limit=15", headers=WIKI_HEADERS, timeout=6)
            if r.status_code == 200:
                for post in r.json().get("data", {}).get("children", []):
                    pdata = post.get("data", {})
                    ptitle = pdata.get("title", "")
                    ups = int(pdata.get("ups", 500))
                    comms = int(pdata.get("num_comments", 50))
                    if len(ptitle) > 12:
                        candidates.append({"title": ptitle, "source": f"reddit_r_{sub}", "weight": 1.25})
                        X_train.append(compute_18_xgboost_features(ptitle, views=ups * 20, subs=50000, likes=ups, comments=comms, hours_age=12.0, source_weight=1.25))
                        y_train.append(math.log10(max(ups * 20, 10)) * 1.4)
        except Exception:
            pass

    outliers = []
    hijacked_tags = []
    if yt is not None:
        try:
            s_resp = yt.search().list(q=ACTIVE_POD["yt_search_query"], part="id,snippet", type="video", order="viewCount", maxResults=12).execute()
            v_ids = [it["id"]["videoId"] for it in s_resp.get("items", []) if "videoId" in it.get("id", {})]
            if v_ids:
                st_resp = yt.videos().list(id=",".join(v_ids), part="snippet,statistics").execute()
                ch_ids = list({v["snippet"]["channelId"] for v in st_resp.get("items", []) if "channelId" in v.get("snippet", {})})
                ch_subs = {}
                if ch_ids:
                    ch_resp = yt.channels().list(id=",".join(ch_ids), part="statistics").execute()
                    for c in ch_resp.get("items", []):
                        ch_subs[c["id"]] = max(int(c.get("statistics", {}).get("subscriberCount", 1000)), 500)

                for v in st_resp.get("items", []):
                    snip, stat = v.get("snippet", {}), v.get("statistics", {})
                    title = snip.get("title", "")
                    views = int(stat.get("viewCount", 0))
                    likes = int(stat.get("likeCount", 0))
                    comments = int(stat.get("commentCount", 0))
                    subs = ch_subs.get(snip.get("channelId", ""), 10000)
                    outlier_mult = round(views / max(subs, 500), 2)
                    hijacked_tags.extend(snip.get("tags", [])[:5])

                    f_vec = compute_18_xgboost_features(title, views, subs, likes, comments, hours_age=72.0, source_weight=1.5)
                    v_score = math.log10(max(views, 10)) * (1.0 + min(outlier_mult / 8.0, 6.0))
                    X_train.append(f_vec)
                    y_train.append(v_score)
                    candidates.append({"title": title, "source": "youtube_outlier", "weight": 1.5})
                    outliers.append({"title": title, "views": views, "outlier_multiplier": f"{outlier_mult}x", "xgb_score": round(v_score, 2)})
        except Exception:
            pass

    while len(X_train) < 12:
        X_train.append(compute_18_xgboost_features("What happens if you simulate $1,000 vs 100 balls in 3D?", 250000, 5000, 12000, 800))
        y_train.append(22.5)

    xgb_model = XGBRegressor(n_estimators=40, max_depth=4, learning_rate=0.08, verbosity=0)
    xgb_model.fit(np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.float32))

    scored = []
    seen = set()
    for c in candidates:
        t_str = c["title"].strip()
        if t_str.lower() in seen:
            continue
        seen.add(t_str.lower())
        fv = np.array([compute_18_xgboost_features(t_str, views=180000, subs=6000, likes=8500, comments=420, hours_age=24.0, source_weight=c["weight"])], dtype=np.float32)
        pred = float(xgb_model.predict(fv)[0])
        scored.append({"title": t_str, "source": c["source"], "xgb_score": round(pred, 3)})

    scored.sort(key=lambda x: x["xgb_score"], reverse=True)
    cutoff = max(5, int(len(scored) * 0.08))
    elite_92_percentile = scored[:cutoff]

    save_pod_memory(mem)
    clean_tags = list(dict.fromkeys([re.sub(r"[^\w\s]", "", t).lower() for t in hijacked_tags if 2 < len(t) < 28]))[:15]
    return {
        "elite_candidates": elite_92_percentile[:12],
        "outliers": outliers[:6],
        "hijacked_tags": clean_tags,
        "run_idx": s_idx
    }


# =====================================================================
# 3. BESPOKE ENGINE 1: SCALEVERSE 3D V6.0 (COLLISION-FREE SPACING + TRUE 3D MODELS + REAL WIKI PHOTOS!)
# =====================================================================
THREEJS_SCALE3D_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #000; overflow: hidden; font-family: 'DejaVu Sans', 'Liberation Sans', sans-serif; }
  #webgl-container { position: absolute; inset: 0; }
  #hud-canvas { position: absolute; inset: 0; pointer-events: none; }
</style>
<script src="__THREE_JS_SRC__"></script>
</head>
<body>
<div id="webgl-container"></div>
<canvas id="hud-canvas"></canvas>
<script>
const DATA = __PAYLOAD_JSON__;
const W = DATA.width;
const H = DATA.height;
const isWide = DATA.is_widescreen;

const hudCanvas = document.getElementById('hud-canvas');
hudCanvas.width = W;
hudCanvas.height = H;
const ctx = hudCanvas.getContext('2d');

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b132b);
scene.fog = new THREE.FogExp2(0x0b132b, 0.00035);

const camera = new THREE.PerspectiveCamera(42, W / H, 0.5, 90000);
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setSize(W, H);
renderer.shadowMap.enabled = true;
document.getElementById('webgl-container').appendChild(renderer.domElement);

const hemiLight = new THREE.HemisphereLight(0xffffff, 0x60a5fa, 1.35);
hemiLight.position.set(0, 1200, 0);
scene.add(hemiLight);

const dirLight = new THREE.DirectionalLight(0xfffbeb, 1.6);
dirLight.position.set(350, 800, 700);
scene.add(dirLight);

const rimLight = new THREE.DirectionalLight(0x38bdf8, 0.9);
rimLight.position.set(-350, 400, -400);
scene.add(rimLight);

function createRealisticTexture(mode, baseHex, accentHex) {
    const c = document.createElement('canvas');
    c.width = 512; c.height = 512;
    const g = c.getContext('2d');
    g.fillStyle = baseHex;
    g.fillRect(0, 0, 512, 512);

    if (mode === 'v2_checkered') {
        g.fillStyle = '#f8fafc'; g.fillRect(0, 0, 512, 512);
        g.fillStyle = '#0f172a';
        for (let y = 0; y < 512; y += 128) {
            for (let x = 0; x < 512; x += 128) {
                if ((x + y) % 256 === 0) g.fillRect(x, y, 128, 128);
            }
        }
    } else if (mode === 'saturn_stages') {
        g.fillStyle = '#f8fafc'; g.fillRect(0, 0, 512, 512);
        g.fillStyle = '#0f172a';
        g.fillRect(0, 80, 512, 55);
        g.fillRect(0, 260, 512, 65);
        g.fillStyle = '#dc2626';
        g.fillRect(0, 430, 512, 30);
    } else if (mode === 'stainless_steel') {
        const gr = g.createLinearGradient(0, 0, 512, 0);
        gr.addColorStop(0, '#64748b'); gr.addColorStop(0.5, '#f1f5f9'); gr.addColorStop(1, '#475569');
        g.fillStyle = gr; g.fillRect(0, 0, 512, 512);
        g.strokeStyle = 'rgba(15,23,42,0.25)'; g.lineWidth = 2;
        for (let y = 0; y < 512; y += 32) { g.beginPath(); g.moveTo(0, y); g.lineTo(512, y); g.stroke(); }
    } else if (mode === 'lunar_craters') {
        g.fillStyle = '#94a3b8'; g.fillRect(0, 0, 512, 512);
        for (let i = 0; i < 90; i++) {
            const cx = (Math.sin(i * 12.3) * 0.5 + 0.5) * 480 + 16;
            const cy = (Math.cos(i * 45.7) * 0.5 + 0.5) * 480 + 16;
            g.fillStyle = (i % 2 === 0) ? '#64748b' : '#cbd5e1';
            g.beginPath(); g.arc(cx, cy, 6 + (i % 22), 0, Math.PI * 2); g.fill();
        }
    } else if (mode === 'earth_globe') {
        g.fillStyle = '#0284c7'; g.fillRect(0, 0, 512, 512);
        g.fillStyle = '#16a34a';
        for (let i = 0; i < 28; i++) {
            const cx = (Math.sin(i * 91.1) * 0.5 + 0.5) * 460 + 25;
            const cy = 80 + (Math.cos(i * 37.4) * 0.5 + 0.5) * 350;
            g.beginPath(); g.ellipse(cx, cy, 35 + (i % 45), 22 + (i % 30), i * 0.4, 0, Math.PI * 2); g.fill();
        }
        g.fillStyle = '#f8fafc'; g.fillRect(0, 0, 512, 36); g.fillRect(0, 476, 512, 36);
    } else if (mode === 'solar_plasma') {
        const grad = g.createLinearGradient(0, 0, 512, 512);
        grad.addColorStop(0, '#fef08a'); grad.addColorStop(0.5, '#f97316'); grad.addColorStop(1, '#dc2626');
        g.fillStyle = grad; g.fillRect(0, 0, 512, 512);
    } else if (mode === 'gas_giant') {
        for (let y = 0; y < 512; y += 4) {
            const w = Math.sin(y * 0.045) + 0.5 * Math.cos(y * 0.11);
            g.fillStyle = w > 0.2 ? accentHex : (w > -0.3 ? baseHex : '#78350f');
            g.fillRect(0, y, 512, 4);
        }
    } else if (mode === 'marble_arches') {
        g.fillStyle = '#fffbeb'; g.fillRect(0, 0, 512, 512);
        g.fillStyle = '#334155'; g.fillRect(156, 140, 200, 372);
        g.beginPath(); g.arc(256, 140, 100, Math.PI, 0); g.fill();
        g.strokeStyle = '#d97706'; g.lineWidth = 8; g.strokeRect(140, 110, 232, 390);
    } else if (mode === 'limestone_blocks') {
        g.fillStyle = '#d97706'; g.fillRect(0, 0, 512, 512);
        for (let y = 0; y < 512; y += 24) {
            const off = (y % 48 === 0) ? 0 : 24;
            for (let x = -24; x < 512; x += 48) {
                g.fillStyle = ((x + y) % 96 === 0) ? '#f59e0b' : '#b45309';
                g.fillRect(x + off, y, 46, 22);
            }
        }
    } else {
        g.fillStyle = '#0f172a'; g.fillRect(0, 0, 512, 512);
        for (let y = 8; y < 504; y += 18) {
            for (let x = 8; x < 504; x += 22) {
                g.fillStyle = (Math.sin(x * 19.3 + y * 71.7) > -0.25) ? accentHex : '#1e293b';
                g.fillRect(x, y, 15, 12);
            }
        }
    }
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = THREE.RepeatWrapping; tex.wrapT = THREE.RepeatWrapping;
    return tex;
}

// Reflective Studio Floor + Vertical 3D Scale Measurement Backdrop Wall!
const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(90000, 90000),
    new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.3, metalness: 0.4 })
);
ground.rotation.x = -Math.PI / 2;
scene.add(ground);

const gridHelper = new THREE.GridHelper(16000, 320, 0x38bdf8, 0x1e293b);
gridHelper.position.y = 0.2;
scene.add(gridHelper);

// Vertical Measurement Grid Wall behind the objects so camera pans NEVER look empty and scale is unmistakable!
const backGrid = new THREE.GridHelper(4000, 100, 0x38bdf8, 0x1e3a8a);
backGrid.rotation.x = Math.PI / 2;
backGrid.position.set(400, 400, -140);
scene.add(backGrid);

const items = DATA.items || [];
const rawMeters = items.map((it, idx) => Math.max(Number(it.meters) || (idx + 1) * 20, 0.1));
const minM = Math.min(...rawMeters);
const maxM = Math.max(...rawMeters);

// True visual height progression so Rank #1 looks visibly smaller beside Rank #2, #3, #4, #5!
const visualHeights = rawMeters.map((m, idx) => {
    if (maxM / minM > 25) {
        const ratio = Math.log10(m / minM + 1) / Math.log10(maxM / minM + 1);
        return 26 + ratio * 230 + idx * 18;
    }
    return Math.max(26, (m / maxM) * 260);
});

const objectWidths = items.map((it, idx) => {
    const h = visualHeights[idx];
    const nl = String((it && it.name) || '').toLowerCase();
    const mt = String((it && it.model_type) || '').toLowerCase();
    if (nl.includes('saturn') && mt === 'planet') return h * 1.65;
    if (mt === 'planet' || mt === 'star' || mt === 'dome_palace' || mt === 'pyramid' || mt === 'mountain') return h * 1.05;
    return h * 0.42;
});

// Tight Side-by-Side Architectural Lineup Spacing so previous smaller objects stay IN FRAME on the left!
const xPositions = [0];
for (let i = 1; i < 5; i++) {
    const stepX = (objectWidths[i - 1] * 0.58) + (objectWidths[i] * 0.58) + Math.max(visualHeights[i] * 0.24, 28);
    xPositions.push(xPositions[i - 1] + stepX);
}

const palettes = [
    ['#065f46', '#6ee7b7', 0x10b981],
    ['#1e3a8a', '#93c5fd', 0x3b82f6],
    ['#78350f', '#fde047', 0xf59e0b],
    ['#831843', '#f9a8d4', 0xec4899],
    ['#311068', '#d8b4fe', 0xa855f7]
];

const loadedImages = [];
window.waitForAllImages = async function() {
    const promises = items.map((it, idx) => new Promise(resolve => {
        if (it && it.image_b64) {
            const img = new Image();
            img.onload = () => { loadedImages[idx] = img; resolve(); };
            img.onerror = () => { loadedImages[idx] = null; resolve(); };
            img.src = it.image_b64;
        } else {
            loadedImages[idx] = null;
            resolve();
        }
    }));
    await Promise.all(promises);
};

function buildDistinct3DObject(itemObj, i, h, baseHex, glowHex, hexNum) {
    const group = new THREE.Group();
    const mType = String(itemObj.model_type || 'skyscraper').toLowerCase();
    const nameLow = String(itemObj.name || '').toLowerCase();

    if (mType === 'rocket' || mType === 'vehicle') {
        // 5 Distinct Rocket/Spacecraft Architectures so they NEVER look like clones!
        const matWhite = new THREE.MeshStandardMaterial({ color: 0xf8fafc, roughness: 0.22, metalness: 0.4 });
        const matDark = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.3, metalness: 0.6 });
        const matOrange = new THREE.MeshStandardMaterial({ color: 0xea580c, roughness: 0.35, metalness: 0.3 });
        const matSteel = new THREE.MeshStandardMaterial({ map: createRealisticTexture('stainless_steel', '#cbd5e1', '#64748b'), roughness: 0.15, metalness: 0.88 });

        if (nameLow.includes('v-2') || nameLow.includes('v2') || i === 0) {
            // V-2 Ballistic Missile: Checkered black/white tapered hull + 4 cruciform tail fins (NO side boosters!)
            const matV2 = new THREE.MeshStandardMaterial({ map: createRealisticTexture('v2_checkered', '#ffffff', '#000000'), roughness: 0.3 });
            const body = new THREE.Mesh(new THREE.CylinderGeometry(h * 0.08, h * 0.10, h * 0.68, 32), matV2);
            body.position.y = h * 0.38; group.add(body);
            const nose = new THREE.Mesh(new THREE.ConeGeometry(h * 0.08, h * 0.28, 32), matDark);
            nose.position.y = h * 0.86; group.add(nose);
            for (let f = 0; f < 4; f++) {
                const fin = new THREE.Mesh(new THREE.BoxGeometry(h * 0.26, h * 0.18, h * 0.02), matDark);
                fin.position.y = h * 0.11; fin.rotation.y = (f * Math.PI) / 2; group.add(fin);
            }
        } else if (nameLow.includes('falcon') || i === 1) {
            // SpaceX Falcon 9: Slender white core + black interstage + 4 black landing legs + payload fairing!
            const rad = h * 0.055;
            const stage1 = new THREE.Mesh(new THREE.CylinderGeometry(rad, rad, h * 0.62, 32), matWhite);
            stage1.position.y = h * 0.33; group.add(stage1);
            const inter = new THREE.Mesh(new THREE.CylinderGeometry(rad * 1.02, rad * 1.02, h * 0.08, 32), matDark);
            inter.position.y = h * 0.68; group.add(inter);
            const stage2 = new THREE.Mesh(new THREE.CylinderGeometry(rad, rad, h * 0.16, 32), matWhite);
            stage2.position.y = h * 0.80; group.add(stage2);
            const fairing = new THREE.Mesh(new THREE.ConeGeometry(rad * 1.25, h * 0.14, 32), matWhite);
            fairing.position.y = h * 0.94; group.add(fairing);
        } else if (nameLow.includes('sls') || nameLow.includes('launch system') || nameLow.includes('shuttle') || i === 2) {
            // NASA SLS: Rust-Orange Cryogenic Core Stage + 2 White Solid Rocket Boosters (SRBs)!
            const rad = h * 0.075;
            const core = new THREE.Mesh(new THREE.CylinderGeometry(rad, rad, h * 0.72, 32), matOrange);
            core.position.y = h * 0.38; group.add(core);
            const upper = new THREE.Mesh(new THREE.ConeGeometry(rad, h * 0.26, 32), matWhite);
            upper.position.y = h * 0.87; group.add(upper);
            [-1, 1].forEach(dir => {
                const srb = new THREE.Mesh(new THREE.CylinderGeometry(rad * 0.48, rad * 0.52, h * 0.54, 24), matWhite);
                srb.position.set(dir * rad * 1.55, h * 0.28, 0); group.add(srb);
                const srbCone = new THREE.Mesh(new THREE.ConeGeometry(rad * 0.48, h * 0.10, 24), matWhite);
                srbCone.position.set(dir * rad * 1.55, h * 0.60, 0); group.add(srbCone);
            });
        } else if (nameLow.includes('saturn') || i === 3) {
            // Apollo Saturn V: Stepped 3-Stage White/Black Heavy Lift Vehicle + Red Launch Escape Tower!
            const matSat = new THREE.MeshStandardMaterial({ map: createRealisticTexture('saturn_stages', '#ffffff', '#000000'), roughness: 0.28 });
            const s1 = new THREE.Mesh(new THREE.CylinderGeometry(h * 0.085, h * 0.095, h * 0.45, 32), matSat);
            s1.position.y = h * 0.225; group.add(s1);
            const s2 = new THREE.Mesh(new THREE.CylinderGeometry(h * 0.072, h * 0.085, h * 0.26, 32), matSat);
            s2.position.y = h * 0.58; group.add(s2);
            const s3 = new THREE.Mesh(new THREE.CylinderGeometry(h * 0.052, h * 0.072, h * 0.16, 32), matWhite);
            s3.position.y = h * 0.79; group.add(s3);
            const les = new THREE.Mesh(new THREE.CylinderGeometry(h * 0.012, h * 0.035, h * 0.13, 16), new THREE.MeshBasicMaterial({ color: 0xef4444 }));
            les.position.y = h * 0.935; group.add(les);
        } else {
            // SpaceX Starship & Super Heavy: Gleaming Stainless Steel Cylinder + Black Heatshield Nose + Aerodynamic Flaps!
            const rad = h * 0.082;
            const booster = new THREE.Mesh(new THREE.CylinderGeometry(rad, rad, h * 0.56, 36), matSteel);
            booster.position.y = h * 0.28; group.add(booster);
            const ship = new THREE.Mesh(new THREE.CylinderGeometry(rad * 0.98, rad, h * 0.28, 36), matSteel);
            ship.position.y = h * 0.70; group.add(ship);
            const nose = new THREE.Mesh(new THREE.ConeGeometry(rad * 0.98, h * 0.16, 36), matSteel);
            nose.position.y = h * 0.92; group.add(nose);
            const flap = new THREE.Mesh(new THREE.BoxGeometry(rad * 2.5, h * 0.12, h * 0.015), matDark);
            flap.position.y = h * 0.64; group.add(flap);
        }
    } else if (mType === 'planet' || mType === 'star') {
        const r = h * 0.46;
        let texMode = 'gas_giant', c1 = baseHex, c2 = glowHex;
        const isStar = (mType === 'star' || nameLow.includes('sun') || nameLow.includes('star') || nameLow.includes('scuti') || nameLow.includes('betelgeuse'));
        if (isStar) texMode = 'solar_plasma';
        else if (nameLow.includes('earth')) texMode = 'earth_globe';
        else if (nameLow.includes('moon') || nameLow.includes('mercury') || nameLow.includes('pluto')) texMode = 'lunar_craters';
        else if (nameLow.includes('mars')) { texMode = 'lunar_craters'; c1 = '#ea580c'; c2 = '#9a3412'; }
        else if (nameLow.includes('saturn') || nameLow.includes('jupiter')) { texMode = 'gas_giant'; c1 = '#d97706'; c2 = '#fef08a'; }

        const sphere = new THREE.Mesh(new THREE.SphereGeometry(r, 48, 48), new THREE.MeshStandardMaterial({
            map: createRealisticTexture(texMode, c1, c2),
            roughness: isStar ? 0.1 : 0.45,
            emissive: isStar ? 0xf97316 : 0x000000,
            emissiveIntensity: isStar ? 0.55 : 0.0
        }));
        sphere.position.y = r + 3; group.add(sphere);

        if (nameLow.includes('saturn') || nameLow.includes('uranus')) {
            const ring = new THREE.Mesh(new THREE.RingGeometry(r * 1.22, r * 1.72, 64), new THREE.MeshBasicMaterial({ color: 0xfde047, side: THREE.DoubleSide, transparent: true, opacity: 0.72 }));
            ring.rotation.x = Math.PI / 2.25; ring.position.y = r + 3; group.add(ring);
        }
    } else if (mType === 'dome_palace' || nameLow.includes('taj') || nameLow.includes('cathedral') || nameLow.includes('capitol')) {
        const matMarble = new THREE.MeshStandardMaterial({ map: createRealisticTexture('marble_arches', '#fffbeb', '#d97706'), roughness: 0.22 });
        const matWhite = new THREE.MeshStandardMaterial({ color: 0xfffbeb, roughness: 0.18 });
        const plinth = new THREE.Mesh(new THREE.BoxGeometry(h * 0.95, h * 0.12, h * 0.95), matWhite);
        plinth.position.y = h * 0.06; group.add(plinth);
        const body = new THREE.Mesh(new THREE.BoxGeometry(h * 0.62, h * 0.46, h * 0.62), matMarble);
        body.position.y = h * 0.35; group.add(body);
        const dome = new THREE.Mesh(new THREE.SphereGeometry(h * 0.21, 36, 36), matWhite);
        dome.position.y = h * 0.75; dome.scale.set(1, 1.28, 1); group.add(dome);
        const mOff = h * 0.42;
        [[-mOff, -mOff], [mOff, -mOff], [-mOff, mOff], [mOff, mOff]].forEach(([mx, mz]) => {
            const minT = new THREE.Mesh(new THREE.CylinderGeometry(h * 0.032, h * 0.048, h * 0.72, 24), matWhite);
            minT.position.set(mx, h * 0.42, mz); group.add(minT);
        });
    } else if (mType === 'pyramid' || nameLow.includes('pyramid')) {
        const matStone = new THREE.MeshStandardMaterial({ map: createRealisticTexture('limestone_blocks', '#d97706', '#f59e0b'), roughness: 0.75 });
        for (let s = 0; s < 14; s++) {
            const sh = (h * 0.88) / 14;
            const sw = h * 0.95 * (1 - s / 15.2);
            const tier = new THREE.Mesh(new THREE.BoxGeometry(sw, sh, sw), matStone);
            tier.position.y = s * sh + sh * 0.5; group.add(tier);
        }
    } else if (mType === 'mountain' || mType === 'trench' || mType === 'crater') {
        const matRock = new THREE.MeshStandardMaterial({ color: hexNum, roughness: 0.85 });
        const mainPeak = new THREE.Mesh(new THREE.ConeGeometry(h * 0.52, h * 0.85, 6), matRock);
        mainPeak.position.y = h * 0.425; group.add(mainPeak);
        const snowCap = new THREE.Mesh(new THREE.ConeGeometry(h * 0.22, h * 0.34, 6), new THREE.MeshStandardMaterial({ color: 0xf8fafc }));
        snowCap.position.y = h * 0.71; group.add(snowCap);
    } else {
        const tex = createRealisticTexture('windows', baseHex, glowHex);
        const matWin = new THREE.MeshStandardMaterial({ map: tex, roughness: 0.2, metalness: 0.65 });
        const tiers = 5 + i;
        for (let s = 0; s < tiers; s++) {
            const sh = (h * 0.86) / tiers;
            const sw = h * 0.28 * (1 - s * (0.68 / tiers));
            const geo = (i % 2 === 1) ? new THREE.CylinderGeometry(sw * 0.62, sw * 0.68, sh, 6) : new THREE.BoxGeometry(sw, sh, sw);
            const seg = new THREE.Mesh(geo, matWin);
            seg.position.y = s * sh + sh * 0.5;
            if (i % 2 === 1) seg.rotation.y = s * 0.18;
            group.add(seg);
        }
        const spire = new THREE.Mesh(new THREE.CylinderGeometry(0.2, h * 0.018, h * 0.16, 16), new THREE.MeshBasicMaterial({ color: 0xfef08a }));
        spire.position.y = h * 0.94; group.add(spire);
    }
    return group;
}

const meshes = [];
for (let i = 0; i < 5; i++) {
    const h = visualHeights[i];
    const x = xPositions[i];
    const [baseHex, glowHex, hexNum] = palettes[i % palettes.length];
    const itemObj = items[i] || {};

    const grp = buildDistinct3DObject(itemObj, i, h, baseHex, glowHex, hexNum);

    const pedR = Math.max(h * 0.22, objectWidths[i] * 0.45);
    const ped = new THREE.Mesh(new THREE.CylinderGeometry(pedR, pedR * 1.08, 2.0, 48), new THREE.MeshBasicMaterial({ color: hexNum }));
    ped.position.y = 1.0;
    grp.add(ped);

    grp.position.set(x, 0, 0);
    scene.add(grp);
    meshes.push(grp);
}

window.renderFrameAtTime = function(t) {
    let stageIndex = 0;
    let stageProgress = 0.0;
    const timings = DATA.scene_timings;
    for (let i = 0; i < timings.length; i++) {
        if (t >= timings[i].start && t <= timings[i].end + 0.05) {
            stageIndex = i;
            stageProgress = Math.min(Math.max((t - timings[i].start) / Math.max(timings[i].duration, 0.1), 0.0), 1.0);
            break;
        }
    }

    const floatIdx = Math.min(4.0, stageIndex + Math.pow(stageProgress, 1.5) * 0.88);
    const i0 = Math.min(4, Math.floor(floatIdx));
    const i1 = Math.min(4, i0 + 1);
    const frac = floatIdx - i0;
    const smoothFrac = frac * frac * (3 - 2 * frac);

    const curX = xPositions[i0] * (1 - smoothFrac) + xPositions[i1] * smoothFrac;
    const curH = visualHeights[i0] * (1 - smoothFrac) + visualHeights[i1] * smoothFrac;
    const prevX = (i0 > 0) ? xPositions[i0 - 1] : curX;

    meshes.forEach((m, idx) => {
        m.rotation.y = Math.sin(t * 0.6 + idx) * 0.25;
    });

    // Frame BOTH the current object AND the previous smaller object on the left so the viewer sees the true scale jump!
    const lookX = (i0 === 0 && smoothFrac < 0.2) ? curX : (curX * 0.72 + prevX * 0.28);
    const dist = Math.max(curH * (isWide ? 2.15 : 2.95), 75);
    camera.position.set(
        lookX - dist * 0.12,
        curH * 0.54 + 10,
        dist
    );
    camera.lookAt(lookX, curH * 0.48, 0);

    renderer.render(scene, camera);

    // 2D Broadcast HUD + Picture-in-Picture Real Photo + Scale Comparison Multiplier Badge!
    ctx.clearRect(0, 0, W, H);
    const activeItem = items[stageIndex] || { name: 'OBJECT ' + (stageIndex + 1), metric: '100 m', subtitle: '3D SCALE' };
    const activeImg = loadedImages[stageIndex];

    const cardW = isWide ? 860 : (W - 84);
    const cardX = (W - cardW) / 2;
    const cardY = isWide ? 44 : 88;

    ctx.save();
    ctx.fillStyle = 'rgba(15, 23, 42, 0.94)';
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.roundRect(cardX, cardY, cardW, 195, 24);
    ctx.fill();
    ctx.stroke();

    let textOffsetX = cardX + 28;
    if (activeImg && activeImg.complete && activeImg.naturalWidth > 0) {
        ctx.save();
        ctx.beginPath();
        ctx.roundRect(cardX + 20, cardY + 22, 150, 150, 16);
        ctx.clip();
        ctx.drawImage(activeImg, cardX + 20, cardY + 22, 150, 150);
        ctx.restore();
        ctx.strokeStyle = '#facc15';
        ctx.lineWidth = 4;
        ctx.strokeRect(cardX + 20, cardY + 22, 150, 150);
        textOffsetX = cardX + 192;
    }

    ctx.fillStyle = '#f59e0b';
    ctx.beginPath();
    ctx.roundRect(textOffsetX, cardY + 24, 150, 40, 10);
    ctx.fill();
    ctx.fillStyle = '#0f172a';
    ctx.font = 'bold 21px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('RANK #0' + (stageIndex + 1), textOffsetX + 75, cardY + 51);

    ctx.textAlign = 'left';
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 34px sans-serif';
    ctx.fillText(String(activeItem.name || '').toUpperCase().slice(0, 22), textOffsetX + 165, cardY + 54);

    ctx.fillStyle = '#38bdf8';
    ctx.font = 'bold 54px sans-serif';
    ctx.fillText(String(activeItem.metric || ''), textOffsetX, cardY + 132);

    // Show Multiplier vs Rank #1 so viewer immediately grasps the scale jump!
    const multVsFirst = (rawMeters[stageIndex] / Math.max(rawMeters[0], 0.1)).toFixed(1);
    const subTxt = stageIndex > 0 ? (`${multVsFirst}X LARGER THAN ${String(items[0].name || 'RANK #1').toUpperCase().slice(0, 14)}`) : String(activeItem.subtitle || '').toUpperCase();
    ctx.fillStyle = '#facc15';
    ctx.font = 'bold 22px sans-serif';
    ctx.fillText(subTxt.slice(0, 36), textOffsetX, cardY + 170);
    ctx.restore();

    const chunks = DATA.chunks || [];
    let activeChunk = null;
    for (let c of chunks) {
        if (t >= c.start - 0.04 && t <= c.end + 0.08) { activeChunk = c; break; }
    }
    if (activeChunk) {
        ctx.save();
        const words = activeChunk.words.map(w => String(w.word || '').toUpperCase()).join(' ');
        ctx.font = isWide ? 'bold 42px sans-serif' : 'bold 44px sans-serif';
        const tw = Math.min(ctx.measureText(words).width + 60, W - 80);
        const by = isWide ? (H - 135) : 1560;
        ctx.fillStyle = 'rgba(15, 23, 42, 0.92)';
        ctx.strokeStyle = '#fef08a';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.roundRect((W - tw) / 2, by - 52, tw, 76, 16);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = '#fef08a';
        ctx.textAlign = 'center';
        ctx.fillText(words, W / 2, by);
        ctx.restore();
    }
};
</script>
</body>
</html>
"""


# =====================================================================
# 4. BESPOKE ENGINE 2: KINETICZEN (PHOTO AVATAR BALLS + 7 RINGS + WINNER BANNER!)
# =====================================================================
MATTERJS_PHYSICS_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0b0716; overflow: hidden; font-family: 'DejaVu Sans', 'Liberation Sans', sans-serif; }
  canvas { display: block; }
</style>
<script src="__MATTER_JS_SRC__"></script>
</head>
<body>
<canvas id="zenCanvas"></canvas>
<script>
const DATA = __PAYLOAD_JSON__;
const W = DATA.width;
const H = DATA.height;
const modeIdx = (DATA.physics_mode || 0) % 5;
const teamAName = String(DATA.team_a_name || 'RONALDO').toUpperCase().slice(0, 10);
const teamBName = String(DATA.team_b_name || 'MESSI').toUpperCase().slice(0, 10);
const hookQuestion = String(DATA.hook_banner || (teamAName + ' VS ' + teamBName + ': WHO WINS?')).toUpperCase();

const canvas = document.getElementById('zenCanvas');
canvas.width = W;
canvas.height = H;
const ctx = canvas.getContext('2d');

let imgA = null, imgB = null;
window.waitForAllImages = async function() {
    const loadOne = (b64) => new Promise(res => {
        if (!b64) return res(null);
        const im = new Image();
        im.onload = () => res(im);
        im.onerror = () => res(null);
        im.src = b64;
    });
    imgA = await loadOne(DATA.img_a_b64);
    imgB = await loadOne(DATA.img_b_b64);
};

const PALETTES = [
    { bg1: '#1a0b2e', bg2: '#070312', teamA: '#00e5ff', teamB: '#ff2a85', rings: ['#ff2a85', '#ffae00', '#00f5d4', '#38bdf8', '#a855f7', '#10b981', '#facc15', '#f43f5e'] },
    { bg1: '#041f1e', bg2: '#020b0e', teamA: '#00ff9d', teamB: '#ff5500', rings: ['#00ff9d', '#00e5ff', '#ff5500', '#facc15', '#38bdf8', '#a855f7', '#ec4899', '#10b981'] },
    { bg1: '#091931', bg2: '#030814', teamA: '#facc15', teamB: '#f43f5e', rings: ['#facc15', '#f43f5e', '#00e5ff', '#8b5cf6', '#10b981', '#ec4899', '#38bdf8', '#ffae00'] },
    { bg1: '#230b1e', bg2: '#0e030c', teamA: '#bef264', teamB: '#d946ef', rings: ['#bef264', '#d946ef', '#06b6d4', '#f59e0b', '#ec4899', '#6366f1', '#10b981', '#f43f5e'] }
];
const pal = PALETTES[(DATA.palette_seed || 0) % PALETTES.length];

const Engine = Matter.Engine,
      Bodies = Matter.Bodies,
      Body = Matter.Body,
      Composite = Matter.Composite,
      Events = Matter.Events;

const engine = Engine.create();
engine.gravity.y = (modeIdx === 2) ? 0.85 : 0.72;
const world = engine.world;

const cx = W / 2;
const cy = H * 0.58;
const rings = [];
const pegs = [];
const particles = [];
const shockwaves = [];
const balls = [];
window.collisionEvents = [];
let shakeFrames = 0;

if (modeIdx === 0 || modeIdx === 3 || modeIdx === 4) {
    const ringRadii = [140, 192, 244, 296, 348, 400, 452];
    ringRadii.forEach((radius, rIdx) => {
        const segments = 72;
        const gapSegments = 6;
        const segBodies = [];
        for (let s = gapSegments; s < segments; s++) {
            const midA = ((s + 0.5) / segments) * Math.PI * 2;
            const sx = cx + Math.cos(midA) * radius;
            const sy = cy + Math.sin(midA) * radius;
            const arcLen = ((2 * Math.PI * radius) / segments) * 1.45;
            const seg = Bodies.rectangle(sx, sy, arcLen, 24, {
                isStatic: true, angle: midA + Math.PI / 2, restitution: 1.02, friction: 0.0
            });
            segBodies.push({ body: seg, relAngle: midA, radius: radius });
            Composite.add(world, seg);
        }
        rings.push({
            idx: rIdx, radius: radius, color: pal.rings[rIdx % pal.rings.length],
            angleOffset: rIdx * 1.05, speed: (rIdx % 2 === 0 ? 1 : -1) * (0.024 + rIdx * 0.003),
            segments: segBodies, escapedCount: 0, destroyed: false, isPolygon: false
        });
    });

    const nBalls = (modeIdx === 3) ? 11 : 21;
    for (let i = 0; i < nBalls; i++) {
        const isA = (i % 2 === 0);
        const angle = (i / nBalls) * Math.PI * 2;
        const r = 25 + (i % 3) * 22;
        const bRad = (modeIdx === 3) ? 17 : 22;
        const ball = Bodies.circle(cx + Math.cos(angle) * r, cy + Math.sin(angle) * r, bRad, {
            restitution: 1.02, friction: 0.0, frictionAir: 0.0002
        });
        Body.setVelocity(ball, { x: Math.cos(angle * 2.1) * 9.2, y: Math.sin(angle * 1.7) * 9.2 });
        Composite.add(world, ball);
        balls.push({ body: ball, isA: isA, rad: bRad, ringStage: 0, trail: [] });
    }
} else if (modeIdx === 1) {
    const polyRadii = [165, 240, 315, 390, 460];
    const sides = 6;
    polyRadii.forEach((radius, rIdx) => {
        const segBodies = [];
        for (let s = 0; s < sides; s++) {
            const a1 = (s / sides) * Math.PI * 2;
            const a2 = ((s + 1) / sides) * Math.PI * 2;
            const midA = (a1 + a2) / 2;
            const sideLen = 2 * radius * Math.sin(Math.PI / sides);
            const actualLen = (s === 0) ? sideLen * 0.42 : sideLen * 1.06;
            const rDist = radius * Math.cos(Math.PI / sides);
            const sx = cx + Math.cos(midA) * rDist;
            const sy = cy + Math.sin(midA) * rDist;
            const seg = Bodies.rectangle(sx, sy, actualLen, 24, {
                isStatic: true, angle: midA + Math.PI / 2, restitution: 1.03, friction: 0.0
            });
            segBodies.push({ body: seg, relAngle: midA, radius: rDist });
            Composite.add(world, seg);
        }
        rings.push({
            idx: rIdx, radius: radius, color: pal.rings[rIdx % pal.rings.length],
            angleOffset: rIdx * 0.7, speed: (rIdx % 2 === 0 ? 1 : -1) * 0.022,
            segments: segBodies, escapedCount: 0, destroyed: false, isPolygon: true
        });
    });

    for (let i = 0; i < 21; i++) {
        const isA = (i % 2 === 0);
        const a = (i / 21) * Math.PI * 2;
        const ball = Bodies.circle(cx + Math.cos(a) * 55, cy + Math.sin(a) * 55, 22, {
            restitution: 1.02, friction: 0.0, frictionAir: 0.0002
        });
        Body.setVelocity(ball, { x: Math.cos(a) * 9, y: Math.sin(a) * 9 });
        Composite.add(world, ball);
        balls.push({ body: ball, isA: isA, rad: 22, ringStage: 0, trail: [] });
    }
} else {
    const rows = 8;
    for (let r = 0; r < rows; r++) {
        const cols = (r % 2 === 0) ? 7 : 6;
        const spacingX = 115;
        const startX = cx - ((cols - 1) * spacingX) / 2;
        const py = 430 + r * 115;
        for (let c = 0; c < cols; c++) {
            const px = startX + c * spacingX;
            const peg = Bodies.circle(px, py, 14, { isStatic: true, restitution: 0.95, friction: 0.0 });
            Composite.add(world, peg);
            pegs.push({ x: px, y: py, r: 14, color: pal.rings[r % pal.rings.length] });
        }
    }
    Composite.add(world, Bodies.rectangle(75, H / 2, 30, H, { isStatic: true, restitution: 0.9 }));
    Composite.add(world, Bodies.rectangle(W - 75, H / 2, 30, H, { isStatic: true, restitution: 0.9 }));

    for (let i = 0; i < 25; i++) {
        const isA = (i % 2 === 0);
        const bx = cx + (Math.sin(i * 17.3) * 260);
        const by = 330 + (i % 5) * 18;
        const ball = Bodies.circle(bx, by, 22, { restitution: 0.92, friction: 0.001, frictionAir: 0.001 });
        Body.setVelocity(ball, { x: Math.cos(i * 5) * 4, y: 2 });
        Composite.add(world, ball);
        balls.push({ body: ball, isA: isA, rad: 22, ringStage: 0, trail: [] });
    }
}

let simTime = 0.0;
let lastNoteTime = -1.0;
let noteCounter = 0;
Events.on(engine, 'collisionStart', function(ev) {
    if (simTime - lastNoteTime >= 0.072) {
        lastNoteTime = simTime;
        noteCounter++;
        window.collisionEvents.push({
            time: Number(simTime.toFixed(3)),
            intensity: 0.85,
            noteIdx: noteCounter % 9
        });
        if (ev.pairs && ev.pairs.length > 0 && shockwaves.length < 18) {
            const pos = ev.pairs[0].bodyA.position;
            shockwaves.push({ x: pos.x, y: pos.y, r: 12, alpha: 0.75, color: noteCounter % 2 === 0 ? pal.teamA : pal.teamB });
        }
    }
});

let scoreA = 15;
let scoreB = 10;

window.stepAndRenderPhysics = function(frameIdx) {
    simTime = frameIdx / 24.0;
    let activeRingIdx = rings.findIndex(r => !r.destroyed);
    if (activeRingIdx === -1) activeRingIdx = rings.length;

    for (let sub = 0; sub < 3; sub++) {
        rings.forEach(r => {
            if (r.destroyed) return;
            r.angleOffset += r.speed * 0.333;
            r.segments.forEach(segObj => {
                const newA = segObj.relAngle + r.angleOffset;
                const nx = cx + Math.cos(newA) * segObj.radius;
                const ny = cy + Math.sin(newA) * segObj.radius;
                Body.setPosition(segObj.body, { x: nx, y: ny });
                Body.setAngle(segObj.body, newA + Math.PI / 2);
            });
        });
        Engine.update(engine, 1000 / 72);
    }

    balls.forEach(b => {
        const pos = b.body.position;
        const vel = b.body.velocity;
        const spd = Math.hypot(vel.x, vel.y);
        if (spd < 5.5 && modeIdx !== 2) {
            Body.setVelocity(b.body, { x: vel.x * 1.35 + (Math.random() - 0.5) * 2, y: vel.y * 1.35 - 1.5 });
        }
        if (modeIdx === 3 && b.rad < 36 && frameIdx % 18 === 0) {
            b.rad += 0.45;
            Body.scale(b.body, 1.025, 1.025);
        }

        b.trail.push({ x: pos.x, y: pos.y });
        if (b.trail.length > 7) b.trail.shift();

        if (modeIdx === 2) {
            if (pos.y > 1440) {
                if (b.isA) scoreA += 17; else scoreB += 14;
                shakeFrames = 5;
                for (let k = 0; k < 12; k++) {
                    particles.push({
                        x: pos.x, y: pos.y,
                        vx: (Math.random() - 0.5) * 16, vy: -Math.random() * 14,
                        color: b.isA ? pal.teamA : pal.teamB, alpha: 1.0
                    });
                }
                Body.setPosition(b.body, { x: cx + (Math.random() - 0.5) * 280, y: 310 });
                Body.setVelocity(b.body, { x: (Math.random() - 0.5) * 5, y: 2 });
            }
        } else {
            const dist = Math.hypot(pos.x - cx, pos.y - cy);
            if (activeRingIdx < rings.length) {
                const curRing = rings[activeRingIdx];
                if (b.ringStage <= activeRingIdx && dist > curRing.radius + 14) {
                    b.ringStage = activeRingIdx + 1;
                    curRing.escapedCount += 1;
                    if (b.isA) scoreA += 14; else scoreB += 11;

                    const minTime = (activeRingIdx + 1) * 4.6;
                    if (curRing.escapedCount >= 4 && simTime >= minTime) {
                        curRing.destroyed = true;
                        shakeFrames = 10;
                        if (b.isA) scoreA += 35; else scoreB += 28;
                        curRing.segments.forEach(s => {
                            Composite.remove(world, s.body);
                            particles.push({
                                x: s.body.position.x, y: s.body.position.y,
                                vx: (Math.random() - 0.5) * 18, vy: (Math.random() - 0.5) * 18,
                                color: curRing.color, alpha: 1.0
                            });
                        });
                        window.collisionEvents.push({ time: Number(simTime.toFixed(3)), intensity: 1.0, noteIdx: 8 });
                    }
                }
            }
            if (dist > 495) {
                const ang = Math.atan2(pos.y - cy, pos.x - cx);
                const nx = Math.cos(ang), ny = Math.sin(ang);
                Body.setPosition(b.body, { x: cx + nx * 490, y: cy + ny * 490 });
                const dot = vel.x * nx + vel.y * ny;
                Body.setVelocity(b.body, { x: (vel.x - 2 * dot * nx) * 0.98, y: (vel.y - 2 * dot * ny) * 0.98 - 2.0 });
            }
        }
    });

    ctx.save();
    if (shakeFrames > 0) {
        shakeFrames--;
        ctx.translate((Math.random() - 0.5) * 14, (Math.random() - 0.5) * 14);
    }

    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, pal.bg1);
    grad.addColorStop(1, pal.bg2);
    ctx.fillStyle = grad;
    ctx.fillRect(-30, -30, W + 60, H + 60);

    ctx.save();
    ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.22)';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.roundRect(48, 58, W - 96, 235, 28);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#fef08a';
    ctx.font = 'bold 26px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(hookQuestion.slice(0, 38), cx, 102);

    if (imgA && imgA.complete && imgA.naturalWidth > 0) {
        ctx.save();
        ctx.beginPath(); ctx.arc(110, 158, 34, 0, Math.PI * 2); ctx.clip();
        ctx.drawImage(imgA, 76, 124, 68, 68);
        ctx.restore();
        ctx.strokeStyle = pal.teamA; ctx.lineWidth = 4;
        ctx.beginPath(); ctx.arc(110, 158, 34, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.fillStyle = pal.teamA;
    ctx.font = 'bold 34px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(teamAName + ': ' + scoreA, 158, 170);

    if (imgB && imgB.complete && imgB.naturalWidth > 0) {
        ctx.save();
        ctx.beginPath(); ctx.arc(W - 110, 158, 34, 0, Math.PI * 2); ctx.clip();
        ctx.drawImage(imgB, W - 144, 124, 68, 68);
        ctx.restore();
        ctx.strokeStyle = pal.teamB; ctx.lineWidth = 4;
        ctx.beginPath(); ctx.arc(W - 110, 158, 34, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.fillStyle = pal.teamB;
    ctx.textAlign = 'right';
    ctx.fillText(teamBName + ': ' + scoreB, W - 158, 170);

    if (scoreA === scoreB) scoreA += 3;
    const totalSc = Math.max(scoreA + scoreB, 1);
    const pctA = scoreA / totalSc;
    const barX = 82, barY = 215, barW = W - 164, barH = 36;

    ctx.fillStyle = pal.teamB;
    ctx.beginPath(); ctx.roundRect(barX, barY, barW, barH, 16); ctx.fill();

    ctx.fillStyle = pal.teamA;
    ctx.beginPath(); ctx.roundRect(barX, barY, Math.max(45, barW * pctA), barH, 16); ctx.fill();

    ctx.fillStyle = '#090d16';
    ctx.font = 'bold 22px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(Math.round(pctA * 100) + '%', barX + 16, barY + 26);
    ctx.textAlign = 'right';
    ctx.fillText((100 - Math.round(pctA * 100)) + '%', barX + barW - 16, barY + 26);
    ctx.restore();

    for (let sw of shockwaves) {
        if (sw.alpha <= 0.03) continue;
        sw.r += 4.5; sw.alpha *= 0.86;
        ctx.save();
        ctx.globalAlpha = sw.alpha;
        ctx.strokeStyle = sw.color;
        ctx.lineWidth = 3;
        ctx.beginPath(); ctx.arc(sw.x, sw.y, sw.r, 0, Math.PI * 2); ctx.stroke();
        ctx.restore();
    }

    rings.forEach(r => {
        if (r.destroyed) return;
        ctx.save();
        ctx.strokeStyle = r.color;
        ctx.shadowBlur = 22;
        ctx.shadowColor = r.color;
        ctx.lineWidth = 18;
        ctx.lineCap = 'round';
        if (r.isPolygon) {
            r.segments.forEach((segObj, sIdx) => {
                const p = segObj.body.position;
                const ang = segObj.body.angle;
                const len = (sIdx === 0) ? r.radius * 0.38 : r.radius * 0.95;
                ctx.beginPath();
                ctx.moveTo(p.x - Math.cos(ang) * len * 0.5, p.y - Math.sin(ang) * len * 0.5);
                ctx.lineTo(p.x + Math.cos(ang) * len * 0.5, p.y + Math.sin(ang) * len * 0.5);
                ctx.stroke();
            });
        } else {
            ctx.beginPath();
            ctx.arc(cx, cy, r.radius, r.angleOffset + (6 / 72) * Math.PI * 2, r.angleOffset + Math.PI * 2);
            ctx.stroke();
        }
        ctx.restore();
    });

    pegs.forEach(pg => {
        ctx.save();
        ctx.fillStyle = pg.color;
        ctx.shadowBlur = 14;
        ctx.shadowColor = pg.color;
        ctx.beginPath(); ctx.arc(pg.x, pg.y, pg.r, 0, Math.PI * 2); ctx.fill();
        ctx.restore();
    });

    for (let p of particles) {
        if (p.alpha <= 0.02) continue;
        p.x += p.vx; p.y += p.vy; p.alpha *= 0.92;
        ctx.save();
        ctx.globalAlpha = p.alpha;
        ctx.fillStyle = p.color;
        ctx.beginPath(); ctx.arc(p.x, p.y, 8, 0, Math.PI * 2); ctx.fill();
        ctx.restore();
    }

    balls.forEach(b => {
        const col = b.isA ? pal.teamA : pal.teamB;
        const avatarImg = b.isA ? imgA : imgB;
        b.trail.forEach((tp, idx) => {
            ctx.save();
            ctx.globalAlpha = (idx / b.trail.length) * 0.32;
            ctx.fillStyle = col;
            ctx.beginPath(); ctx.arc(tp.x, tp.y, 8 + idx, 0, Math.PI * 2); ctx.fill();
            ctx.restore();
        });

        const pos = b.body.position;
        ctx.save();
        if (avatarImg && avatarImg.complete && avatarImg.naturalWidth > 0) {
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, b.rad, 0, Math.PI * 2);
            ctx.clip();
            ctx.drawImage(avatarImg, pos.x - b.rad, pos.y - b.rad, b.rad * 2, b.rad * 2);
            ctx.restore();
            ctx.save();
            ctx.strokeStyle = col;
            ctx.lineWidth = 4;
            ctx.shadowBlur = 14;
            ctx.shadowColor = col;
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, b.rad, 0, Math.PI * 2);
            ctx.stroke();
        } else {
            ctx.shadowBlur = 16;
            ctx.shadowColor = col;
            ctx.fillStyle = col;
            ctx.beginPath(); ctx.arc(pos.x, pos.y, b.rad, 0, Math.PI * 2); ctx.fill();
            ctx.fillStyle = '#090d16';
            ctx.font = `bold ${Math.max(12, Math.floor(b.rad * 0.75))}px sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(b.isA ? teamAName.slice(0, 2) : teamBName.slice(0, 2), pos.x, pos.y + 1);
        }
        ctx.restore();
    });

    const allRingsBroken = rings.length > 0 && rings.every(r => r.destroyed);
    if (allRingsBroken || simTime >= 33.2) {
        const winName = scoreA >= scoreB ? teamAName : teamBName;
        const winCol = scoreA >= scoreB ? pal.teamA : pal.teamB;
        ctx.save();
        ctx.fillStyle = 'rgba(9, 13, 22, 0.92)';
        ctx.strokeStyle = winCol;
        ctx.lineWidth = 6;
        ctx.beginPath();
        ctx.roundRect(90, cy - 110, W - 180, 220, 32);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = '#facc15';
        ctx.font = 'bold 34px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('CHAMPIONSHIP WINNER', cx, cy - 35);

        ctx.fillStyle = winCol;
        ctx.font = 'bold 64px sans-serif';
        ctx.fillText(winName + ' WINS!', cx, cy + 55);
        ctx.restore();
    }

    ctx.restore();
};
</script>
</body>
</html>
"""


# =====================================================================
# 5. BESPOKE ENGINE 3: CAPITALGRID (TRUE STAGE CRASH LOGIC + REAL ASSET PHOTOS!)
# =====================================================================
TRADINGVIEW_WEALTH_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #090d16; overflow: hidden; font-family: 'DejaVu Sans', 'Liberation Sans', sans-serif; }
  canvas { display: block; }
</style>
</head>
<body>
<canvas id="wealthCanvas"></canvas>
<script>
const DATA = __PAYLOAD_JSON__;
const W = DATA.width;
const H = DATA.height;
const isWide = DATA.is_widescreen;
const canvas = document.getElementById('wealthCanvas');
canvas.width = W;
canvas.height = H;
const ctx = canvas.getContext('2d');

const FIN_THEMES = [
    { bg: '#090d16', colA: '#3b82f6', colB: '#10b981', fillA: 'rgba(59,130,246,0.26)', fillB: 'rgba(16,185,129,0.34)' },
    { bg: '#110c1d', colA: '#f59e0b', colB: '#ec4899', fillA: 'rgba(245,158,11,0.26)', fillB: 'rgba(236,72,153,0.34)' },
    { bg: '#07161b', colA: '#06b6d4', colB: '#facc15', fillA: 'rgba(6,182,212,0.26)', fillB: 'rgba(250,204,21,0.34)' },
    { bg: '#140a0e', colA: '#a855f7', colB: '#22c55e', fillA: 'rgba(168,85,247,0.26)', fillB: 'rgba(34,197,94,0.34)' }
];
const theme = FIN_THEMES[(DATA.theme_seed || 0) % FIN_THEMES.length];

const waypointsA = DATA.series_a || [5000, 6200, 7800, 9500, 12400, 16800];
const waypointsB = DATA.series_b || [5000, 9400, 6100, 22000, 54000, 118000];
const nameA = String(DATA.asset_a_name || 'ASSET A').toUpperCase();
const nameB = String(DATA.asset_b_name || 'ASSET B').toUpperCase();
const startYear = Number(DATA.start_year || 2005);
const endYear = Number(DATA.end_year || 2025);

let imgA = null, imgB = null;
window.waitForAllImages = async function() {
    const loadOne = (b64) => new Promise(res => {
        if (!b64) return res(null);
        const im = new Image();
        im.onload = () => res(im);
        im.onerror = () => res(null);
        im.src = b64;
    });
    imgA = await loadOne(DATA.img_a_b64);
    imgB = await loadOne(DATA.img_b_b64);
};

const coins = [];
for (let c = 0; c < 25; c++) {
    coins.push({ x: (c * 73) % W, y: (c * 137) % H, speed: 2.5 + (c % 5), size: 6 + (c % 6) });
}

function sampleMarketValue(waypoints, exactStageFloat, isVolatile) {
    const clamped = Math.min(Math.max(exactStageFloat, 0.0), 5.0);
    const i0 = Math.min(4, Math.floor(clamped));
    const i1 = Math.min(5, i0 + 1);
    const frac = clamped - i0;
    const baseVal = waypoints[i0] * (1 - frac) + waypoints[i1] * frac;
    const pinEnvelope = Math.sin(frac * Math.PI);
    const wave = (Math.sin(clamped * 24.0) * 0.032 + Math.cos(clamped * 55.0) * 0.018) * (isVolatile ? 1.25 : 0.55);
    return Math.max(10, baseVal * (1 + wave * pinEnvelope));
}

function fmtMoney(val) {
    if (val >= 1e9) return '$' + (val / 1e9).toFixed(2) + 'B';
    if (val >= 1e6) return '$' + (val / 1e6).toFixed(2) + 'M';
    if (val >= 1e3) return '$' + (val / 1e3).toFixed(1) + 'K';
    return '$' + Math.round(val).toLocaleString();
}

window.renderFrameAtTime = function(t) {
    let stageIndex = 0;
    let stageProgress = 0.0;
    const timings = DATA.scene_timings || [];
    for (let i = 0; i < timings.length; i++) {
        if (t >= timings[i].start && t <= timings[i].end + 0.05) {
            stageIndex = i;
            stageProgress = Math.min(Math.max((t - timings[i].start) / Math.max(timings[i].duration, 0.1), 0.0), 1.0);
            break;
        }
    }
    const exactStageFloat = Math.min(5.0, stageIndex + stageProgress);
    const normProg = exactStageFloat / 5.0;

    const curValA = sampleMarketValue(waypointsA, exactStageFloat, false);
    const curValB = sampleMarketValue(waypointsB, exactStageFloat, true);

    const stStartA = waypointsA[Math.min(4, stageIndex)];
    const stEndA = waypointsA[Math.min(5, stageIndex + 1)];
    const stStartB = waypointsB[Math.min(4, stageIndex)];
    const stEndB = waypointsB[Math.min(5, stageIndex + 1)];
    const isCrashing = (stEndB < stStartB * 0.92) || (stEndA < stStartA * 0.92);

    ctx.save();
    if (isCrashing) {
        ctx.translate(Math.sin(t * 50) * 4, Math.cos(t * 45) * 4);
    }

    ctx.fillStyle = theme.bg;
    ctx.fillRect(-20, -20, W + 40, H + 40);

    if (isCrashing) {
        const cGrad = ctx.createRadialGradient(W/2, H/2, W*0.2, W/2, H/2, W*0.9);
        cGrad.addColorStop(0, 'rgba(239, 68, 68, 0)');
        cGrad.addColorStop(1, `rgba(239, 68, 68, ${0.22 + 0.14 * Math.sin(t * 16)})`);
        ctx.fillStyle = cGrad;
        ctx.fillRect(0, 0, W, H);
    }

    ctx.fillStyle = 'rgba(250, 204, 21, 0.12)';
    coins.forEach(c => {
        const cyPos = (c.y - t * 45 * c.speed % H + H) % H;
        ctx.beginPath(); ctx.arc(c.x, cyPos, c.size, 0, Math.PI * 2); ctx.fill();
    });

    ctx.fillStyle = 'rgba(17, 24, 39, 0.95)';
    ctx.fillRect(0, 0, W, 76);
    ctx.fillStyle = isCrashing ? '#ef4444' : theme.colB;
    ctx.font = 'bold 24px sans-serif';
    const curYear = Math.round(startYear + normProg * (endYear - startYear));
    ctx.fillText((isCrashing ? '⚠️ MARKET CRASH ALERT  •  YEAR: ' : 'LIVE WEALTH BACKTEST  •  YEAR: ') + curYear, 48, 47);

    const padX = 48;
    const cardW = (W - padX * 3) / 2;
    const cardY = 102;
    const cardH = isWide ? 190 : 240;

    ctx.fillStyle = '#111827';
    ctx.strokeStyle = theme.colA;
    ctx.lineWidth = 4;
    ctx.beginPath(); ctx.roundRect(padX, cardY, cardW, cardH, 22); ctx.fill(); ctx.stroke();

    if (imgA && imgA.complete && imgA.naturalWidth > 0) {
        ctx.save();
        ctx.beginPath(); ctx.arc(padX + cardW - 56, cardY + 56, 38, 0, Math.PI * 2); ctx.clip();
        ctx.drawImage(imgA, padX + cardW - 94, cardY + 18, 76, 76);
        ctx.restore();
        ctx.strokeStyle = theme.colA; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.arc(padX + cardW - 56, cardY + 56, 38, 0, Math.PI * 2); ctx.stroke();
    }

    ctx.fillStyle = '#cbd5e1';
    ctx.font = 'bold 23px sans-serif';
    ctx.fillText(nameA.slice(0, 15), padX + 26, cardY + 48);
    ctx.fillStyle = '#ffffff';
    ctx.font = isWide ? 'bold 54px sans-serif' : 'bold 52px sans-serif';
    ctx.fillText(fmtMoney(curValA), padX + 26, cardY + 130);

    const roiA = (((curValA - waypointsA[0]) / Math.max(waypointsA[0], 1)) * 100).toFixed(0);
    ctx.fillStyle = Number(roiA) >= 0 ? theme.colA : '#ef4444';
    ctx.font = 'bold 24px sans-serif';
    ctx.fillText((Number(roiA) >= 0 ? '+' : '') + Number(roiA).toLocaleString() + '% ROI', padX + 26, cardY + cardH - 26);

    const cardBX = padX * 2 + cardW;
    ctx.fillStyle = '#111827';
    ctx.strokeStyle = isCrashing ? '#ef4444' : theme.colB;
    ctx.lineWidth = 4;
    ctx.beginPath(); ctx.roundRect(cardBX, cardY, cardW, cardH, 22); ctx.fill(); ctx.stroke();

    if (imgB && imgB.complete && imgB.naturalWidth > 0) {
        ctx.save();
        ctx.beginPath(); ctx.arc(cardBX + cardW - 56, cardY + 56, 38, 0, Math.PI * 2); ctx.clip();
        ctx.drawImage(imgB, cardBX + cardW - 94, cardY + 18, 76, 76);
        ctx.restore();
        ctx.strokeStyle = theme.colB; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.arc(cardBX + cardW - 56, cardY + 56, 38, 0, Math.PI * 2); ctx.stroke();
    }

    ctx.fillStyle = '#cbd5e1';
    ctx.font = 'bold 23px sans-serif';
    ctx.fillText(nameB.slice(0, 15), cardBX + 26, cardY + 48);
    ctx.fillStyle = isCrashing ? '#ef4444' : theme.colB;
    ctx.font = isWide ? 'bold 54px sans-serif' : 'bold 52px sans-serif';
    ctx.fillText(fmtMoney(curValB), cardBX + 26, cardY + 130);

    const roiB = (((curValB - waypointsB[0]) / Math.max(waypointsB[0], 1)) * 100).toFixed(0);
    ctx.fillStyle = Number(roiB) >= 0 ? '#fbbf24' : '#ef4444';
    ctx.font = 'bold 24px sans-serif';
    ctx.fillText((Number(roiB) >= 0 ? '+' : '') + Number(roiB).toLocaleString() + '% ROI', cardBX + 26, cardY + cardH - 26);

    const mult = (Math.max(curValA, curValB) / Math.max(Math.min(curValA, curValB), 1)).toFixed(1);
    ctx.fillStyle = '#facc15';
    ctx.beginPath(); ctx.roundRect(W / 2 - 125, cardY + cardH + 8, 250, 42, 20); ctx.fill();
    ctx.fillStyle = '#090d16';
    ctx.font = 'bold 22px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('GAP: ' + mult + 'X WEALTHIER', W / 2, cardY + cardH + 36);
    ctx.textAlign = 'left';

    const chartX = padX;
    const chartY = cardY + cardH + 62;
    const chartW = W - padX * 2;
    const chartH = (isWide ? (H - 180) : 1385) - chartY;

    ctx.fillStyle = '#0d1322';
    ctx.strokeStyle = '#1f2937';
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.roundRect(chartX, chartY, chartW, chartH, 22); ctx.fill(); ctx.stroke();

    const numSamples = 120;
    const ptsA = [];
    const ptsB = [];
    let currentMax = Math.max(waypointsA[0], waypointsB[0]) * 1.15;

    for (let i = 0; i <= numSamples; i++) {
        const subStage = (i / numSamples) * Math.max(exactStageFloat, 0.08);
        const va = sampleMarketValue(waypointsA, subStage, false);
        const vb = sampleMarketValue(waypointsB, subStage, true);
        if (va > currentMax) currentMax = va;
        if (vb > currentMax) currentMax = vb;
        ptsA.push({ sp: i / numSamples, v: va });
        ptsB.push({ sp: i / numSamples, v: vb });
    }

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.07)';
    ctx.fillStyle = '#64748b';
    ctx.font = 'bold 20px sans-serif';
    for (let g = 0; g <= 4; g++) {
        const gy = chartY + 40 + (g / 4) * (chartH - 110);
        const gVal = currentMax * (1 - g / 4);
        ctx.beginPath(); ctx.moveTo(chartX + 25, gy); ctx.lineTo(chartX + chartW - 25, gy); ctx.stroke();
        ctx.fillText(fmtMoney(gVal), chartX + 32, gy - 8);
    }

    for (let b = 0; b < 36; b++) {
        const bx = chartX + 40 + (b / 36) * (chartW - 95);
        const bh = 12 + Math.abs(Math.sin(b * 3.7 + exactStageFloat * 4)) * 45;
        ctx.fillStyle = isCrashing ? 'rgba(239, 68, 68, 0.25)' : 'rgba(16, 185, 129, 0.22)';
        ctx.fillRect(bx, chartY + chartH - 12 - bh, 14, bh);
    }

    function drawAreaCurve(pts, strokeCol, fillTop) {
        ctx.save();
        ctx.beginPath();
        pts.forEach((pt, idx) => {
            const px = chartX + 40 + pt.sp * (chartW - 115);
            const py = chartY + chartH - 55 - (pt.v / currentMax) * (chartH - 105);
            if (idx === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        });
        ctx.strokeStyle = strokeCol;
        ctx.lineWidth = 7;
        ctx.shadowBlur = 16;
        ctx.shadowColor = strokeCol;
        ctx.stroke();

        const lastPt = pts[pts.length - 1];
        const lastX = chartX + 40 + lastPt.sp * (chartW - 115);
        const lastY = chartY + chartH - 55 - (lastPt.v / currentMax) * (chartH - 105);
        ctx.lineTo(lastX, chartY + chartH - 55);
        ctx.lineTo(chartX + 40, chartY + chartH - 55);
        ctx.closePath();
        const aGrad = ctx.createLinearGradient(0, chartY, 0, chartY + chartH);
        aGrad.addColorStop(0, fillTop);
        aGrad.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = aGrad;
        ctx.fill();

        ctx.beginPath();
        ctx.arc(lastX, lastY, 12, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
        ctx.restore();
    }

    drawAreaCurve(ptsA, theme.colA, theme.fillA);
    drawAreaCurve(ptsB, isCrashing ? '#ef4444' : theme.colB, theme.fillB);

    const chunks = DATA.chunks || [];
    let activeChunk = null;
    for (let c of chunks) {
        if (t >= c.start - 0.04 && t <= c.end + 0.08) { activeChunk = c; break; }
    }
    if (activeChunk) {
        ctx.save();
        const words = activeChunk.words.map(w => String(w.word || '').toUpperCase()).join(' ');
        ctx.font = isWide ? 'bold 42px sans-serif' : 'bold 44px sans-serif';
        const tw = Math.min(ctx.measureText(words).width + 60, W - 80);
        const by = isWide ? (H - 65) : 1545;
        ctx.fillStyle = '#111827';
        ctx.strokeStyle = theme.colB;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.roundRect((W - tw) / 2, by - 52, tw, 76, 16);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'center';
        ctx.fillText(words, W / 2, by);
        ctx.restore();
    }
    ctx.restore();
};
</script>
</body>
</html>
"""


# =====================================================================
# 6. AGENT 2 & 3: DYNAMIC XGBOOST TOPIC ARCHITECT + WIKIMEDIA PHOTOS
# =====================================================================
SCALE3D_THEME_WHEEL = [
    ("Largest Rockets & Spacecraft in History (from V-2 Rocket to Falcon 9, Saturn V & Starship)", "rocket"),
    ("Tallest Mountains & Volcanoes in the Solar System (from Mount Fuji to Everest & Olympus Mons)", "mountain"),
    ("Iconic World Wonders & Towers (from Taj Mahal to Great Pyramid, Eiffel Tower & Burj Khalifa)", "dome_palace"),
    ("Planets, Moons & Stars of the Universe (from Moon to Earth, Saturn & The Sun)", "planet"),
    ("Deepest Ocean Trenches & Submarines (from Scuba Limit to Titanic Wreck & Mariana Trench)", "trench"),
    ("Fastest Vehicles, Jets & Spacecraft Ever Built (from Bugatti to SR-71 Blackbird & Parker Probe)", "rocket"),
    ("Colossal Asteroid Impacts & Craters Compared to Cities", "crater"),
    ("Biggest Megastructures & Skyscrapers Around the Globe", "skyscraper")
]

WEALTH_THEME_WHEEL = [
    "$5/Day Coffee Habit vs Investing in Nvidia Stock over 15 Years",
    "High-Yield Savings Account vs Dividend Aristocrats Snowball ($10,000 Start)",
    "Cash Under the Mattress (Inflation Erosion) vs Bitcoin over 12 Years",
    "Warren Buffett Value Strategy vs Day Trading Meme Stocks ($10,000 Simulation)",
    "Renting & Investing the Difference vs Buying a Luxury House (25-Year Math)",
    "Buying Every New iPhone vs Buying Apple Stock Instead ($15,000 Math)",
    "Investing $100/Month Starting at Age 20 vs Starting at Age 35 (Compounding Secret)",
    "Gold Bullion vs Real Estate: Investing $5,000 over 30 Years"
]

PHYSICS_RIVALRY_WHEEL = [
    (1, "IPHONE", "ANDROID", "iPhone", "Android (operating system)", "IPHONE VS ANDROID: HEXAGON CAGE SURVIVAL!"),
    (2, "DOGS", "CATS", "Dog", "Cat", "DOGS VS CATS: 25-BALL PLINKO JACKPOT RACE!"),
    (0, "RONALDO", "MESSI", "Cristiano Ronaldo", "Lionel Messi", "RONALDO VS MESSI: 7-RING ESCAPE CHAMPIONSHIP!"),
    (3, "GOLD", "BITCOIN", "Gold bar", "Bitcoin", "GOLD VS BITCOIN: BALLS GROW ON EVERY BOUNCE!"),
    (4, "TESLA", "FERRARI", "Tesla, Inc.", "Ferrari", "TESLA VS FERRARI: VORTEX RING ESCAPE!"),
    (0, "REAL MADRID", "BARCELONA", "Real Madrid CF", "FC Barcelona", "REAL MADRID VS BARCELONA: 7-RING ARENA!"),
    (1, "NETFLIX", "YOUTUBE", "Netflix", "YouTube", "NETFLIX VS YOUTUBE: HEXAGON TUMBLER BATTLE!"),
    (2, "USA", "JAPAN", "Flag of the United States", "Flag of Japan", "USA VS JAPAN: ULTIMATE PLINKO PHYSICS RACE!")
]


def generate_pod_plan(spy_data, past_topics, unit_idx=0, is_long=False):
    run_idx = spy_data.get("run_idx", 0) + unit_idx
    elite_titles = [c["title"] for c in spy_data["elite_candidates"][:10]]
    word_rule = "13 to 16 spoken words" if not is_long else "30 to 35 spoken words"

    if POD_NAME == "PHYSICS":
        mode_id, team_a, team_b, wiki_a, wiki_b, hook_banner = PHYSICS_RIVALRY_WHEEL[run_idx % len(PHYSICS_RIVALRY_WHEEL)]
        prompt = f"""
        We are generating a viral, zero-language ASMR Physics Short featuring the rivalry: "{team_a} vs {team_b}" ({hook_banner}).
        Top XGBoost Search Queries: {json.dumps(elite_titles[:6])}
        PAST TITLES (DO NOT REPEAT): {json.dumps(past_topics[-35:])}

        Write a fresh, high-CTR title (under 58 chars with #shorts), summary, and interactive pinned comment asking viewers whether they backed {team_a} or {team_b}.
        Output JSON ONLY:
        {{
          "title": "{team_a} vs {team_b}: Who Escapes the Physics Cage? #shorts",
          "topic_summary": "60FPS Matter.js physics rivalry between {team_a} and {team_b}.",
          "pinned_comment": "🔥 Team {team_a} or Team {team_b} — who did you bet on before the final winner banner appeared? Comment below! 👇"
        }}
        """
        plan = extract_json(call_ai(SCRIPT_MODELS, "Output valid JSON only.", prompt, 0.8))
        plan["physics_mode"] = mode_id
        plan["team_a_name"] = team_a
        plan["team_b_name"] = team_b
        plan["hook_banner"] = hook_banner
        plan["img_a_b64"] = fetch_wikipedia_image_b64(wiki_a)
        plan["img_b_b64"] = fetch_wikipedia_image_b64(wiki_b)
        plan["palette_seed"] = random.randint(0, 999)
        return plan

    elif POD_NAME == "SCALE3D":
        theme_desc, default_model = SCALE3D_THEME_WHEEL[run_idx % len(SCALE3D_THEME_WHEEL)]
        prompt = f"""
        Top XGBoost-Scored 3D Trends: {json.dumps(elite_titles[:8])}
        Mandatory Theme for this Video: "{theme_desc}"
        (FORBIDDEN: Do NOT repeat past titles: {json.dumps(past_topics[-35:])})

        Create a 5-step 3D scale comparison for "{theme_desc}".
        CRITICAL RULES:
        1. Order the 5 `items` strictly from SMALLEST `meters` (Rank #1) to LARGEST `meters` (Rank #5).
        2. Set `"model_type"` for each item accurately: `"dome_palace"`, `"pyramid"`, `"lattice_tower"`, `"planet"`, `"star"`, `"rocket"`, `"mountain"`, `"trench"`, `"crater"`, `"skyscraper"`, `"statue"`, or `"vehicle"`.
        3. Include `"wiki_query"` with the exact Wikipedia article title so we fetch its real-world photograph!
        4. Write 5 natural `narrations` ({word_rule} each) matching the 5 sorted items. NEVER say "Stage 1" or "Stage 2"!

        Output JSON ONLY:
        {{
          "title": "<Viral 3D Comparison Title under 58 chars>",
          "topic_summary": "<1-sentence summary>",
          "pinned_comment": "<Interactive question>",
          "thumb_badge": "<SHORT BADGE>",
          "thumb_line_1": "<WORD 1>",
          "thumb_line_2": "<WORD 2>",
          "items": [
            {{"name": "<Object 1>", "wiki_query": "<Exact Wikipedia Title>", "meters": 100, "metric": "<Formatted Metric>", "subtitle": "<2-3 WORD FACT>", "model_type": "{default_model}"}},
            {{"name": "<Object 2>", "wiki_query": "<Exact Wikipedia Title>", "meters": 500, "metric": "<Formatted Metric>", "subtitle": "<2-3 WORD FACT>", "model_type": "{default_model}"}},
            {{"name": "<Object 3>", "wiki_query": "<Exact Wikipedia Title>", "meters": 2500, "metric": "<Formatted Metric>", "subtitle": "<2-3 WORD FACT>", "model_type": "{default_model}"}},
            {{"name": "<Object 4>", "wiki_query": "<Exact Wikipedia Title>", "meters": 12000, "metric": "<Formatted Metric>", "subtitle": "<2-3 WORD FACT>", "model_type": "{default_model}"}},
            {{"name": "<Object 5>", "wiki_query": "<Exact Wikipedia Title>", "meters": 60000, "metric": "<Formatted Metric>", "subtitle": "<2-3 WORD FACT>", "model_type": "{default_model}"}}
          ],
          "narrations": [
            "<Natural narration for item 1>",
            "<Natural narration for item 2>",
            "<Natural narration for item 3>",
            "<Natural narration for item 4>",
            "<Natural narration for item 5>"
          ]
        }}
        """
        plan = extract_json(call_ai(SCRIPT_MODELS, "Output valid JSON only.", prompt, 0.78))
        items = plan.get("items", [])
        nars = [strip_stage_prefix(n) for n in plan.get("narrations", [])]
        if len(items) == 5 and len(nars) == 5:
            paired = sorted(zip(items, nars), key=lambda x: float(re.sub(r"[^\d.]", "", str(x[0].get("meters", 10))) or 10))
            plan["items"] = [p[0] for p in paired]
            plan["narrations"] = [p[1] for p in paired]
        else:
            plan["narrations"] = nars

        for it in plan.get("items", []):
            wq = it.get("wiki_query") or it.get("name", "")
            it["image_b64"] = fetch_wikipedia_image_b64(wq)

        plan["theme_seed"] = random.randint(0, 999)
        return plan

    else:
        fin_theme = WEALTH_THEME_WHEEL[run_idx % len(WEALTH_THEME_WHEEL)]
        prompt = f"""
        Top XGBoost-Scored Wealth & Investing Queries: {json.dumps(elite_titles[:8])}
        Mandatory Theme for this Video: "{fin_theme}"
        (FORBIDDEN: Do NOT repeat past titles: {json.dumps(past_topics[-35:])})

        Create a 5-part financial backtest comparing two assets or habits based on "{fin_theme}".
        CRITICAL RULES:
        1. `series_a` and `series_b` MUST each contain 6 numbers (`[initial_value, point_1, point_2, point_3, point_4, final_value]`).
        2. Include `"wiki_a"` and `"wiki_b"` with exact Wikipedia article titles so we display their real photos!
        3. Each of the 5 `narrations` ({word_rule} each) MUST state the exact dollar figures in `series_a[i+1]` and `series_b[i+1]`. NEVER say "Stage 1" or "Stage 2"!

        Output JSON ONLY:
        {{
          "title": "<Unique Viral Title under 58 chars>",
          "topic_summary": "<1-sentence summary>",
          "pinned_comment": "<Question asking viewers which side they would pick>",
          "thumb_badge": "<SHORT BADGE>",
          "thumb_line_1": "<ASSET A VS>",
          "thumb_line_2": "<ASSET B>",
          "asset_a_name": "<Name of Strategy/Asset A>",
          "asset_b_name": "<Name of Strategy/Asset B>",
          "wiki_a": "<Wikipedia Title A>",
          "wiki_b": "<Wikipedia Title B>",
          "start_year": 2010,
          "end_year": 2025,
          "series_a": [5000, 6200, 7800, 9500, 12400, 16800],
          "series_b": [5000, 9400, 6100, 22000, 54000, 118000],
          "narrations": [
            "<Natural narration 1>",
            "<Natural narration 2>",
            "<Natural narration 3>",
            "<Natural narration 4>",
            "<Natural narration 5>"
          ]
        }}
        """
        plan = extract_json(call_ai(SCRIPT_MODELS, "Output valid JSON only.", prompt, 0.78))
        plan["narrations"] = [strip_stage_prefix(n) for n in plan.get("narrations", [])]
        plan["img_a_b64"] = fetch_wikipedia_image_b64(plan.get("wiki_a") or plan.get("asset_a_name", ""))
        plan["img_b_b64"] = fetch_wikipedia_image_b64(plan.get("wiki_b") or plan.get("asset_b_name", ""))
        plan["theme_seed"] = random.randint(0, 999)
        return plan


# =====================================================================
# 7. AUDIO SYNTHESIS (4 MUSICAL SCALES/TIMBRES FOR PHYSICS + TTS)
# =====================================================================
def synthesize_collision_marimba_track(total_duration, collision_events, output_wav, seed_val=0):
    sr = 44100
    n_samples = int((total_duration + 0.3) * sr)
    audio = np.zeros(n_samples, dtype=np.float32)

    musical_scales = [
        [261.63, 329.63, 392.00, 493.88, 523.25, 659.25, 783.99, 987.77, 1046.50],
        [293.66, 349.23, 440.00, 523.25, 587.33, 698.46, 880.00, 1046.50, 1174.66],
        [220.00, 261.63, 329.63, 392.00, 440.00, 523.25, 659.25, 783.99, 880.00],
        [329.63, 349.23, 440.00, 493.88, 659.25, 698.46, 880.00, 987.77, 1318.51]
    ]
    scale_hz = musical_scales[seed_val % len(musical_scales)]

    t_all = np.linspace(0, total_duration + 0.3, n_samples, endpoint=False)
    root_hz = scale_hz[0] * 0.5
    audio += (np.sin(2 * np.pi * root_hz * t_all) * 0.015 + np.sin(2 * np.pi * (root_hz * 1.5) * t_all) * 0.011)

    for ev in collision_events:
        t_sec = float(ev.get("time", 0.0))
        intensity = float(ev.get("intensity", 0.8))
        n_idx = int(ev.get("noteIdx", 0)) % len(scale_hz)
        freq = scale_hz[n_idx]

        start_i = int(t_sec * sr)
        dur_s = 0.26
        c_len = int(dur_s * sr)
        if 0 <= start_i and start_i + c_len < n_samples:
            ct = np.linspace(0, dur_s, c_len, endpoint=False)
            env = np.exp(-ct * 15.0) * np.sin(np.pi * np.minimum(ct / 0.005, 1.0))
            tone = (
                np.sin(2 * np.pi * freq * ct) * 0.75 +
                np.sin(2 * np.pi * (freq * 2.0) * ct) * np.exp(-ct * 30.0) * 0.22 +
                np.sin(2 * np.pi * (freq * 3.0) * ct) * np.exp(-ct * 48.0) * 0.10
            )
            audio[start_i:start_i + c_len] += tone * env * (0.16 * max(intensity, 0.4))

    peak = np.max(np.abs(audio))
    if peak > 0.01:
        audio = (audio / peak) * 0.88

    pcm = (np.clip(audio, -0.95, 0.95) * 32767).astype(np.int16)
    with wave.open(output_wav, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def generate_ambient_pad_audio(total_duration, output_wav):
    sr = 44100
    n_samples = int((total_duration + 0.4) * sr)
    chord_freqs = [(110.0, 130.8, 164.8), (87.3, 110.0, 130.8), (130.8, 164.8, 196.0), (98.0, 123.5, 146.8)]
    audio = np.zeros(n_samples, dtype=np.float32)
    seg_len = int(6.0 * sr)

    for start_idx in range(0, n_samples, seg_len):
        end_idx = min(start_idx + seg_len, n_samples)
        cur_len = end_idx - start_idx
        ct = np.linspace(0, cur_len / sr, cur_len, endpoint=False)
        f1, f2, f3 = chord_freqs[(start_idx // seg_len) % len(chord_freqs)]
        env = np.sin(np.pi * ct / max(cur_len / sr, 0.1)) ** 0.7
        audio[start_idx:end_idx] += (np.sin(2 * np.pi * f1 * ct) * 0.015 + np.sin(2 * np.pi * f2 * ct) * 0.012 + np.sin(2 * np.pi * f3 * ct) * 0.009) * env

    pcm = (np.clip(audio, -0.95, 0.95) * 32767).astype(np.int16)
    with wave.open(output_wav, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def compute_weighted_word_timings(text, start_offset, clip_duration):
    raw_words = clean_text(text).split()
    if not raw_words:
        return []
    weights = [max(len(re.sub(r"[^a-zA-Z0-9]", "", w)), 2) + (3.0 if w.endswith((",", ";", ":")) else (4.2 if w.endswith((".", "!", "?")) else 0)) for w in raw_words]
    total_w = sum(weights) or 1.0
    effective_dur = max(clip_duration - 0.08, clip_duration * 0.96)
    timings, cursor = [], start_offset
    for w, weight in zip(raw_words, weights):
        w_dur = (weight / total_w) * effective_dur
        timings.append({"word": w, "start": cursor, "end": cursor + w_dur})
        cursor += w_dur
    return timings


async def fetch_tts_with_retries(narration):
    text_to_speak = clean_text(narration)
    if len(text_to_speak) < 3:
        text_to_speak = "Watch closely as the comparison unfolds."

    voices = [ACTIVE_POD["voice"] or "en-US-ChristopherNeural", "en-US-ChristopherNeural", "en-US-GuyNeural"]
    for attempt in range(5):
        voice = voices[attempt % len(voices)]
        try:
            try:
                comm = edge_tts.Communicate(text_to_speak, voice=voice, rate="+7%", boundary="WordBoundary")
            except TypeError:
                comm = edge_tts.Communicate(text_to_speak, voice=voice, rate="+7%")

            words, audio_bytes = [], bytearray()
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    audio_bytes.extend(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    w_clean = clean_text(chunk["text"])
                    if w_clean:
                        s = chunk["offset"] / 10_000_000.0
                        d = chunk["duration"] / 10_000_000.0
                        words.append({"word": w_clean, "rel_start": s, "rel_dur": max(d, 0.12)})

            if len(audio_bytes) > 500:
                return bytes(audio_bytes), words
        except Exception as e:
            print(f"[TTS Retry {attempt + 1}/5] {e}", flush=True)
            await asyncio.sleep(1.5 * (attempt + 1))

    return None, []


async def synthesize_chapter_audio(narrations, prefix="ch0"):
    scene_timings, all_word_timings, temp_wavs = [], [], []
    current_offset = 0.0

    for idx, raw_nar in enumerate(narrations):
        nar = strip_stage_prefix(raw_nar)
        mp3_path = f"{prefix}_sc_{idx}.mp3"
        wav_path = f"{prefix}_sc_{idx}.wav"

        audio_data, rel_words = await fetch_tts_with_retries(nar)
        await asyncio.sleep(0.2)

        if audio_data:
            with open(mp3_path, "wb") as f:
                f.write(audio_data)
            subprocess.run([
                "ffmpeg", "-y", "-i", mp3_path,
                "-af", "highpass=f=75,equalizer=f=180:width_type=o:width=2:g=2.5,acompressor=threshold=-16dB:ratio=2.5:attack=10:release=80",
                "-ar", "44100", "-ac", "1", wav_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "6.0", wav_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", wav_path]
        clip_dur = float(subprocess.check_output(cmd).decode().strip())

        if rel_words:
            scene_words = [
                {"word": rw["word"], "start": current_offset + rw["rel_start"], "end": current_offset + rw["rel_start"] + rw["rel_dur"]}
                for rw in rel_words
            ]
        else:
            scene_words = compute_weighted_word_timings(nar, current_offset, clip_dur)

        scene_timings.append({"scene_idx": idx, "start": current_offset, "end": current_offset + clip_dur, "duration": clip_dur})
        all_word_timings.extend(scene_words)
        temp_wavs.append(wav_path)
        current_offset += clip_dur
        if os.path.exists(mp3_path):
            os.remove(mp3_path)

    out_wav = f"{prefix}_voice.wav"
    concat_txt = f"{prefix}_concat.txt"
    with open(concat_txt, "w") as f:
        for w_file in temp_wavs:
            f.write(f"file '{w_file}'\n")

    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_txt, "-c", "copy", out_wav], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for w_file in temp_wavs:
        if os.path.exists(w_file):
            os.remove(w_file)
    if os.path.exists(concat_txt):
        os.remove(concat_txt)

    return current_offset, scene_timings, all_word_timings, out_wav


# =====================================================================
# 8. MULTI-ENGINE PLAYWRIGHT CHROMIUM RENDERER
# =====================================================================
def ensure_local_js_libs():
    os.makedirs("libs", exist_ok=True)
    three_path = Path("libs/three.min.js").resolve()
    matter_path = Path("libs/matter.min.js").resolve()

    if not three_path.exists():
        u = mk_url("https", ["cdnjs", "cloudflare", "com"], "/ajax/libs/three.js/r128/three.min.js")
        three_path.write_bytes(requests.get(u, headers=BROWSER_HEADERS, timeout=20).content)
    if not matter_path.exists():
        u = mk_url("https", ["cdnjs", "cloudflare", "com"], "/ajax/libs/matter-js/0.19.0/matter.min.js")
        matter_path.write_bytes(requests.get(u, headers=BROWSER_HEADERS, timeout=20).content)

    return f"file://{three_path}", f"file://{matter_path}"


def render_bespoke_pod_chapter(ch_idx, plan, is_widescreen, total_dur=36.0, scene_timings=None, word_timings=None, voice_wav=None):
    out_w = 1920 if is_widescreen else 1080
    out_h = 1080 if is_widescreen else 1920
    prefix = f"{POD_NAME.lower()}_ch_{ch_idx}"
    output_mp4 = f"{prefix}.mp4"
    three_src, matter_src = ensure_local_js_libs()

    if POD_NAME == "PHYSICS":
        total_dur = 36.0
        total_frames = int(total_dur * FPS)
        raw_mp4 = f"{prefix}_silent.mp4"

        payload = {
            "width": out_w,
            "height": out_h,
            "is_widescreen": is_widescreen,
            "total_duration": total_dur,
            "physics_mode": plan.get("physics_mode", 0),
            "team_a_name": plan.get("team_a_name", "TEAM A"),
            "team_b_name": plan.get("team_b_name", "TEAM B"),
            "img_a_b64": plan.get("img_a_b64", ""),
            "img_b_b64": plan.get("img_b_b64", ""),
            "hook_banner": plan.get("hook_banner", "WHO WILL ESCAPE FIRST?"),
            "palette_seed": plan.get("palette_seed", 0)
        }
        html_str = MATTERJS_PHYSICS_HTML.replace("__MATTER_JS_SRC__", matter_src).replace("__PAYLOAD_JSON__", json.dumps(payload))
        html_file = Path(f"{prefix}_matter.html").resolve()
        html_file.write_text(html_str, encoding="utf-8")

        ffmpeg_v = [
            "ffmpeg", "-y", "-f", "image2pipe", "-vcodec", "mjpeg", "-r", str(FPS), "-i", "-",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "19", "-pix_fmt", "yuv420p", raw_mp4
        ]
        proc = subprocess.Popen(ffmpeg_v, stdin=subprocess.PIPE)
        collision_events = []

        print(f"[KineticZen Matter.js V6.0] Simulating Photo-Avatar Rivalry {payload['team_a_name']} vs {payload['team_b_name']}...", flush=True)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-gpu"])
            page = browser.new_page(viewport={"width": out_w, "height": out_h})
            page.goto(f"file://{html_file}")
            page.evaluate("window.waitForAllImages && window.waitForAllImages()")
            page.wait_for_timeout(250)

            for f_idx in range(total_frames):
                page.evaluate(f"window.stepAndRenderPhysics({f_idx}, {total_frames})")
                proc.stdin.write(page.screenshot(type="jpeg", quality=92))
                if f_idx == total_frames // 3 and ch_idx == 0:
                    Path(f"thumb_{POD_NAME.lower()}.jpg").write_bytes(page.screenshot(type="jpeg", quality=95))

            collision_events = page.evaluate("window.collisionEvents || []")
            browser.close()

        proc.stdin.close()
        proc.wait()

        marimba_wav = f"{prefix}_marimba.wav"
        synthesize_collision_marimba_track(total_dur, collision_events, marimba_wav, seed_val=payload["palette_seed"])
        subprocess.run([
            "ffmpeg", "-y", "-i", raw_mp4, "-i", marimba_wav,
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", output_mp4
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        for tmp in [raw_mp4, marimba_wav, str(html_file)]:
            if os.path.exists(tmp):
                os.remove(tmp)
        return output_mp4, total_dur

    bgm_wav = f"{prefix}_bgm.wav"
    generate_ambient_pad_audio(total_dur, bgm_wav)

    mixed_audio = f"{prefix}_mixed.m4a"
    subprocess.run([
        "ffmpeg", "-y", "-i", voice_wav, "-i", bgm_wav,
        "-filter_complex", "[0:a]volume=1.55[v];[1:a]volume=0.55[b];[v][b]amix=inputs=2:duration=first:dropout_transition=2[aout]",
        "-map", "[aout]", "-c:a", "aac", "-b:a", "192k", mixed_audio
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    chunks = []
    w_per_chunk = 5 if is_widescreen else 3
    for st_info in scene_timings:
        s_start, s_end = st_info["start"], st_info["end"]
        s_words = [w for w in word_timings if s_start - 0.05 <= w["start"] < s_end + 0.05]
        for i in range(0, len(s_words), w_per_chunk):
            grp = s_words[i:i + w_per_chunk]
            if grp:
                chunks.append({"start": grp[0]["start"], "end": grp[-1]["end"], "words": grp})

    payload = {
        "width": out_w,
        "height": out_h,
        "is_widescreen": is_widescreen,
        "total_duration": total_dur,
        "scene_timings": scene_timings,
        "chunks": chunks,
        "theme_seed": plan.get("theme_seed", 0),
        "items": plan.get("items", []),
        "asset_a_name": plan.get("asset_a_name", "STRATEGY A"),
        "asset_b_name": plan.get("asset_b_name", "STRATEGY B"),
        "img_a_b64": plan.get("img_a_b64", ""),
        "img_b_b64": plan.get("img_b_b64", ""),
        "start_year": plan.get("start_year", 2010),
        "end_year": plan.get("end_year", 2025),
        "series_a": plan.get("series_a", [5000, 6200, 7800, 9500, 12400, 16800]),
        "series_b": plan.get("series_b", [5000, 9400, 6100, 22000, 54000, 118000])
    }

    if POD_NAME == "SCALE3D":
        html_str = (
            THREEJS_SCALE3D_HTML
            .replace("__THREE_JS_SRC__", three_src)
            .replace("__PAYLOAD_JSON__", json.dumps(payload))
        )
    else:
        html_str = TRADINGVIEW_WEALTH_HTML.replace("__PAYLOAD_JSON__", json.dumps(payload))

    html_file = Path(f"{prefix}_engine.html").resolve()
    html_file.write_text(html_str, encoding="utf-8")

    total_frames = int(total_dur * FPS)
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-f", "image2pipe", "-vcodec", "mjpeg", "-r", str(FPS), "-i", "-",
        "-i", mixed_audio, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "19",
        "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", output_mp4
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    print(f"[{POD_NAME} Bespoke Engine V6.0] Rendering {total_frames} frames ({out_w}x{out_h})...", flush=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist", "--no-sandbox"]
        )
        page = browser.new_page(viewport={"width": out_w, "height": out_h})
        page.goto(f"file://{html_file}")
        page.evaluate("window.waitForAllImages && window.waitForAllImages()")
        page.wait_for_timeout(450)

        for f_idx in range(total_frames):
            t = f_idx / FPS
            page.evaluate(f"window.renderFrameAtTime({t:.4f})")
            proc.stdin.write(page.screenshot(type="jpeg", quality=92))

        if ch_idx == 0:
            page.evaluate(f"window.renderFrameAtTime({total_dur * 0.65:.4f})")
            Path(f"thumb_{POD_NAME.lower()}.jpg").write_bytes(page.screenshot(type="jpeg", quality=95))

        browser.close()

    proc.stdin.close()
    proc.wait()

    for tmp in [voice_wav, bgm_wav, mixed_audio, str(html_file)]:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)

    return output_mp4, total_dur


# =====================================================================
# 9. INTERACTIVE MICRO-SAAS WEB PORTAL & YOUTUBE UPLOADER
# =====================================================================
def update_interactive_micro_saas_portal(video_id, title, summary):
    os.makedirs("docs", exist_ok=True)
    portal_json = Path("docs/videos.json")
    items = []
    if portal_json.exists():
        try:
            items = json.loads(portal_json.read_text(encoding="utf-8"))
        except Exception:
            items = []

    items.insert(0, {
        "pod": POD_NAME,
        "video_id": video_id,
        "title": title,
        "summary": summary,
        "timestamp": int(time.time())
    })
    items = items[:30]
    portal_json.write_text(json.dumps(items, indent=2), encoding="utf-8")

    yt_embed_base = mk_url("https", ["www", "youtube", "com"], "/embed/")
    cards_html = ""
    for it in items[:12]:
        cards_html += f"""
        <div class="card">
          <span class="badge">{it['pod']}</span>
          <iframe src="{yt_embed_base}{it['video_id']}" allowfullscreen></iframe>
          <h3>{it['title']}</h3>
          <p>{it['summary']}</p>
        </div>
        """

    html_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AETHER Interactive Wealth Calculator, 3D Scale & Physics Lab</title>
<style>
  body {{ background: #070b14; color: #f8fafc; font-family: system-ui, sans-serif; margin: 0; padding: 30px; }}
  h1, h2 {{ text-align: center; color: #38bdf8; }}
  .tools-wrap {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 24px; max-width: 1280px; margin: 25px auto; }}
  .tool-box {{ background: #111827; border: 1px solid #1e293b; border-radius: 18px; padding: 22px; }}
  input {{ width: 100%; padding: 10px; margin: 8px 0 14px; border-radius: 8px; border: 1px solid #334155; background: #0f172a; color: #fff; }}
  .res {{ font-size: 1.6rem; font-weight: 800; color: #10b981; margin-top: 10px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 24px; max-width: 1280px; margin: 30px auto; }}
  .card {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 16px; padding: 18px; }}
  .badge {{ background: #38bdf8; color: #070b14; font-weight: 800; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; }}
  iframe {{ width: 100%; height: 220px; border: none; border-radius: 10px; margin-top: 12px; }}
</style>
</head>
<body>
  <h1>AETHER INTERACTIVE SIMULATION & WEALTH TOOLS</h1>
  <div class="tools-wrap">
    <div class="tool-box">
      <h3>💰 Live Compound Interest & Wealth Simulator</h3>
      <label>Monthly Investment ($):</label>
      <input type="number" id="mInv" value="500" oninput="calcWealth()">
      <label>Annual Return (%) & Years:</label>
      <input type="number" id="rate" value="12" oninput="calcWealth()">
      <input type="number" id="yrs" value="25" oninput="calcWealth()">
      <div class="res" id="wealthOut">$948,817</div>
    </div>
    <div class="tool-box">
      <h3>🌍 3D Cosmic & Trench Scale Converter</h3>
      <label>Enter Height / Depth in Meters:</label>
      <input type="number" id="mScale" value="11000" oninput="calcScale()">
      <div class="res" id="scaleOut">13.3x Burj Khalifa | 1.24x Mt. Everest</div>
    </div>
  </div>
  <h2>LATEST AUTONOMOUS SIMULATIONS</h2>
  <div class="grid">{cards_html}</div>
<script>
function calcWealth() {{
  const p = Number(document.getElementById('mInv').value || 0);
  const r = Number(document.getElementById('rate').value || 0) / 100 / 12;
  const n = Number(document.getElementById('yrs').value || 0) * 12;
  const fv = r > 0 ? p * ((Math.pow(1 + r, n) - 1) / r) : p * n;
  document.getElementById('wealthOut').innerText = '$' + Math.round(fv).toLocaleString();
}}
function calcScale() {{
  const m = Number(document.getElementById('mScale').value || 0);
  document.getElementById('scaleOut').innerText = (m / 828).toFixed(1) + 'x Burj Khalifa | ' + (m / 8849).toFixed(2) + 'x Everest';
}}
</script>
</body>
</html>"""
    Path("docs/index.html").write_text(html_page, encoding="utf-8")


def format_srt_time(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def build_seo_srt_file(all_words, srt_path):
    lines, idx = [], 1
    for i in range(0, len(all_words), 6):
        grp = all_words[i:i + 6]
        if not grp:
            continue
        lines.append(f"{idx}\n{format_srt_time(grp[0]['start'])} --> {format_srt_time(grp[-1]['end'])}\n{' '.join(w['word'] for w in grp)}\n")
        idx += 1
    Path(srt_path).write_text("\n".join(lines), encoding="utf-8")
    return srt_path


def generate_pod_thumbnail(lead_plan, output_jpg):
    tw, th = 1280, 720
    base_file = f"thumb_{POD_NAME.lower()}.jpg"
    if os.path.exists(base_file):
        bg = Image.open(base_file).convert("RGB").resize((tw, th), Image.Resampling.LANCZOS)
    else:
        bg = Image.new("RGB", (tw, th), (8, 12, 24))

    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
    ]
    f_path = next((p for p in font_candidates if os.path.exists(p)), None)
    f_huge = ImageFont.truetype(f_path, 88) if f_path else ImageFont.load_default()
    f_badge = ImageFont.truetype(f_path, 36) if f_path else ImageFont.load_default()
    draw = ImageDraw.Draw(bg)

    badge = clean_display_text(lead_plan.get("thumb_badge", "WATCH UNTIL END!")).upper()[:18]
    l1 = clean_display_text(lead_plan.get("thumb_line_1", "INSANE")).upper()[:13]
    l2 = clean_display_text(lead_plan.get("thumb_line_2", "SIMULATION")).upper()[:13]

    bw = int(draw.textlength(badge, font=f_badge)) + 54
    draw.rounded_rectangle([55, 55, 55 + bw, 130], radius=16, fill=(235, 18, 40), outline=(255, 255, 255), width=4)
    draw.text((82, 68), badge, font=f_badge, fill=(255, 255, 255))
    draw.text((55, 185), l1, font=f_huge, fill=(255, 255, 255), stroke_width=12, stroke_fill=(0, 0, 0))
    draw.text((55, 310), l2, font=f_huge, fill=(255, 225, 0), stroke_width=12, stroke_fill=(0, 0, 0))
    draw.rectangle([8, 8, tw - 8, th - 8], outline=(255, 215, 0), width=10)
    bg.save(output_jpg, quality=95)
    return output_jpg


def seed_telegraph_backlink(video_id, title, description):
    yt_url = mk_url("https", ["www", "youtube", "com"], f"/watch?v={video_id}")
    telegraph_api = mk_url("https", ["api", "telegra", "ph"], "")
    try:
        acc_r = requests.get(f"{telegraph_api}/createAccount?short_name={POD_NAME}&author_name={POD_NAME}+Lab", timeout=10).json()
        token = acc_r.get("result", {}).get("access_token", "")
        if token:
            content_nodes = [
                {"tag": "p", "children": [description[:400]]},
                {"tag": "p", "children": [{"tag": "a", "attrs": {"href": yt_url}, "children": [f"Watch Live Simulation: {title}"]}]}
            ]
            page_r = requests.post(f"{telegraph_api}/createPage", data={
                "access_token": token,
                "title": title,
                "author_name": f"{POD_NAME} Lab",
                "content": json.dumps(content_nodes),
                "return_content": "false"
            }, timeout=10).json()
            page_url = page_r.get("result", {}).get("url", "")
            if page_url:
                print(f"✅ [{POD_NAME} Backlink] Seeded Telegraph Page: {page_url}", flush=True)
    except Exception:
        pass


def upload_pod_video(video_path, title, description, tags, pinned_comment, is_long, thumbnail_path, chapter_timestamps, srt_path):
    youtube = get_youtube_client()
    if not is_long and "#shorts" not in title.lower():
        title = f"{title[:88]} #shorts"
    elif is_long:
        title = title.replace("#shorts", "").replace("#Shorts", "").strip()

    slug_mp4 = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60] + ".mp4"
    seo_mp4 = slug_mp4 if slug_mp4 != ".mp4" else video_path
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path, "-c", "copy",
            "-metadata", f"title={title}",
            "-metadata", f"comment={description[:200]}",
            seo_mp4
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        seo_mp4 = video_path

    if is_long and chapter_timestamps:
        description = f"{description}\n\nTIMESTAMPS:\n" + "\n".join(chapter_timestamps)

    body = {
        "snippet": {
            "title": title[:100],
            "description": description,
            "tags": tags[:25],
            "categoryId": "28",
            "defaultLanguage": "en",
            "defaultAudioLanguage": "en"
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": False,
            "containsSyntheticMedia": False
        }
    }

    media = MediaFileUpload(seo_mp4, chunksize=-1, resumable=True, mimetype="video/mp4")
    response = youtube.videos().insert(part="snippet,status", body=body, media_body=media).execute()
    video_id = response["id"]
    short_link = mk_url("https", ["youtu", "be"], f"/{video_id}")
    print(f"🎉 [{POD_NAME} V6.0 PUBLISHED!] Video ID: {video_id} | URL: {short_link}", flush=True)

    mem = load_pod_memory()
    pub = mem.get("published_videos", [])
    pub.append({"video_id": video_id, "title": title, "published_at": int(time.time())})
    mem["published_videos"] = pub[-40:]
    save_pod_memory(mem)

    if is_long and thumbnail_path and os.path.exists(thumbnail_path):
        try:
            youtube.thumbnails().set(videoId=video_id, media_body=MediaFileUpload(thumbnail_path, mimetype="image/jpeg")).execute()
        except Exception:
            pass

    if srt_path and os.path.exists(srt_path):
        try:
            youtube.captions().insert(
                part="snippet",
                body={"snippet": {"videoId": video_id, "language": "en", "name": "English", "isDraft": False}},
                media_body=MediaFileUpload(srt_path, mimetype="application/octet-stream")
            ).execute()
        except Exception:
            pass

    try:
        time.sleep(3)
        youtube.commentThreads().insert(
            part="snippet",
            body={"snippet": {"videoId": video_id, "topLevelComment": {"snippet": {"textOriginal": pinned_comment}}}}
        ).execute()
    except Exception:
        pass

    seed_telegraph_backlink(video_id, title, description)
    update_interactive_micro_saas_portal(video_id, title, description[:160])


def main():
    is_long = (VIDEO_MODE == "long")
    mem = load_pod_memory()
    past_topics = mem.get("topics", [])
    spy_data = run_global_harvester_and_18_feature_xgboost()

    num_chapters = 5 if is_long else 1
    chapter_files, chapter_timestamps, master_words = [], [], []
    elapsed_sec = 0.0
    lead_plan = None

    for idx in range(num_chapters):
        plan = generate_pod_plan(spy_data, past_topics, unit_idx=idx, is_long=is_long)
        if idx == 0:
            lead_plan = plan
        past_topics.append(plan.get("title", ""))

        if POD_NAME == "PHYSICS":
            ch_mp4, dur = render_bespoke_pod_chapter(idx, plan, is_long)
        else:
            nars = plan.get("narrations", [])
            dur, s_timings, w_timings, v_wav = asyncio.run(synthesize_chapter_audio(nars, prefix=f"{POD_NAME.lower()}_ch_{idx}"))
            for w in w_timings:
                master_words.append({"word": w["word"], "start": elapsed_sec + w["start"], "end": elapsed_sec + w["end"]})
            ch_mp4, dur = render_bespoke_pod_chapter(idx, plan, is_long, dur, s_timings, w_timings, v_wav)

        mins, secs = int(elapsed_sec // 60), int(elapsed_sec % 60)
        chapter_timestamps.append(f"{mins:02d}:{secs:02d} - {clean_display_text(plan.get('topic_summary', f'Part {idx + 1}'))[:48]}")
        chapter_files.append(ch_mp4)
        elapsed_sec += dur

    mem["topics"] = past_topics[-100:]
    save_pod_memory(mem)

    if len(chapter_files) == 1:
        final_mp4 = chapter_files[0]
    else:
        concat_file = f"concat_{POD_NAME.lower()}.txt"
        with open(concat_file, "w") as f:
            for cf in chapter_files:
                f.write(f"file '{cf}'\n")
        final_mp4 = f"final_{POD_NAME.lower()}.mp4"
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file, "-c", "copy", final_mp4], check=True)

    srt_file = build_seo_srt_file(master_words, f"captions_{POD_NAME.lower()}.srt") if master_words else None
    thumb_jpg = generate_pod_thumbnail(lead_plan, f"final_thumb_{POD_NAME.lower()}.jpg") if is_long else None
    tags = list(dict.fromkeys(spy_data["hijacked_tags"] + ["simulation", "3d", "physics", "wealth", POD_NAME.lower(), "shorts"]))

    upload_pod_video(
        final_mp4,
        title=lead_plan.get("title", "Live Simulation Experiment"),
        description=f"{lead_plan.get('topic_summary', '')}\n\nSubscribe for daily simulations!",
        tags=tags,
        pinned_comment=lead_plan.get("pinned_comment", "What did you think of this simulation? Comment below!"),
        is_long=is_long,
        thumbnail_path=thumb_jpg,
        chapter_timestamps=chapter_timestamps,
        srt_path=srt_file
    )


if __name__ == "__main__":
    main()
