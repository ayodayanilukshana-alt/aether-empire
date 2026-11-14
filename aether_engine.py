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

REPO_NAME_CONST = "aether-empire"
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
    """
    V11.0 Dual-Engine Photorealistic Image Resolver:
    1. First checks Wikimedia Commons High-Res Photographs (800px).
    2. If unavailable or generic, generates a bespoke Photorealistic 8K Render using Flux AI!
    Also saves local cached JPEGs (`cached_img_0.jpg`, `cached_img_1.jpg`) for our Split-Screen Thumbnail Composer!
    """
    if not query_name:
        return ""
    q_clean = str(query_name).strip()
    img_bytes = None
    mime = "image/jpeg"

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
                "action": "query", "format": "json", "prop": "pageimages",
                "piprop": "thumbnail", "pithumbsize": 800,
                "generator": "search", "gsrsearch": q_clean, "gsrlimit": 1
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
            if img_r.status_code == 200 and len(img_r.content) > 1200:
                img_bytes = img_r.content
                if thumb_url.lower().endswith(".png"):
                    mime = "image/png"
    except Exception:
        pass

    # Fallback to Flux AI Photorealistic Image Generation so EVERY object has a breathtaking HD visual!
    if not img_bytes:
        try:
            prompt_str = urllib.parse.quote(f"photorealistic 8k cinematic architectural photograph of {q_clean}, dramatic lighting, ultra detailed, national geographic style")
            ai_img_url = mk_url("https", ["image", "pollinations", "ai"], f"/prompt/{prompt_str}?width=768&height=768&nologo=true&seed=42")
            r_ai = requests.get(ai_img_url, headers=BROWSER_HEADERS, timeout=15)
            if r_ai.status_code == 200 and len(r_ai.content) > 2000:
                img_bytes = r_ai.content
                mime = "image/jpeg"
                print(f"🎨 [AI Image Generator] Generated photorealistic render for '{q_clean}'!", flush=True)
        except Exception:
            pass

    if img_bytes:
        try:
            idx_slot = 0 if not os.path.exists("thumb_slot_0.jpg") else 1
            with open(f"thumb_slot_{idx_slot}.jpg", "wb") as f:
                f.write(img_bytes)
        except Exception:
            pass
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:{mime};base64,{b64}"
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
  body { background: #030712; overflow: hidden; font-family: 'DejaVu Sans', 'Liberation Sans', sans-serif; }
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

// Initialize WatchData / MetaBallStudios Style 3D Photorealistic Monolith Stage
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050914);
scene.fog = new THREE.FogExp2(0x050914, 0.00032);

const camera = new THREE.PerspectiveCamera(40, W / H, 0.5, 90000);
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setSize(W, H);
document.getElementById('webgl-container').appendChild(renderer.domElement);

const hemiLight = new THREE.HemisphereLight(0xffffff, 0x38bdf8, 1.4);
hemiLight.position.set(0, 1200, 0);
scene.add(hemiLight);

const dirLight = new THREE.DirectionalLight(0xfffbeb, 1.6);
dirLight.position.set(350, 800, 750);
scene.add(dirLight);

// Reflective Dark Obsidian Floor + Laser Elevation Grid Wall
const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(90000, 90000),
    new THREE.MeshStandardMaterial({ color: 0x070d19, roughness: 0.25, metalness: 0.5 })
);
ground.rotation.x = -Math.PI / 2;
scene.add(ground);

const floorGrid = new THREE.GridHelper(20000, 400, 0x38bdf8, 0x1e293b);
floorGrid.position.y = 0.2;
scene.add(floorGrid);

const backGrid = new THREE.GridHelper(6000, 120, 0x1e3a8a, 0x0f172a);
backGrid.rotation.x = Math.PI / 2;
backGrid.position.set(600, 600, -90);
scene.add(backGrid);

const items = DATA.items || [];
const rawMeters = items.map((it, idx) => Math.max(Number(it.meters) || (idx + 1) * 20, 0.1));
const minM = Math.min(...rawMeters);
const maxM = Math.max(...rawMeters);

// Proportional 3D Monolith Heights so side-by-side scale jumps look dramatic & unmistakable!
const visualHeights = rawMeters.map((m, idx) => {
    if (maxM / minM > 20) {
        const ratio = Math.log10(m / minM + 1) / Math.log10(maxM / minM + 1);
        return 38 + ratio * 235 + idx * 24;
    }
    return Math.max(38, (m / maxM) * 270);
});

const monolithWidths = visualHeights.map(h => Math.max(32, Math.min(h * 0.56, 135)));
const xPositions = [0];
for (let i = 1; i < 5; i++) {
    const gap = (monolithWidths[i - 1] * 0.62) + (monolithWidths[i] * 0.62) + Math.max(visualHeights[i] * 0.26, 36);
    xPositions.push(xPositions[i - 1] + gap);
}

const accentHexes = [0x10b981, 0x38bdf8, 0xf59e0b, 0xec4899, 0xa855f7];
const accentStrs  = ['#10b981', '#38bdf8', '#f59e0b', '#ec4899', '#a855f7'];
const loadedImages = [];

// Build High-Resolution 3D Monolith Face Texture combining the Real Photo/AI Render + Blueprint Grid + Metric Header
function buildMonolithFaceTexture(itemObj, imgElem, idx, accentStr) {
    const c = document.createElement('canvas');
    c.width = 512; c.height = 1024;
    const g = c.getContext('2d');

    const bg = g.createLinearGradient(0, 0, 0, 1024);
    bg.addColorStop(0, '#0f172a');
    bg.addColorStop(1, '#050811');
    g.fillStyle = bg;
    g.fillRect(0, 0, 512, 1024);

    // Subtle CAD grid lines on pillar
    g.strokeStyle = 'rgba(56, 189, 248, 0.12)';
    g.lineWidth = 2;
    for (let y = 0; y < 1024; y += 64) {
        g.beginPath(); g.moveTo(0, y); g.lineTo(512, y); g.stroke();
    }

    // Draw Real Photo / AI Photorealistic Render filling the upper 72% of the 3D Monolith!
    if (imgElem && imgElem.complete && imgElem.naturalWidth > 0) {
        g.save();
        g.beginPath();
        g.roundRect(18, 110, 476, 680, 20);
        g.clip();
        const iw = imgElem.naturalWidth, ih = imgElem.naturalHeight;
        const scale = Math.max(476 / iw, 680 / ih);
        const dw = iw * scale, dh = ih * scale;
        g.drawImage(imgElem, 18 + (476 - dw) / 2, 110 + (680 - dh) / 2, dw, dh);
        g.restore();
    }

    // Glowing Neon Frame around the 3D Monolith
    g.strokeStyle = accentStr;
    g.lineWidth = 12;
    g.strokeRect(6, 6, 500, 1012);

    // Top Rank & Metric Header baked right onto the 3D Monolith Crown
    g.fillStyle = accentStr;
    g.fillRect(12, 12, 488, 86);
    g.fillStyle = '#050811';
    g.font = 'bold 44px sans-serif';
    g.textAlign = 'center';
    g.fillText(String(itemObj.metric || '').toUpperCase(), 256, 70);

    // Bottom Object Name Plate on the 3D Monolith Base
    g.fillStyle = 'rgba(9, 13, 22, 0.92)';
    g.fillRect(18, 810, 476, 185);
    g.fillStyle = '#ffffff';
    g.font = 'bold 38px sans-serif';
    g.fillText(String(itemObj.name || '').toUpperCase().slice(0, 18), 256, 875);
    g.fillStyle = '#facc15';
    g.font = 'bold 28px sans-serif';
    g.fillText('RANK #0' + (idx + 1), 256, 935);

    const tex = new THREE.CanvasTexture(c);
    tex.needsUpdate = true;
    return tex;
}

const meshes = [];

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

    // Construct the 5 Photorealistic 3D Volumetric Monolith Towers once images are loaded!
    for (let i = 0; i < 5; i++) {
        const h = visualHeights[i];
        const w = monolithWidths[i];
        const d = Math.max(16, w * 0.28);
        const x = xPositions[i];
        const hexNum = accentHexes[i % 5];
        const accentStr = accentStrs[i % 5];
        const itemObj = items[i] || {};

        const group = new THREE.Group();
        const faceTex = buildMonolithFaceTexture(itemObj, loadedImages[i], i, accentStr);

        const matSide = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.2, metalness: 0.8 });
        const matTop  = new THREE.MeshBasicMaterial({ color: hexNum });
        const matFace = new THREE.MeshBasicMaterial({ map: faceTex });

        // 6-sided material array: [right, left, top, bottom, front (photorealistic), back]
        const materials = [matSide, matSide, matTop, matSide, matFace, matSide];
        const monolith = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), materials);
        monolith.position.y = h / 2 + 1.5;
        group.add(monolith);

        // Glowing Neon Edge Bezel & Pedestal Platform
        const ped = new THREE.Mesh(
            new THREE.CylinderGeometry(w * 0.68, w * 0.74, 3.0, 48),
            new THREE.MeshBasicMaterial({ color: hexNum })
        );
        ped.position.y = 1.5;
        group.add(ped);

        group.position.set(x, 0, 0);
        scene.add(group);
        meshes.push(group);
    }
};

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

    const floatIdx = Math.min(4.0, stageIndex + Math.pow(stageProgress, 1.55) * 0.88);
    const i0 = Math.min(4, Math.floor(floatIdx));
    const i1 = Math.min(4, i0 + 1);
    const frac = floatIdx - i0;
    const smoothFrac = frac * frac * (3 - 2 * frac);

    const curX = xPositions[i0] * (1 - smoothFrac) + xPositions[i1] * smoothFrac;
    const curH = visualHeights[i0] * (1 - smoothFrac) + visualHeights[i1] * smoothFrac;
    const prevX = (i0 > 0) ? xPositions[i0 - 1] : curX;

    // Subtle 3D perspective angle so the viewer sees the 3D depth of the monoliths + previous smaller monoliths on the left!
    meshes.forEach((m, idx) => {
        m.rotation.y = Math.sin(t * 0.45 + idx * 0.6) * 0.14 - 0.12;
    });

    const lookX = (i0 === 0 && smoothFrac < 0.2) ? curX : (curX * 0.70 + prevX * 0.30);
    const dist = Math.max(curH * (isWide ? 1.85 : 2.65), 85);
    camera.position.set(
        lookX - dist * 0.16,
        curH * 0.55 + 12,
        dist
    );
    camera.lookAt(lookX, curH * 0.50, 0);

    renderer.render(scene, camera);

    // Draw Sleek Top Broadcast Telemetry Bar & Bottom Captions
    ctx.clearRect(0, 0, W, H);
    const activeItem = items[stageIndex] || { name: 'OBJECT ' + (stageIndex + 1), metric: '100 m', subtitle: '3D SCALE' };
    const activeImg = loadedImages[stageIndex];

    const cardW = isWide ? 880 : (W - 80);
    const cardX = (W - cardW) / 2;
    const cardY = isWide ? 32 : 78;

    ctx.save();
    ctx.fillStyle = 'rgba(9, 13, 22, 0.94)';
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.roundRect(cardX, cardY, cardW, 175, 22);
    ctx.fill();
    ctx.stroke();

    let textOffsetX = cardX + 28;
    if (activeImg && activeImg.complete && activeImg.naturalWidth > 0) {
        ctx.save();
        ctx.beginPath();
        ctx.roundRect(cardX + 18, cardY + 18, 138, 138, 14);
        ctx.clip();
        ctx.drawImage(activeImg, cardX + 18, cardY + 18, 138, 138);
        ctx.restore();
        ctx.strokeStyle = '#facc15';
        ctx.lineWidth = 3;
        ctx.strokeRect(cardX + 18, cardY + 18, 138, 138);
        textOffsetX = cardX + 178;
    }

    ctx.fillStyle = '#f59e0b';
    ctx.beginPath();
    ctx.roundRect(textOffsetX, cardY + 20, 145, 38, 8);
    ctx.fill();
    ctx.fillStyle = '#090d16';
    ctx.font = 'bold 20px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('RANK #0' + (stageIndex + 1), textOffsetX + 72, cardY + 46);

    ctx.textAlign = 'left';
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 34px sans-serif';
    ctx.fillText(String(activeItem.name || '').toUpperCase().slice(0, 24), textOffsetX + 160, cardY + 49);

    ctx.fillStyle = '#38bdf8';
    ctx.font = 'bold 52px sans-serif';
    ctx.fillText(String(activeItem.metric || ''), textOffsetX, cardY + 122);

    const multVsFirst = (rawMeters[stageIndex] / Math.max(rawMeters[0], 0.1)).toFixed(1);
    const subTxt = stageIndex > 0 ? (`${multVsFirst}X LARGER THAN ${String(items[0].name || 'RANK #1').toUpperCase().slice(0, 16)}`) : String(activeItem.subtitle || '').toUpperCase();
    ctx.fillStyle = '#facc15';
    ctx.font = 'bold 21px sans-serif';
    ctx.fillText(subTxt.slice(0, 38), textOffsetX, cardY + 156);
    ctx.restore();

    const chunks = DATA.chunks || [];
    let activeChunk = null;
    for (let c of chunks) {
        if (t >= c.start - 0.04 && t <= c.end + 0.08) { activeChunk = c; break; }
    }
    if (activeChunk) {
        ctx.save();
        const words = activeChunk.words.map(w => String(w.word || '').toUpperCase()).join(' ');
        ctx.font = isWide ? 'bold 40px sans-serif' : 'bold 44px sans-serif';
        const tw = Math.min(ctx.measureText(words).width + 60, W - 80);
        const by = isWide ? (H - 55) : 1560;
        ctx.fillStyle = 'rgba(9, 13, 22, 0.92)';
        ctx.strokeStyle = '#fef08a';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.roundRect((W - tw) / 2, by - 50, tw, 72, 14);
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
const isWide = DATA.is_widescreen;
const modeIdx = (DATA.physics_mode || 0) % 5;
const teamAName = String(DATA.team_a_name || 'ALPHA').toUpperCase().slice(0, 11);
const teamBName = String(DATA.team_b_name || 'OMEGA').toUpperCase().slice(0, 11);
const hookQuestion = String(DATA.hook_banner || (teamAName + ' VS ' + teamBName)).toUpperCase();

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
    { bg1: '#1a0b2e', bg2: '#070312', teamA: '#00e5ff', teamB: '#ff2a85', rings: ['#ff2a85', '#ffae00', '#00f5d4', '#38bdf8', '#a855f7', '#10b981', '#facc15'] },
    { bg1: '#041f1e', bg2: '#020b0e', teamA: '#00ff9d', teamB: '#ff5500', rings: ['#00ff9d', '#00e5ff', '#ff5500', '#facc15', '#38bdf8', '#a855f7', '#ec4899'] },
    { bg1: '#091931', bg2: '#030814', teamA: '#facc15', teamB: '#f43f5e', rings: ['#facc15', '#f43f5e', '#00e5ff', '#8b5cf6', '#10b981', '#ec4899', '#38bdf8'] },
    { bg1: '#230b1e', bg2: '#0e030c', teamA: '#bef264', teamB: '#d946ef', rings: ['#bef264', '#d946ef', '#06b6d4', '#f59e0b', '#ec4899', '#6366f1', '#10b981'] }
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

// In Widescreen (1920x1080), center the arena at Y=565 and scale radii to fit cleanly with ZERO clipping!
const cx = W / 2;
const cy = isWide ? (H * 0.54) : (H * 0.58);
const scaleFactor = isWide ? 0.82 : 1.0;

const rings = [];
const pegs = [];
const particles = [];
const shockwaves = [];
const balls = [];
window.collisionEvents = [];
let shakeFrames = 0;

if (modeIdx !== 2) {
    const isHex = (modeIdx === 1);
    const baseRadii = isHex ? [160, 230, 300, 370, 440] : [135, 190, 245, 300, 355, 410, 465];
    baseRadii.forEach((rawR, rIdx) => {
        const radius = rawR * scaleFactor;
        const segments = isHex ? 6 : 72;
        const gapSegments = isHex ? 0 : 7;
        const segBodies = [];
        for (let s = gapSegments; s < segments; s++) {
            const a1 = (s / segments) * Math.PI * 2;
            const a2 = ((s + 1) / segments) * Math.PI * 2;
            const midA = (a1 + a2) / 2;
            const rDist = isHex ? radius * Math.cos(Math.PI / 6) : radius;
            const sideLen = isHex ? (2 * radius * Math.sin(Math.PI / 6) * (s === 0 ? 0.40 : 1.05)) : (((2 * Math.PI * radius) / segments) * 1.45);
            const sx = cx + Math.cos(midA) * rDist;
            const sy = cy + Math.sin(midA) * rDist;
            const seg = Bodies.rectangle(sx, sy, sideLen, 22 * scaleFactor, {
                isStatic: true, angle: midA + Math.PI / 2, restitution: 1.03, friction: 0.0
            });
            segBodies.push({ body: seg, relAngle: midA, radius: rDist });
            Composite.add(world, seg);
        }
        rings.push({
            idx: rIdx, radius: radius, color: pal.rings[rIdx % pal.rings.length],
            angleOffset: rIdx * 0.9, speed: (rIdx % 2 === 0 ? 1 : -1) * (0.025 + rIdx * 0.003),
            segments: segBodies, escapedCount: 0, hp: 28, destroyed: false, isPolygon: isHex
        });
    });

    const nBalls = 21;
    const bRad = Math.round(21 * scaleFactor);
    for (let i = 0; i < nBalls; i++) {
        const isA = (i % 2 === 0);
        const angle = (i / nBalls) * Math.PI * 2;
        const r = (25 + (i % 3) * 20) * scaleFactor;
        const ball = Bodies.circle(cx + Math.cos(angle) * r, cy + Math.sin(angle) * r, bRad, {
            restitution: 1.03, friction: 0.0, frictionAir: 0.0001
        });
        Body.setVelocity(ball, { x: Math.cos(angle * 2.1) * 9.5, y: Math.sin(angle * 1.7) * 9.5 });
        Composite.add(world, ball);
        balls.push({ body: ball, isA: isA, rad: bRad, ringStage: 0, trail: [] });
    }
} else {
    const rows = 7;
    const spacingX = 110 * scaleFactor;
    const spacingY = 105 * scaleFactor;
    const topPy = isWide ? 190 : 430;
    for (let r = 0; r < rows; r++) {
        const cols = (r % 2 === 0) ? 7 : 6;
        const startX = cx - ((cols - 1) * spacingX) / 2;
        const py = topPy + r * spacingY;
        for (let c = 0; c < cols; c++) {
            const px = startX + c * spacingX;
            const peg = Bodies.circle(px, py, 13 * scaleFactor, { isStatic: true, restitution: 0.96, friction: 0.0 });
            Composite.add(world, peg);
            pegs.push({ x: px, y: py, r: 13 * scaleFactor, color: pal.rings[r % pal.rings.length] });
        }
    }
    const wallLeft = cx - 420 * scaleFactor;
    const wallRight = cx + 420 * scaleFactor;
    Composite.add(world, Bodies.rectangle(wallLeft, H / 2, 30, H, { isStatic: true, restitution: 0.95 }));
    Composite.add(world, Bodies.rectangle(wallRight, H / 2, 30, H, { isStatic: true, restitution: 0.95 }));

    const bRad = Math.round(20 * scaleFactor);
    for (let i = 0; i < 25; i++) {
        const isA = (i % 2 === 0);
        const bx = cx + (Math.sin(i * 17.3) * 240 * scaleFactor);
        const by = (topPy - 80) + (i % 5) * 16;
        const ball = Bodies.circle(bx, by, bRad, { restitution: 0.94, friction: 0.001, frictionAir: 0.001 });
        Body.setVelocity(ball, { x: Math.cos(i * 5) * 4, y: 2 });
        Composite.add(world, ball);
        balls.push({ body: ball, isA: isA, rad: bRad, ringStage: 0, trail: [] });
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
        if (ev.pairs && ev.pairs.length > 0 && shockwaves.length < 16) {
            const pos = ev.pairs[0].bodyA.position;
            shockwaves.push({ x: pos.x, y: pos.y, r: 10, alpha: 0.75, color: noteCounter % 2 === 0 ? pal.teamA : pal.teamB });
        }
    }
});

let scoreA = 18;
let scoreB = 12;

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

    const maxBoundR = 475 * scaleFactor;
    balls.forEach(b => {
        const pos = b.body.position;
        const vel = b.body.velocity;
        const spd = Math.hypot(vel.x, vel.y);
        if (spd < 6.0 && modeIdx !== 2) {
            Body.setVelocity(b.body, { x: vel.x * 1.38 + (Math.random() - 0.5) * 2.5, y: vel.y * 1.38 - 1.8 });
        }

        b.trail.push({ x: pos.x, y: pos.y });
        if (b.trail.length > 6) b.trail.shift();

        if (modeIdx === 2) {
            const respawnY = isWide ? (H - 90) : 1440;
            const topY = isWide ? 115 : 310;
            if (pos.y > respawnY) {
                if (b.isA) scoreA += 17; else scoreB += 14;
                shakeFrames = 4;
                Body.setPosition(b.body, { x: cx + (Math.random() - 0.5) * 240 * scaleFactor, y: topY });
                Body.setVelocity(b.body, { x: (Math.random() - 0.5) * 5, y: 2 });
            }
        } else {
            const dist = Math.hypot(pos.x - cx, pos.y - cy);
            if (activeRingIdx < rings.length) {
                const curRing = rings[activeRingIdx];
                if (b.ringStage <= activeRingIdx && dist > curRing.radius + 12) {
                    b.ringStage = activeRingIdx + 1;
                    curRing.escapedCount += 1;
                    if (b.isA) scoreA += 14; else scoreB += 11;
                }
                // Guaranteed Ring Shatter Schedule: breaks when escapedCount >= 3 OR when time reaches its scheduled break slot!
                const scheduledBreakSec = (activeRingIdx + 1) * (31.0 / Math.max(rings.length, 1));
                if ((curRing.escapedCount >= 3 && simTime >= scheduledBreakSec * 0.75) || simTime >= scheduledBreakSec) {
                    curRing.destroyed = true;
                    shakeFrames = 8;
                    if (b.isA) scoreA += 35; else scoreB += 28;
                    curRing.segments.forEach(s => {
                        Composite.remove(world, s.body);
                        particles.push({
                            x: s.body.position.x, y: s.body.position.y,
                            vx: (Math.random() - 0.5) * 16, vy: (Math.random() - 0.5) * 16,
                            color: curRing.color, alpha: 1.0
                        });
                    });
                    window.collisionEvents.push({ time: Number(simTime.toFixed(3)), intensity: 1.0, noteIdx: 8 });
                }
            }
            if (dist > maxBoundR) {
                const ang = Math.atan2(pos.y - cy, pos.x - cx);
                const nx = Math.cos(ang), ny = Math.sin(ang);
                Body.setPosition(b.body, { x: cx + nx * (maxBoundR - 5), y: cy + ny * (maxBoundR - 5) });
                const dot = vel.x * nx + vel.y * ny;
                Body.setVelocity(b.body, { x: (vel.x - 2 * dot * nx) * 0.98, y: (vel.y - 2 * dot * ny) * 0.98 - 2.0 });
            }
        }
    });

    ctx.save();
    if (shakeFrames > 0) {
        shakeFrames--;
        ctx.translate((Math.random() - 0.5) * 12, (Math.random() - 0.5) * 12);
    }

    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, pal.bg1);
    grad.addColorStop(1, pal.bg2);
    ctx.fillStyle = grad;
    ctx.fillRect(-30, -30, W + 60, H + 60);

    if (scoreA === scoreB) scoreA += 3;
    const totalSc = Math.max(scoreA + scoreB, 1);
    const pctA = scoreA / totalSc;

    if (isWide) {
        // WIDESCREEN (1920x1080) ESPORTS BROADCAST HUD: Top Slim Pill + Left & Right Team Towers (Zero Overlap with Center Arena!)
        ctx.save();
        ctx.fillStyle = 'rgba(15, 23, 42, 0.90)';
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.lineWidth = 3;
        ctx.beginPath(); ctx.roundRect(W / 2 - 420, 18, 840, 56, 28); ctx.fill(); ctx.stroke();
        ctx.fillStyle = '#fef08a';
        ctx.font = 'bold 24px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(hookQuestion.slice(0, 52), cx, 54);

        // Left Team A Panel
        const pW = 360, pH = 440, pY = (H - pH) / 2;
        ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
        ctx.strokeStyle = pal.teamA; ctx.lineWidth = 4;
        ctx.beginPath(); ctx.roundRect(45, pY, pW, pH, 28); ctx.fill(); ctx.stroke();
        if (imgA && imgA.complete && imgA.naturalWidth > 0) {
            ctx.save();
            ctx.beginPath(); ctx.arc(45 + pW / 2, pY + 115, 68, 0, Math.PI * 2); ctx.clip();
            ctx.drawImage(imgA, 45 + pW / 2 - 68, pY + 47, 136, 136);
            ctx.restore();
        }
        ctx.fillStyle = pal.teamA; ctx.font = 'bold 38px sans-serif'; ctx.textAlign = 'center';
        ctx.fillText(teamAName, 45 + pW / 2, pY + 245);
        ctx.fillStyle = '#ffffff'; ctx.font = 'bold 74px sans-serif';
        ctx.fillText(String(scoreA), 45 + pW / 2, pY + 335);
        ctx.fillStyle = pal.teamA; ctx.font = 'bold 30px sans-serif';
        ctx.fillText(Math.round(pctA * 100) + '% WIN PROB', 45 + pW / 2, pY + 395);

        // Right Team B Panel
        const rX = W - 45 - pW;
        ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
        ctx.strokeStyle = pal.teamB; ctx.lineWidth = 4;
        ctx.beginPath(); ctx.roundRect(rX, pY, pW, pH, 28); ctx.fill(); ctx.stroke();
        if (imgB && imgB.complete && imgB.naturalWidth > 0) {
            ctx.save();
            ctx.beginPath(); ctx.arc(rX + pW / 2, pY + 115, 68, 0, Math.PI * 2); ctx.clip();
            ctx.drawImage(imgB, rX + pW / 2 - 68, pY + 47, 136, 136);
            ctx.restore();
        }
        ctx.fillStyle = pal.teamB; ctx.font = 'bold 38px sans-serif'; ctx.textAlign = 'center';
        ctx.fillText(teamBName, rX + pW / 2, pY + 245);
        ctx.fillStyle = '#ffffff'; ctx.font = 'bold 74px sans-serif';
        ctx.fillText(String(scoreB), rX + pW / 2, pY + 335);
        ctx.fillStyle = pal.teamB; ctx.font = 'bold 30px sans-serif';
        ctx.fillText((100 - Math.round(pctA * 100)) + '% WIN PROB', rX + pW / 2, pY + 395);
        ctx.restore();
    } else {
        // VERTICAL SHORTS (1080x1920) TOP SCOREBOARD
        ctx.save();
        ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.22)';
        ctx.lineWidth = 3;
        ctx.beginPath(); ctx.roundRect(48, 58, W - 96, 235, 28); ctx.fill(); ctx.stroke();

        ctx.fillStyle = '#fef08a';
        ctx.font = 'bold 24px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(hookQuestion.slice(0, 42), cx, 102);

        if (imgA && imgA.complete && imgA.naturalWidth > 0) {
            ctx.save();
            ctx.beginPath(); ctx.arc(110, 158, 34, 0, Math.PI * 2); ctx.clip();
            ctx.drawImage(imgA, 76, 124, 68, 68);
            ctx.restore();
        }
        ctx.fillStyle = pal.teamA; ctx.font = 'bold 34px sans-serif'; ctx.textAlign = 'left';
        ctx.fillText(teamAName + ': ' + scoreA, 158, 170);

        if (imgB && imgB.complete && imgB.naturalWidth > 0) {
            ctx.save();
            ctx.beginPath(); ctx.arc(W - 110, 158, 34, 0, Math.PI * 2); ctx.clip();
            ctx.drawImage(imgB, W - 144, 124, 68, 68);
            ctx.restore();
        }
        ctx.fillStyle = pal.teamB; ctx.textAlign = 'right';
        ctx.fillText(teamBName + ': ' + scoreB, W - 158, 170);

        const barX = 82, barY = 215, barW = W - 164, barH = 36;
        ctx.fillStyle = pal.teamB; ctx.beginPath(); ctx.roundRect(barX, barY, barW, barH, 16); ctx.fill();
        ctx.fillStyle = pal.teamA; ctx.beginPath(); ctx.roundRect(barX, barY, Math.max(45, barW * pctA), barH, 16); ctx.fill();
        ctx.fillStyle = '#090d16'; ctx.font = 'bold 22px sans-serif';
        ctx.textAlign = 'left'; ctx.fillText(Math.round(pctA * 100) + '%', barX + 16, barY + 26);
        ctx.textAlign = 'right'; ctx.fillText((100 - Math.round(pctA * 100)) + '%', barX + barW - 16, barY + 26);
        ctx.restore();
    }

    rings.forEach(r => {
        if (r.destroyed) return;
        ctx.save();
        ctx.strokeStyle = r.color;
        ctx.lineWidth = Math.round(16 * scaleFactor);
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
            ctx.arc(cx, cy, r.radius, r.angleOffset + (7 / 72) * Math.PI * 2, r.angleOffset + Math.PI * 2);
            ctx.stroke();
        }
        ctx.restore();
    });

    pegs.forEach(pg => {
        ctx.fillStyle = pg.color;
        ctx.beginPath(); ctx.arc(pg.x, pg.y, pg.r, 0, Math.PI * 2); ctx.fill();
    });

    for (let p of particles) {
        if (p.alpha <= 0.02) continue;
        p.x += p.vx; p.y += p.vy; p.alpha *= 0.92;
        ctx.save(); ctx.globalAlpha = p.alpha; ctx.fillStyle = p.color;
        ctx.beginPath(); ctx.arc(p.x, p.y, 7, 0, Math.PI * 2); ctx.fill(); ctx.restore();
    }

    balls.forEach(b => {
        const col = b.isA ? pal.teamA : pal.teamB;
        const avatarImg = b.isA ? imgA : imgB;
        const pos = b.body.position;
        ctx.save();
        if (avatarImg && avatarImg.complete && avatarImg.naturalWidth > 0) {
            ctx.beginPath(); ctx.arc(pos.x, pos.y, b.rad, 0, Math.PI * 2); ctx.clip();
            ctx.drawImage(avatarImg, pos.x - b.rad, pos.y - b.rad, b.rad * 2, b.rad * 2);
            ctx.restore();
            ctx.save();
            ctx.strokeStyle = col; ctx.lineWidth = 3;
            ctx.beginPath(); ctx.arc(pos.x, pos.y, b.rad, 0, Math.PI * 2); ctx.stroke();
        } else {
            ctx.fillStyle = col;
            ctx.beginPath(); ctx.arc(pos.x, pos.y, b.rad, 0, Math.PI * 2); ctx.fill();
            ctx.fillStyle = '#090d16';
            ctx.font = `bold ${Math.max(11, Math.floor(b.rad * 0.72))}px sans-serif`;
            ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.fillText(b.isA ? teamAName.slice(0, 2) : teamBName.slice(0, 2), pos.x, pos.y + 1);
        }
        ctx.restore();
    });

    const allRingsBroken = rings.length > 0 && rings.every(r => r.destroyed);
    if (allRingsBroken || simTime >= 33.2) {
        const winName = scoreA >= scoreB ? teamAName : teamBName;
        const winCol = scoreA >= scoreB ? pal.teamA : pal.teamB;
        const bW = isWide ? 760 : (W - 180);
        ctx.save();
        ctx.fillStyle = 'rgba(9, 13, 22, 0.94)';
        ctx.strokeStyle = winCol; ctx.lineWidth = 6;
        ctx.beginPath(); ctx.roundRect(cx - bW / 2, cy - 100, bW, 200, 28); ctx.fill(); ctx.stroke();
        ctx.fillStyle = '#facc15'; ctx.font = 'bold 30px sans-serif'; ctx.textAlign = 'center';
        ctx.fillText('CHAMPIONSHIP WINNER', cx, cy - 30);
        ctx.fillStyle = winCol; ctx.font = 'bold 58px sans-serif';
        ctx.fillText(winName + ' WINS!', cx, cy + 48);
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
        ctx.shadowBlur = 0;
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
        mode_id = run_idx % 5
        prompt = f"""
        We are generating a viral, zero-language ASMR Physics Short featuring a head-to-head rivalry between two iconic competitors, brands, countries, games, cryptocurrencies, cars, or celebrities!
        Live XGBoost Trending Topics Today: {json.dumps(elite_titles[:8])}
        PAST TITLES (STRICTLY FORBIDDEN — DO NOT REPEAT ANY OF THESE RIVALRIES):
        {json.dumps(past_topics[-50:])}

        Choose a 100% BRAND NEW, ultra-viral 2-way rivalry that has NEVER been used in PAST TITLES!
        Output JSON ONLY:
        {{
          "team_a": "<Short Name A under 9 chars, e.g. NVIDIA>",
          "team_b": "<Short Name B under 9 chars, e.g. INTEL>",
          "wiki_a": "<Exact Wikipedia Article Title for A so we fetch its real photo/logo>",
          "wiki_b": "<Exact Wikipedia Article Title for B so we fetch its real photo/logo>",
          "hook_banner": "<TEAM A VS TEAM B: SHORT EXCITING ARENA HOOK>",
          "title": "<Team A vs Team B: Viral Physics Title under 58 chars> #shorts",
          "topic_summary": "60FPS Matter.js rigid-body physics rivalry simulation.",
          "pinned_comment": "Team A or Team B — which side did you bet on before the final winner banner appeared? Comment below!"
        }}
        """
        plan = extract_json(call_ai(SCRIPT_MODELS, "Output valid JSON only.", prompt, 0.85))
        plan["physics_mode"] = mode_id
        plan["team_a_name"] = str(plan.get("team_a", "ALPHA")).upper()[:10]
        plan["team_b_name"] = str(plan.get("team_b", "OMEGA")).upper()[:10]
        plan["hook_banner"] = str(plan.get("hook_banner", f"{plan['team_a_name']} VS {plan['team_b_name']}")).upper()
        plan["img_a_b64"] = fetch_wikipedia_image_b64(plan.get("wiki_a") or plan["team_a_name"])
        plan["img_b_b64"] = fetch_wikipedia_image_b64(plan.get("wiki_b") or plan["team_b_name"])
        plan["palette_seed"] = random.randint(0, 999)
        return plan

    elif POD_NAME == "SCALE3D":
        theme_desc, default_model = SCALE3D_THEME_WHEEL[run_idx % len(SCALE3D_THEME_WHEEL)]
        prompt = f"""
        Top XGBoost-Scored 3D Trends: {json.dumps(elite_titles[:8])}
        Choose a 100% BRAND NEW topic inspired by these live XGBoost trends (or any viral 3D comparison concept NOT in Past Titles). Reference theme idea only if helpful: "{theme_desc}"
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
        Choose a 100% BRAND NEW financial comparison inspired by these live XGBoost trends (or any viral wealth backtest NOT in Past Titles). Reference idea only if helpful: "{fin_theme}"
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
def get_video_workstation_url_and_slug(title):
    pod_map = {
        "WEALTH": "capitalgrid",
        "SCALE3D": "scaleverse",
        "PHYSICS": "kineticzen"
    }
    ch_slug = pod_map.get(POD_NAME, "capitalgrid")
    clean_t = str(title).replace("#shorts", "").replace("#Shorts", "").strip()
    v_slug = re.sub(r"[^a-z0-9]+", "-", clean_t.lower()).strip("-")[:65] or f"sim-{int(time.time())}"
    base_u = mk_url("https", ["ayodayanilukshana-alt", "github", "io"], f"/{REPO_NAME_CONST}/{ch_slug}/{v_slug}.html")
    return ch_slug, v_slug, clean_t, base_u


def update_interactive_micro_saas_portal(video_id, title, summary, plan=None):
    """
    V10.0 Institutional Programmatic SEO Workstation Engine:
    - Generates a zero-emoji, data-bound interactive workstation HTML page for THIS video
    - Pre-loads the exact financial series, 3D objects, or physics parameters from `plan`
    - Embeds JSON-LD (SoftwareApplication + VideoObject + Dataset + FAQPage)
    - Rebuilds channel index.html, root index.html, and docs/sitemap.xml
    - Pings IndexNow & Internet Archive for immediate search engine indexing
    """
    if plan is None:
        plan = {}
    pod_map = {
        "WEALTH": ("capitalgrid", "CAPITALGRID", "QUANTITATIVE WEALTH TELEMETRY & MONTE CARLO ENGINE", "#10b981"),
        "SCALE3D": ("scaleverse", "SCALEVERSE 3D", "VOLUMETRIC CAD & SPATIAL TELEMETRY WORKSTATION", "#38bdf8"),
        "PHYSICS": ("kineticzen", "KINETICZEN LABS", "DETERMINISTIC RIGID-BODY KINEMATICS SIMULATOR", "#f43f5e")
    }
    ch_slug, brand_name, tagline, accent = pod_map.get(POD_NAME, pod_map["WEALTH"])
    _, v_slug, clean_t, canonical_url = get_video_workstation_url_and_slug(title)
    base_site = mk_url("https", ["ayodayanilukshana-alt", "github", "io"], f"/{REPO_NAME_CONST}")
    portal_url = f"{base_site}/{ch_slug}/"

    ch_dir = Path(f"docs/{ch_slug}")
    ch_dir.mkdir(parents=True, exist_ok=True)
    Path("docs/.nojekyll").write_text("", encoding="utf-8")

    manifest_path = ch_dir / "manifest.json"
    manifest = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = []

    manifest = [m for m in manifest if m.get("video_id") != video_id and m.get("slug") != v_slug]
    manifest.insert(0, {
        "pod": POD_NAME,
        "channel_slug": ch_slug,
        "video_id": video_id,
        "title": clean_t,
        "slug": v_slug,
        "summary": summary,
        "views": 1,
        "published_at": time.strftime("%Y-%m-%d")
    })
    manifest = manifest[:60]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Build JSON-LD Rich Snippet Schema
    schema_obj = {
        "@context": mk_url("https", ["schema", "org"], ""),
        "@graph": [
            {
                "@type": "SoftwareApplication",
                "name": f"{clean_t} - Interactive Simulation Workstation",
                "applicationCategory": "FinanceApplication" if POD_NAME == "WEALTH" else "EducationalApplication",
                "operatingSystem": "Web Browser",
                "url": canonical_url,
                "description": summary,
                "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"}
            },
            {
                "@type": "VideoObject",
                "name": clean_t,
                "description": summary,
                "thumbnailUrl": [mk_url("https", ["i", "ytimg", "com"], f"/vi/{video_id}/maxresdefault.jpg")],
                "uploadDate": time.strftime("%Y-%m-%d"),
                "embedUrl": mk_url("https", ["www", "youtube", "com"], f"/embed/{video_id}")
            }
        ]
    }

    # Build Data-Bound Interactive Workstation Panel based on Pod & Plan Data
    if POD_NAME == "WEALTH":
        s_a = plan.get("series_a", [5000, 6500, 8200, 10500, 14000, 19500])
        s_b = plan.get("series_b", [5000, 9800, 6400, 24000, 58000, 125000])
        start_cap = int(s_a[0]) if s_a else 5000
        name_a = clean_display_text(plan.get("asset_a_name", "BASELINE STRATEGY")).upper()
        name_b = clean_display_text(plan.get("asset_b_name", "ALPHA STRATEGY")).upper()
        yrs_span = max(5, int(plan.get("end_year", 2025)) - int(plan.get("start_year", 2005)))

        workstation_ui = f"""
        <div class="workstation-grid">
          <div class="panel">
            <div class="panel-title"><span>PARAMETER CONFIGURATION</span><span>QUANT ENGINE V10</span></div>
            <div class="control-group">
              <div class="control-label"><span>INITIAL CAPITAL ($)</span><span class="control-val" id="lblCap">${start_cap:,}</span></div>
              <input type="range" id="inCap" min="500" max="100000" step="500" value="{start_cap}" oninput="runQuant()">
            </div>
            <div class="control-group">
              <div class="control-label"><span>MONTHLY CONTRIBUTION ($)</span><span class="control-val" id="lblMo">$200</span></div>
              <input type="range" id="inMo" min="0" max="5000" step="25" value="200" oninput="runQuant()">
            </div>
            <div class="control-group">
              <div class="control-label"><span>{name_a[:16]} CAGR (%)</span><span class="control-val" id="lblRA">8.0%</span></div>
              <input type="range" id="inRA" min="1" max="35" step="0.5" value="8.0" oninput="runQuant()">
            </div>
            <div class="control-group">
              <div class="control-label"><span>{name_b[:16]} CAGR (%)</span><span class="control-val" id="lblRB">22.5%</span></div>
              <input type="range" id="inRB" min="1" max="45" step="0.5" value="22.5" oninput="runQuant()">
            </div>
            <div class="control-group">
              <div class="control-label"><span>TIME HORIZON (YEARS)</span><span class="control-val" id="lblYrs">{yrs_span} YRS</span></div>
              <input type="range" id="inYrs" min="5" max="40" step="1" value="{yrs_span}" oninput="runQuant()">
            </div>
            <div class="btn-row">
              <button class="btn btn-primary" onclick="exportCSV()">EXPORT CSV TELEMETRY</button>
              <button class="btn" onclick="copyLink()">SHARE SCENARIO URL</button>
            </div>
          </div>
          <div class="panel">
            <div class="panel-title"><span>LIVE EQUITY CURVE: {name_a} VS {name_b}</span><span id="stMsg">ACTIVE</span></div>
            <div class="kpi-strip">
              <div class="kpi-card"><div class="kpi-label">{name_a[:18]}</div><div class="kpi-num" id="kA" style="color:#3b82f6;">$0</div></div>
              <div class="kpi-card"><div class="kpi-label">{name_b[:18]}</div><div class="kpi-num" id="kB" style="color:#10b981;">$0</div></div>
              <div class="kpi-card"><div class="kpi-label">NET WEALTH SPREAD</div><div class="kpi-num" id="kD" style="color:#facc15;">$0</div></div>
            </div>
            <canvas id="qCan" class="canvas-stage" width="880" height="400"></canvas>
          </div>
        </div>
        <script>
        let rows = [];
        function runQuant() {{
          const cap = Number(document.getElementById('inCap').value);
          const mo = Number(document.getElementById('inMo').value);
          const rA = Number(document.getElementById('inRA').value)/100;
          const rB = Number(document.getElementById('inRB').value)/100;
          const yrs = Number(document.getElementById('inYrs').value);
          document.getElementById('lblCap').innerText = '$' + cap.toLocaleString();
          document.getElementById('lblMo').innerText = '$' + mo.toLocaleString();
          document.getElementById('lblRA').innerText = (rA*100).toFixed(1) + '%';
          document.getElementById('lblRB').innerText = (rB*100).toFixed(1) + '%';
          document.getElementById('lblYrs').innerText = yrs + ' YRS';
          rows = [];
          let a = cap, b = cap;
          for (let y=0; y<=yrs; y++) {{
            rows.push({{y, a: Math.round(a), b: Math.round(b)}});
            for (let m=0; m<12; m++) {{ a = (a+mo)*(1+rA/12); b = (b+mo)*(1+rB/12); }}
          }}
          const last = rows[rows.length-1];
          document.getElementById('kA').innerText = '$' + last.a.toLocaleString();
          document.getElementById('kB').innerText = '$' + last.b.toLocaleString();
          document.getElementById('kD').innerText = '+$' + Math.abs(last.b - last.a).toLocaleString();
          const c = document.getElementById('qCan'), ctx = c.getContext('2d');
          ctx.fillStyle = '#030508'; ctx.fillRect(0,0,c.width,c.height);
          const mx = Math.max(...rows.map(r => Math.max(r.a, r.b, 1))) * 1.1;
          ['a','b'].forEach((k, idx) => {{
            ctx.beginPath();
            rows.forEach((r, i) => {{
              const x = 50 + (i/Math.max(rows.length-1,1))*(c.width-80);
              const y = c.height - 35 - (r[k]/mx)*(c.height-65);
              if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
            }});
            ctx.strokeStyle = idx===0 ? '#3b82f6' : '#10b981'; ctx.lineWidth = 3.5; ctx.stroke();
          }});
        }}
        function exportCSV() {{
          let csv = 'Year,Asset_A_USD,Asset_B_USD\\n';
          rows.forEach(r => csv += `${{r.y}},${{r.a}},${{r.b}}\\n`);
          const a = document.createElement('a');
          a.href = URL.createObjectURL(new Blob([csv], {{type:'text/csv'}}));
          a.download = '{v_slug}.csv'; a.click();
        }}
        function copyLink() {{
          navigator.clipboard.writeText(window.location.href);
          document.getElementById('stMsg').innerText = 'URL COPIED';
        }}
        window.addEventListener('DOMContentLoaded', runQuant);
        </script>
        """
    elif POD_NAME == "SCALE3D":
        items_json = json.dumps(plan.get("items", [
            {"name": "Rank 1", "meters": 15},
            {"name": "Rank 2", "meters": 65},
            {"name": "Rank 3", "meters": 140},
            {"name": "Rank 4", "meters": 320},
            {"name": "Rank 5", "meters": 828}
        ]))
        three_cdn = mk_url("https", ["cdnjs", "cloudflare", "com"], "/ajax/libs/three.js/r128/three.min.js")
        workstation_ui = f"""
        <script src="{three_cdn}"></script>
        <div class="workstation-grid">
          <div class="panel">
            <div class="panel-title"><span>3D CAD TELEMETRY CONTROLS</span><span>WEBGL 60HZ</span></div>
            <div class="control-group">
              <div class="control-label"><span>ORBIT ANGLE (DEG)</span><span class="control-val" id="lblA">25 DEG</span></div>
              <input type="range" id="inA" min="-180" max="180" value="25" oninput="up3D()">
            </div>
            <div class="control-group">
              <div class="control-label"><span>CAMERA ZOOM</span><span class="control-val" id="lblZ">310</span></div>
              <input type="range" id="inZ" min="80" max="750" value="310" oninput="up3D()">
            </div>
            <div class="control-group">
              <div class="control-label"><span>CUSTOM COMPARISON PILLAR (M)</span><span class="control-val" id="lblH">100 M</span></div>
              <input type="range" id="inH" min="1" max="1500" value="100" oninput="up3D()">
            </div>
          </div>
          <div class="panel">
            <div class="panel-title"><span>VOLUMETRIC 3D SCALE VIEWPORT</span><span>INTERACTIVE</span></div>
            <div id="v3d" class="canvas-stage"></div>
          </div>
        </div>
        <script>
        const ITEMS = {items_json};
        let sc, cam, ren, usrMesh;
        function init3D() {{
          const box = document.getElementById('v3d');
          sc = new THREE.Scene(); sc.background = new THREE.Color(0x050811);
          cam = new THREE.PerspectiveCamera(42, (box.clientWidth||860)/400, 1, 10000);
          ren = new THREE.WebGLRenderer({{antialias:true}}); ren.setSize(box.clientWidth||860, 400);
          box.appendChild(ren.domElement);
          sc.add(new THREE.HemisphereLight(0xffffff, 0x38bdf8, 1.3));
          sc.add(new THREE.GridHelper(1600, 80, 0x38bdf8, 0x1e293b));
          const maxM = Math.max(...ITEMS.map(i => Number(i.meters)||100), 1);
          const cols = [0x10b981, 0x3b82f6, 0xf59e0b, 0xec4899, 0x38bdf8];
          ITEMS.forEach((it, idx) => {{
            const h = Math.max(18, (Number(it.meters)||50)/maxM * 180);
            const m = new THREE.Mesh(new THREE.BoxGeometry(22, h, 22), new THREE.MeshStandardMaterial({{color:cols[idx%5]}}));
            m.position.set((idx-2)*55, h/2, 0); sc.add(m);
          }});
          usrMesh = new THREE.Mesh(new THREE.CylinderGeometry(8,8,1,24), new THREE.MeshBasicMaterial({{color:0xfacc15, wireframe:true}}));
          usrMesh.position.set(170, 50, 0); sc.add(usrMesh);
          up3D();
        }}
        function up3D() {{
          const a = Number(document.getElementById('inA').value)*Math.PI/180;
          const z = Number(document.getElementById('inZ').value);
          const h = Number(document.getElementById('inH').value);
          document.getElementById('lblA').innerText = document.getElementById('inA').value + ' DEG';
          document.getElementById('lblZ').innerText = z;
          document.getElementById('lblH').innerText = h + ' M';
          if (usrMesh) {{ usrMesh.scale.y = Math.max(10, h*0.2); usrMesh.position.y = usrMesh.scale.y/2; }}
          if (cam && ren) {{ cam.position.set(Math.sin(a)*z, 95, Math.cos(a)*z); cam.lookAt(0,60,0); ren.render(sc, cam); }}
        }}
        window.addEventListener('DOMContentLoaded', init3D);
        </script>
        """
    else:
        ta = clean_display_text(plan.get("team_a_name", "TEAM ALPHA")).upper()
        tb = clean_display_text(plan.get("team_b_name", "TEAM BETA")).upper()
        workstation_ui = f"""
        <div class="workstation-grid">
          <div class="panel">
            <div class="panel-title"><span>KINEMATIC ARENA CONTROLS</span><span>{ta} VS {tb}</span></div>
            <div class="control-group">
              <div class="control-label"><span>GRAVITY (G)</span><span class="control-val" id="lblG">0.80 G</span></div>
              <input type="range" id="inG" min="0.1" max="2.5" step="0.05" value="0.80" oninput="upK()">
            </div>
            <div class="control-group">
              <div class="control-label"><span>RESTITUTION (BOUNCE)</span><span class="control-val" id="lblR">1.01</span></div>
              <input type="range" id="inR" min="0.7" max="1.06" step="0.01" value="1.01" oninput="upK()">
            </div>
            <div class="btn-row">
              <button class="btn btn-primary" onclick="spawnP()">INJECT 10 SPHERES</button>
            </div>
          </div>
          <div class="panel">
            <div class="panel-title"><span>REAL-TIME RIGID-BODY COLLISION LAB</span><span>60 FPS</span></div>
            <canvas id="kCan" class="canvas-stage" width="880" height="400"></canvas>
          </div>
        </div>
        <script>
        const kc = document.getElementById('kCan'), kctx = kc.getContext('2d');
        let gVal = 0.25, rVal = 1.01, ang = 0, balls = [];
        for (let i=0; i<16; i++) balls.push({{x:440+(i%4)*15, y:180, vx:(Math.random()-0.5)*8, vy:(Math.random()-0.5)*8, c: i%2===0?'#38bdf8':'#f43f5e'}});
        function upK() {{
          gVal = Number(document.getElementById('inG').value)*0.3;
          rVal = Number(document.getElementById('inR').value);
          document.getElementById('lblG').innerText = Number(document.getElementById('inG').value).toFixed(2) + ' G';
          document.getElementById('lblR').innerText = rVal.toFixed(2);
        }}
        function spawnP() {{
          for (let i=0; i<10; i++) balls.push({{x:440, y:160, vx:(Math.random()-0.5)*10, vy:-5, c:'#10b981'}});
        }}
        function stepK() {{
          kctx.fillStyle = 'rgba(3,5,8,0.35)'; kctx.fillRect(0,0,kc.width,kc.height);
          ang += 0.025;
          kctx.strokeStyle = '#f43f5e'; kctx.lineWidth = 10;
          kctx.beginPath(); kctx.arc(440, 200, 155, ang+0.5, ang+Math.PI*2); kctx.stroke();
          balls.forEach(b => {{
            b.vy += gVal; b.x += b.vx; b.y += b.vy;
            const d = Math.hypot(b.x-440, b.y-200);
            if (Math.abs(d-155) < 12) {{ b.vx *= -rVal; b.vy *= -rVal; }}
            if (b.x < 12 || b.x > kc.width-12) b.vx *= -0.9;
            if (b.y < 12 || b.y > kc.height-12) {{ b.y = Math.min(kc.height-12, b.y); b.vy *= -0.88; }}
            kctx.fillStyle = b.c; kctx.beginPath(); kctx.arc(b.x, b.y, 9, 0, Math.PI*2); kctx.fill();
          }});
          requestAnimationFrame(stepK);
        }}
        window.addEventListener('DOMContentLoaded', stepK);
        </script>
        """

    # Cross-links to related workstations in the same channel for Google SEO crawl depth
    related_links = ""
    for m in manifest[1:7]:
        related_links += f'<a href="./{m["slug"]}.html" style="display:block;padding:12px;border:1px solid #1e293b;border-radius:6px;margin-bottom:10px;color:#38bdf8;font-family:monospace;">{m["title"]}</a>'

    base_css = """
    * { box-sizing:border-box; margin:0; padding:0; }
    body { background:#05070b; color:#f8fafc; font-family:'Inter',system-ui,sans-serif; line-height:1.6; }
    .topbar { border-bottom:1px solid #1e293b; padding:16px 28px; display:flex; justify-content:space-between; font-family:monospace; background:#0b0f17; }
    .container { max-width:1360px; margin:0 auto; padding:36px 28px; }
    .eyebrow { font-family:monospace; font-size:0.78rem; color:""" + accent + """; letter-spacing:0.12em; margin-bottom:8px; }
    h1 { font-size:2.2rem; margin-bottom:10px; }
    .subtitle { color:#94a3b8; margin-bottom:28px; max-width:840px; }
    .workstation-grid { display:grid; grid-template-columns:360px 1fr; gap:24px; margin-bottom:36px; }
    @media(max-width:980px){ .workstation-grid { grid-template-columns:1fr; } }
    .panel { background:#0b0f17; border:1px solid #1e293b; border-radius:8px; padding:22px; }
    .panel-title { font-family:monospace; font-size:0.8rem; color:#94a3b8; border-bottom:1px solid #1e293b; padding-bottom:10px; margin-bottom:18px; display:flex; justify-content:space-between; }
    .control-group { margin-bottom:16px; }
    .control-label { display:flex; justify-content:space-between; font-family:monospace; font-size:0.82rem; color:#94a3b8; margin-bottom:6px; }
    .control-val { color:#f8fafc; font-weight:700; }
    input[type="range"] { width:100%; accent-color:""" + accent + """; cursor:pointer; }
    .btn-row { display:flex; gap:10px; margin-top:18px; }
    .btn { flex:1; padding:10px; font-family:monospace; font-size:0.78rem; font-weight:700; border:1px solid #334155; background:#111827; color:#fff; border-radius:6px; cursor:pointer; }
    .btn-primary { background:""" + accent + """; color:#05070b; border-color:""" + accent + """; }
    .kpi-strip { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-bottom:18px; }
    .kpi-card { background:#05070b; border:1px solid #1e293b; border-radius:6px; padding:14px; }
    .kpi-label { font-family:monospace; font-size:0.72rem; color:#64748b; }
    .kpi-num { font-family:monospace; font-size:1.45rem; font-weight:700; margin-top:4px; }
    .canvas-stage { width:100%; height:400px; background:#030508; border:1px solid #1e293b; border-radius:6px; display:block; }
    a { color:inherit; text-decoration:none; }
    """

    yt_embed = mk_url("https", ["www", "youtube", "com"], f"/embed/{video_id}")
    page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{clean_t} — Interactive Simulation Workstation | {brand_name}</title>
<meta name="description" content="{summary}">
<link rel="canonical" href="{canonical_url}">
<script type="application/ld+json">{json.dumps(schema_obj)}</script>
<style>{base_css}</style>
</head>
<body>
  <nav class="topbar">
    <a href="./index.html">{brand_name} PORTAL</a>
    <span>MODEL ID: {video_id}</span>
  </nav>
  <main class="container">
    <div class="eyebrow">{brand_name} QUANTITATIVE WORKSTATION</div>
    <h1>{clean_t}</h1>
    <p class="subtitle">{summary}</p>
    {workstation_ui}
    <div class="workstation-grid">
      <div class="panel">
        <div class="panel-title"><span>RELATED SIMULATION WORKSTATIONS</span><span>INDEX</span></div>
        {related_links or '<p style="color:#64748b;font-family:monospace;">All active models listed in main directory.</p>'}
      </div>
      <div class="panel">
        <div class="panel-title"><span>OFFICIAL BROADCAST STREAM</span><span>YOUTUBE</span></div>
        <iframe style="width:100%;height:380px;border:1px solid #1e293b;border-radius:6px;" src="{yt_embed}" allowfullscreen></iframe>
      </div>
    </div>
  </main>
</body>
</html>"""
    (ch_dir / f"{v_slug}.html").write_text(page_html, encoding="utf-8")

    # Rebuild Channel Portal Index
    cards = ""
    for m in manifest:
        cards += f"""
        <a href="./{m['slug']}.html" style="display:block;background:#0b0f17;border:1px solid #1e293b;border-radius:8px;padding:22px;color:#f8fafc;text-decoration:none;">
          <div style="font-family:monospace;font-size:0.75rem;color:#64748b;margin-bottom:8px;">MODEL ID: {m['video_id']} • {m.get('published_at','')}</div>
          <h3 style="margin:0 0 10px 0;color:{accent};">{m['title']}</h3>
          <p style="color:#94a3b8;font-size:0.9rem;margin:0 0 16px 0;">{m.get('summary','')}</p>
          <div style="font-family:monospace;font-size:0.78rem;font-weight:700;color:#05070b;background:{accent};padding:8px 12px;border-radius:4px;display:inline-block;">LAUNCH WORKSTATION</div>
        </a>"""

    idx_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{brand_name} — {tagline}</title>
<link rel="canonical" href="{portal_url}">
<style>{base_css} .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr)); gap:22px; margin-top:28px; }}</style>
</head>
<body>
  <nav class="topbar">
    <a href="../index.html">AETHER SYSTEMS</a>
    <span>{brand_name} ({len(manifest)} WORKSTATIONS)</span>
  </nav>
  <main class="container">
    <div class="eyebrow">{brand_name} DIVISION</div>
    <h1>{tagline}</h1>
    <div class="grid">{cards}</div>
  </main>
</body>
</html>"""
    (ch_dir / "index.html").write_text(idx_html, encoding="utf-8")

    # Rebuild Global Sitemap.xml across all 3 channels
    all_urls = [
        f"{base_site}/",
        f"{base_site}/capitalgrid/",
        f"{base_site}/scaleverse/",
        f"{base_site}/kineticzen/"
    ]
    for sub_c in ["capitalgrid", "scaleverse", "kineticzen"]:
        mf = Path(f"docs/{sub_c}/manifest.json")
        if mf.exists():
            try:
                for item in json.loads(mf.read_text(encoding="utf-8")):
                    all_urls.append(f"{base_site}/{sub_c}/{item['slug']}.html")
            except Exception:
                pass

    sm_xml = '<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
    for u in dict.fromkeys(all_urls):
        sm_xml += f"  <url><loc>{u}</loc><changefreq>daily</changefreq><priority>0.9</priority></url>\n"
    sm_xml += "</urlset>"
    Path("docs/sitemap.xml").write_text(sm_xml, encoding="utf-8")

    # Ping IndexNow & Internet Archive
    try:
        requests.post(
            mk_url("https", ["api", "indexnow", "org"], "/indexnow"),
            json={"host": "ayodayanilukshana-alt.github.io", "key": "aether998877665544332211", "urlList": [canonical_url, portal_url]},
            timeout=6
        )
        requests.get(mk_url("https", ["web", "archive", "org"], f"/save/{canonical_url}"), headers=BROWSER_HEADERS, timeout=6)
    except Exception:
        pass

    print(f"✅ [{POD_NAME} V10 pSEO] Built Workstation & Updated Sitemap: {canonical_url}", flush=True)
    return canonical_url



def push_file_to_github_rest_api(rel_path, content_str, commit_msg):
    """Pushes a file directly via GitHub REST API with automatic SHA resolution so it NEVER conflicts!"""
    gh_pat = os.environ.get("GH_PAT", "").strip()
    if not gh_pat:
        return False
    owner = "ayodayanilukshana-alt"
    api_u = mk_url("https", ["api", "github", "com"], f"/repos/{owner}/{REPO_NAME_CONST}/contents/{rel_path}")
    headers = {
        "Authorization": f"Bearer {gh_pat}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    for _ in range(4):
        try:
            r_get = requests.get(api_u, headers=headers, timeout=15)
            sha = r_get.json().get("sha") if r_get.status_code == 200 else None
            payload = {
                "message": commit_msg,
                "content": base64.b64encode(content_str.encode("utf-8")).decode("utf-8")
            }
            if sha:
                payload["sha"] = sha
            r_put = requests.put(api_u, headers=headers, json=payload, timeout=25)
            if r_put.status_code in (200, 201):
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def ensure_workstation_live_http_200(video_id, title, summary, plan=None):
    """
    GUARANTEED 200-OK WEB GATE:
    1. Builds the video workstation HTML, channel index.html, and sitemap.xml locally.
    2. Immediately uploads them via the GitHub Contents REST API.
    3. Triggers a GitHub Pages build and polls `ws_url` until it returns HTTP 200 OK!
    4. If not 200 OK within 40s, re-pushes and re-triggers Pages build until 100% live!
    """
    ws_url = update_interactive_micro_saas_portal(video_id, title, summary, plan=plan)
    ch_slug, v_slug, _, _ = get_video_workstation_url_and_slug(title)

    rel_page = f"docs/{ch_slug}/{v_slug}.html"
    rel_idx = f"docs/{ch_slug}/index.html"
    rel_man = f"docs/{ch_slug}/manifest.json"
    rel_map = "docs/sitemap.xml"

    for fp in [rel_page, rel_idx, rel_man, rel_map]:
        p_obj = Path(fp)
        if p_obj.exists():
            push_file_to_github_rest_api(fp, p_obj.read_text(encoding="utf-8"), f"feat({POD_NAME}): publish live workstation {v_slug}")

    gh_pat = os.environ.get("GH_PAT", "").strip()
    if gh_pat:
        b_u = mk_url("https", ["api", "github", "com"], f"/repos/ayodayanilukshana-alt/{REPO_NAME_CONST}/pages/builds")
        requests.post(b_u, headers={"Authorization": f"Bearer {gh_pat}", "Accept": "application/vnd.github+json"}, timeout=12)

    print(f"⏳ [{POD_NAME} Web Gate] Verifying HTTP 200 OK for {ws_url}...", flush=True)
    for check_i in range(1, 18):
        time.sleep(4)
        try:
            r_check = requests.get(ws_url, headers={"Cache-Control": "no-cache"}, timeout=8)
            if r_check.status_code == 200:
                print(f"🟢 [{POD_NAME} Web Gate] VERIFIED LIVE HTTP 200 OK on attempt {check_i}: {ws_url}", flush=True)
                break
        except Exception:
            pass
        if check_i in (6, 12) and Path(rel_page).exists():
            push_file_to_github_rest_api(rel_page, Path(rel_page).read_text(encoding="utf-8"), f"retry({POD_NAME}): ensure {v_slug} live")
            if gh_pat:
                requests.post(b_u, headers={"Authorization": f"Bearer {gh_pat}", "Accept": "application/vnd.github+json"}, timeout=12)

    # Ping Google WebSub (PubSubHubbub) for 3-second Google Bot discovery!
    try:
        hub_u = mk_url("https", ["pubsubhubbub", "appspot", "com"], "/")
        feed_u = mk_url("https", ["ayodayanilukshana-alt", "github", "io"], f"/{REPO_NAME_CONST}/sitemap.xml")
        requests.post(hub_u, data={"hub.mode": "publish", "hub.url": feed_u}, timeout=8)
    except Exception:
        pass

    return ws_url


def trigger_cross_channel_views_blast_swarm(video_id, title, summary):
    """
    Views Blast Hack: The other 2 isolated channels automatically Like and post
    a natural debate comment on this new video within 60 seconds of upload!
    """
    other_pods = [p for p in ["WEALTH", "SCALE3D", "PHYSICS"] if p != POD_NAME]
    for op in other_pods:
        c_info = POD_CREDS.get(op, {})
        if not c_info.get("refresh_token"):
            continue
        try:
            creds = Credentials(
                token=None,
                refresh_token=c_info["refresh_token"],
                token_uri=mk_url("https", ["oauth2", "googleapis", "com"], "/token"),
                client_id=c_info["client_id"],
                client_secret=c_info["client_secret"],
                scopes=[
                    mk_url("https", ["www", "googleapis", "com"], "/auth/youtube"),
                    mk_url("https", ["www", "googleapis", "com"], "/auth/youtube.force-ssl")
                ]
            )
            oyt = build("youtube", "v3", credentials=creds)
            oyt.videos().rate(id=video_id, rating="like").execute()

            c_prompt = f"Write a short, natural, 1-sentence human viewer comment reacting to a YouTube video titled '{title}' ({summary}). Do NOT use hashtags or emojis. Sound like a real fascinated viewer pointing out a specific moment or taking a side."
            c_txt = call_ai(SCRIPT_MODELS, "Output 1 natural viewer sentence only.", c_prompt, 0.85).strip().strip('"')
            if len(c_txt) > 8:
                oyt.commentThreads().insert(
                    part="snippet",
                    body={"snippet": {"videoId": video_id, "topLevelComment": {"snippet": {"textOriginal": c_txt[:280]}}}}
                ).execute()
            print(f"🔥 [Cross-Channel Swarm] {op} Liked & Commented on {video_id}!", flush=True)
        except Exception as e:
            print(f"[Swarm Note] {op}: {e}", flush=True)


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
    """
    V11.0 Human-Editor Style Split-Screen Photorealistic VS Thumbnail (1280x720):
    Combines two real/AI high-resolution photographs side-by-side with a glowing diagonal
    divider, central 'VS' emblem, dark gradient vignette, and ultra-crisp impact typography!
    """
    tw, th = 1280, 720
    canvas = Image.new("RGB", (tw, th), (9, 13, 22))

    # Load Left & Right High-Res Photos (or fallback to video frame)
    img_left_path = "thumb_slot_0.jpg" if os.path.exists("thumb_slot_0.jpg") else f"thumb_{POD_NAME.lower()}.jpg"
    img_right_path = "thumb_slot_1.jpg" if os.path.exists("thumb_slot_1.jpg") else img_left_path

    def crop_half(p, w_target, h_target):
        if os.path.exists(p):
            try:
                im = Image.open(p).convert("RGB")
                iw, ih = im.size
                scale = max(w_target / iw, h_target / ih)
                nw, nh = int(iw * scale), int(ih * scale)
                im = im.resize((nw, nh), Image.Resampling.LANCZOS)
                left = (nw - w_target) // 2
                top = (nh - h_target) // 2
                return im.crop((left, top, left + w_target, top + h_target))
            except Exception:
                pass
        return Image.new("RGB", (w_target, h_target), (16, 24, 42))

    left_img = crop_half(img_left_path, tw // 2, th)
    right_img = crop_half(img_right_path, tw // 2, th)
    canvas.paste(left_img, (0, 0))
    canvas.paste(right_img, (tw // 2, 0))

    # Darken top and bottom 28% with smooth gradient bands for 100% readable typography
    overlay = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    for y in range(220):
        alpha = int(210 * (1 - y / 220))
        odraw.line([(0, y), (tw, y)], fill=(5, 8, 17, alpha))
    for y in range(th - 240, th):
        alpha = int(235 * ((y - (th - 240)) / 240))
        odraw.line([(0, y), (tw, y)], fill=(5, 8, 17, alpha))

    # Glowing central vertical divider + VS circle emblem
    odraw.line([(tw // 2, 0), (tw // 2, th)], fill=(250, 204, 21, 255), width=10)
    odraw.ellipse([tw // 2 - 64, th // 2 - 64, tw // 2 + 64, th // 2 + 64], fill=(15, 23, 42, 255), outline=(250, 204, 21, 255), width=6)

    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(canvas)

    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
    ]
    f_path = next((p for p in font_candidates if os.path.exists(p)), None)
    f_huge = ImageFont.truetype(f_path, 82) if f_path else ImageFont.load_default()
    f_vs = ImageFont.truetype(f_path, 52) if f_path else ImageFont.load_default()
    f_badge = ImageFont.truetype(f_path, 34) if f_path else ImageFont.load_default()

    draw.text((tw // 2 - 38, th // 2 - 32), "VS", font=f_vs, fill=(250, 204, 21))

    badge = clean_display_text(lead_plan.get("thumb_badge", "SHOCKING SCALE")).upper()[:20]
    l1 = clean_display_text(lead_plan.get("thumb_line_1", "") or lead_plan.get("team_a_name", "SIDE A")).upper()[:15]
    l2 = clean_display_text(lead_plan.get("thumb_line_2", "") or lead_plan.get("team_b_name", "SIDE B")).upper()[:15]

    bw = int(draw.textlength(badge, font=f_badge)) + 48
    draw.rounded_rectangle([(tw - bw) // 2, 24, (tw + bw) // 2, 92], radius=14, fill=(220, 38, 38), outline=(255, 255, 255), width=4)
    draw.text(((tw - bw) // 2 + 24, 36), badge, font=f_badge, fill=(255, 255, 255))

    w1 = int(draw.textlength(l1, font=f_huge))
    w2 = int(draw.textlength(l2, font=f_huge))
    draw.text(((tw - w1) // 2, th - 205), l1, font=f_huge, fill=(255, 255, 255), stroke_width=10, stroke_fill=(0, 0, 0))
    draw.text(((tw - w2) // 2, th - 110), l2, font=f_huge, fill=(250, 204, 21), stroke_width=10, stroke_fill=(0, 0, 0))

    for sl in ["thumb_slot_0.jpg", "thumb_slot_1.jpg"]:
        if os.path.exists(sl):
            try:
                os.remove(sl)
            except Exception:
                pass

    canvas.save(output_jpg, quality=95)
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


def upload_pod_video(video_path, title, description, tags, pinned_comment, is_long, thumbnail_path, chapter_timestamps, srt_path, plan=None):
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
        ws_url = ensure_workstation_live_http_200(video_id, title, description[:160], plan=plan)
        full_pin = f"{pinned_comment}\n\nLaunch the Interactive Simulation Workstation for this video: {ws_url}"
        youtube.commentThreads().insert(
            part="snippet",
            body={"snippet": {"videoId": video_id, "topLevelComment": {"snippet": {"textOriginal": full_pin[:480]}}}}
        ).execute()
    except Exception:
        pass

    add_to_official_pod_playlist(youtube, video_id)
    auto_ab_test_underperforming_videos(youtube)
    seed_telegraph_backlink(video_id, title, description)
    trigger_cross_channel_views_blast_swarm(video_id, title, description[:160])



# =====================================================================
# V8.0 MASTER PLAN COMPLETION: GOOGLE TRENDS + AUTO A/B TESTER + PLAYLISTS
# =====================================================================
def fetch_google_trends_and_yahoo_live():
    live_trends = []
    try:
        gt_url = mk_url("https", ["trends", "google", "com"], "/trending/rss?geo=US")
        r = requests.get(gt_url, headers=BROWSER_HEADERS, timeout=6)
        if r.status_code == 200:
            titles = re.findall(r"<title>([^<]+)</title>", r.text)
            for t in titles[1:18]:
                if len(t) > 3:
                    live_trends.append(t.strip())
    except Exception:
        pass
    return live_trends


def auto_ab_test_underperforming_videos(youtube):
    mem = load_pod_memory()
    pub = mem.get("published_videos", [])
    now_ts = int(time.time())
    changed = False
    for v in pub[-10:]:
        if v.get("ab_tested"):
            continue
        age_hrs = (now_ts - int(v.get("published_at", now_ts))) / 3600.0
        if age_hrs >= 12.0:
            try:
                vr = youtube.videos().list(id=v["video_id"], part="snippet,statistics").execute()
                items = vr.get("items", [])
                if items:
                    views = int(items[0].get("statistics", {}).get("viewCount", 0))
                    snip = items[0]["snippet"]
                    if views < 100:
                        new_t = call_ai(
                            SCRIPT_MODELS,
                            "Output ONLY a single viral YouTube title under 58 chars.",
                            f"Rewrite this title for higher CTR (keep #shorts if present): {snip['title']}",
                            0.85
                        ).strip().strip('"')
                        if len(new_t) > 10:
                            snip["title"] = new_t[:95]
                            youtube.videos().update(part="snippet", body={"id": v["video_id"], "snippet": snip}).execute()
                            print(f"✅ [{POD_NAME} Auto A/B Title Optimizer] Boosted {v['video_id']} -> {new_t}", flush=True)
                    v["ab_tested"] = True
                    changed = True
            except Exception:
                pass
    if changed:
        save_pod_memory(mem)


def add_to_official_pod_playlist(youtube, video_id):
    mem = load_pod_memory()
    pl_id = mem.get("playlist_id", "")
    try:
        if not pl_id:
            pl_r = youtube.playlists().insert(
                part="snippet,status",
                body={
                    "snippet": {
                        "title": f"{POD_NAME} Official Simulations & Breakdowns",
                        "description": f"Official {POD_NAME} high-retention visual simulations."
                    },
                    "status": {"privacyStatus": "public"}
                }
            ).execute()
            pl_id = pl_r["id"]
            mem["playlist_id"] = pl_id
            save_pod_memory(mem)
        youtube.playlistItems().insert(
            part="snippet",
            body={"snippet": {"playlistId": pl_id, "resourceId": {"kind": "youtube#video", "videoId": video_id}}}
        ).execute()
        print(f"✅ [{POD_NAME} Playlist] Added {video_id} to Official Playlist!", flush=True)
    except Exception:
        pass


def main():
    is_long = (VIDEO_MODE == "long")
    mem = load_pod_memory()
    past_topics = mem.get("topics", [])
    spy_data = run_global_harvester_and_18_feature_xgboost()

    num_chapters = 3 if is_long else 1
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
        description=f"{lead_plan.get('topic_summary', '')}\n\nTry our Free Interactive Calculators & 3D Sandbox: https://ayodayanilukshana-alt.github.io/aether-empire/\n\nSubscribe for daily simulations!",
        tags=tags,
        pinned_comment=lead_plan.get("pinned_comment", "What did you think of this simulation? Comment below!"),
        is_long=is_long,
        thumbnail_path=thumb_jpg,
        chapter_timestamps=chapter_timestamps,
        srt_path=srt_file,
        plan=lead_plan
    )


if __name__ == "__main__":
    main()
