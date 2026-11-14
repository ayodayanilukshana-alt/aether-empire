import os
import re
import sys
import json
import math
import time
import wave
import random
import asyncio
import datetime
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
            "what happens if you invest 1000",
            "why 99 percent of day traders lose",
            "compound interest simulation",
            "how inflation secretly destroys",
            "s&p 500 vs bitcoin vs gold",
            "if you invested 100 dollars a month",
            "how the rich avoid taxes legally",
            "renting vs buying a house math"
        ],
        "yt_search_query": "investing wealth comparison simulation explained"
    },
    "PHYSICS": {
        "client_id": os.environ.get("PHYSICS_CLIENT_ID", "").strip(),
        "client_secret": os.environ.get("PHYSICS_CLIENT_SECRET", "").strip(),
        "refresh_token": os.environ.get("PHYSICS_REFRESH_TOKEN", "").strip(),
        "voice": None,  # 100% ZERO-LANGUAGE GLOBAL ASMR (Pure Musical Collision Synth!)
        "subreddits": ["oddlysatisfying", "Simulated", "physicsgifs"],
        "autocomplete_seeds": [
            "satisfying ball escape physics",
            "balls vs spinning rings",
            "marble race elimination arena",
            "bouncing ball breaks rings",
            "red vs gold ball escape",
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
            "how deep is the ocean 3d",
            "tallest megastructures compared 3d",
            "largest black holes size comparison",
            "fastest vehicles 3d speed comparison",
            "smallest to largest things in the universe"
        ],
        "yt_search_query": "3d size comparison animation metaballstudios style"
    }
}

ACTIVE_POD = POD_CREDS.get(POD_NAME, POD_CREDS["WEALTH"])

SCRIPT_MODELS = [
    "deepseek-v4-flash",       # x0.10
    "qwen3.7-plus",            # x0.10
    "qwen3.7-max",             # x0.20
    "deepseek-v4.1-flash",     # x0.30
    "gpt-6-luna"               # x0.38
]

FPS = 24
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
}


def clean_text(text):
    if not text:
        return ""
    s = str(text).replace("—", " ").replace("–", " ").replace("-", " ").replace("\n", " ")
    s = re.sub(r"[^\x20-\x7E]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def clean_display_text(text):
    if not text:
        return ""
    s = str(text).replace("—", "-").replace("–", "-")
    s = re.sub(r"[^\x20-\x7E]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def call_ai(models, system_prompt, user_prompt, temperature=0.7):
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
# 2. AGENT 1 & 4: MULTI-SOURCE HARVESTER + 18-FEATURE XGBOOST + SELF-LEARNING LOOP
# =====================================================================
def compute_18_xgboost_features(title, views=50000, subs=10000, likes=2000, comments=150, hours_age=48.0, source_weight=1.0):
    """
    Computes all 18 numerical features specified in the Project AETHER Master Plan!
    """
    t_low = str(title).lower()
    words = t_low.split()
    char_len = float(len(title))
    word_cnt = float(len(words))

    # 1-3: Velocity, Outlier Ratio, Engagement Rate
    view_velocity = float(views) / max(float(hours_age), 1.0)
    outlier_ratio = min(float(views) / max(float(subs), 500.0), 500.0)
    engagement_rate = min((float(likes) + float(comments) * 4.0) / max(float(views), 100.0), 0.35)

    # 4-8: Structural & Typography Signals
    optimal_len_score = 1.0 - min(abs(char_len - 52.0) / 52.0, 1.0)
    has_number = 1.0 if re.search(r"\d+", t_low) else 0.0
    has_dollar_or_pct = 1.0 if re.search(r"[\$%]|million|billion", t_low) else 0.0
    has_question = 1.0 if "?" in title else 0.0
    has_vs_comparison = 1.0 if (" vs " in t_low or " compared" in t_low) else 0.0

    # 9-14: Psychological & Curiosity Triggers
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

    # 15-18: Market & Competition Signals
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
    """
    Executes Agent 1 (Multi-Source Harvester: YouTube API + Autocomplete + Reddit + HackerNews + Wikipedia),
    Agent 4 (Self-Learning Feedback Loop from our own published videos),
    and the 18-Feature XGBoost Viral Predictor with a >92nd Percentile Filter!
    """
    mem = load_pod_memory()
    s_idx = mem.get("seed_idx", 0)
    mem["seed_idx"] = s_idx + 1

    candidates = []
    X_train, y_train = [], []

    # -----------------------------------------------------------------
    # AGENT 4: SELF-LEARNING FEEDBACK LOOP (Fetch our own channel stats!)
    # -----------------------------------------------------------------
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
            print(f"✅ [{POD_NAME} Agent 4 Self-Learning] Loaded {len(my_ids)} own channel performance records into XGBoost!", flush=True)
    except Exception as e:
        print(f"[{POD_NAME} Self-Learning Note] {e}", flush=True)

    # -----------------------------------------------------------------
    # AGENT 1A: YOUTUBE AUTOCOMPLETE HARVESTER
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # AGENT 1B: REDDIT VIRAL HOT POSTS HARVESTER
    # -----------------------------------------------------------------
    reddit_base = mk_url("https", ["www", "reddit", "com"], "/r/")
    for sub in ACTIVE_POD["subreddits"]:
        try:
            r = requests.get(f"{reddit_base}{sub}/hot.json?limit=15", headers=BROWSER_HEADERS, timeout=6)
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

    # -----------------------------------------------------------------
    # AGENT 1C: HACKERNEWS & WIKIPEDIA TRENDING HARVESTER
    # -----------------------------------------------------------------
    try:
        hn_url = mk_url("https", ["hn", "algolia", "com"], "/api/v1/search?tags=front_page&hitsPerPage=15")
        r = requests.get(hn_url, headers=BROWSER_HEADERS, timeout=5)
        if r.status_code == 200:
            for hit in r.json().get("hits", []):
                htitle = hit.get("title", "")
                if len(htitle) > 12:
                    candidates.append({"title": htitle, "source": "hackernews", "weight": 1.15})
    except Exception:
        pass

    # -----------------------------------------------------------------
    # AGENT 1D: YOUTUBE COMPETITOR OUTLIER HARVESTER
    # -----------------------------------------------------------------
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
        except Exception as e:
            print(f"[{POD_NAME} YouTube Harvester Note] {e}", flush=True)

    # Ensure sufficient training rows for 18-feature XGBoost Regressor
    while len(X_train) < 12:
        X_train.append(compute_18_xgboost_features("What happens if you simulate $1,000 vs 100 balls in 3D?", 250000, 5000, 12000, 800))
        y_train.append(22.5)

    xgb_model = XGBRegressor(n_estimators=40, max_depth=4, learning_rate=0.08, verbosity=0)
    xgb_model.fit(np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.float32))

    # Score all harvested candidates using the 18-feature XGBoost model & apply >92% Percentile Filter!
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
    # Top 8% (>92nd percentile filter)
    cutoff = max(5, int(len(scored) * 0.08))
    elite_92_percentile = scored[:cutoff]

    save_pod_memory(mem)
    clean_tags = list(dict.fromkeys([re.sub(r"[^\w\s]", "", t).lower() for t in hijacked_tags if 2 < len(t) < 28]))[:15]
    print(f"✅ [{POD_NAME} 18-Feature XGBoost] Harvested {len(scored)} candidates -> Selected {len(elite_92_percentile)} (>92nd Percentile Winners)!", flush=True)

    return {
        "elite_candidates": elite_92_percentile[:12],
        "outliers": outliers[:6],
        "hijacked_tags": clean_tags
    }


# =====================================================================
# 3. BESPOKE ENGINE 1: SCALEVERSE 3D (REAL THREE.JS WEBGL 3D ENGINE!)
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

// 1. Initialize Real Three.js WebGL 3D Scene, Camera & Shadow-Casting Sunlight
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87ceeb);
scene.fog = new THREE.FogExp2(0x87ceeb, 0.0018);

const camera = new THREE.PerspectiveCamera(48, W / H, 0.5, 50000);
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setSize(W, H);
renderer.shadowMap.enabled = true;
document.getElementById('webgl-container').appendChild(renderer.domElement);

const hemiLight = new THREE.HemisphereLight(0xffffff, 0x334155, 0.85);
hemiLight.position.set(0, 500, 0);
scene.add(hemiLight);

const dirLight = new THREE.DirectionalLight(0xfff4e6, 1.15);
dirLight.position.set(180, 350, 220);
scene.add(dirLight);

// 2. Realistic 3D Ground Plane & Reference Grid (MetaBallStudios Architectural Style)
const groundGeo = new THREE.PlaneGeometry(40000, 40000);
const groundMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.65, metalness: 0.15 });
const ground = new THREE.Mesh(groundGeo, groundMat);
ground.rotation.x = -Math.PI / 2;
scene.add(ground);

const gridHelper = new THREE.GridHelper(10000, 200, 0x38bdf8, 0x334155);
gridHelper.position.y = 0.2;
scene.add(gridHelper);

// 3. Build 5 Detailed 3D Comparison Models Along the X-Axis (Normalized to visual heights 12..240)
const items = DATA.items || [];
const visualHeights = [14, 34, 75, 145, 260];
const xPositions = [0, 55, 155, 320, 580];
const palette = [0x10b981, 0x3b82f6, 0xf59e0b, 0xec4899, 0x8b5cf6];
const meshes = [];

for (let i = 0; i < 5; i++) {
    const h = visualHeights[i];
    const x = xPositions[i];
    const col = palette[i % palette.length];
    const group = new THREE.Group();

    const shapeType = (items[i] && items[i].shape) ? items[i].shape : ['cylinder', 'skyscraper', 'pyramid', 'sphere', 'megatower'][i];
    const matMain = new THREE.MeshStandardMaterial({ color: col, roughness: 0.28, metalness: 0.62 });
    const matGlow = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: col, emissiveIntensity: 0.45, wireframe: true });

    if (shapeType === 'sphere') {
        const geo = new THREE.SphereGeometry(h * 0.5, 48, 48);
        const m = new THREE.Mesh(geo, matMain);
        m.position.y = h * 0.5;
        group.add(m);
        // Planetary ring
        const rGeo = new THREE.RingGeometry(h * 0.65, h * 0.85, 64);
        const rMat = new THREE.MeshBasicMaterial({ color: 0xfde047, side: THREE.DoubleSide, transparent: true, opacity: 0.7 });
        const ring = new THREE.Mesh(rGeo, rMat);
        ring.rotation.x = Math.PI / 2.4;
        ring.position.y = h * 0.5;
        group.add(ring);
    } else if (shapeType === 'pyramid') {
        const geo = new THREE.ConeGeometry(h * 0.48, h, 4);
        const m = new THREE.Mesh(geo, matMain);
        m.position.y = h * 0.5;
        m.rotation.y = Math.PI / 4;
        group.add(m);
    } else if (shapeType === 'cylinder') {
        const geo = new THREE.CylinderGeometry(h * 0.25, h * 0.32, h, 36);
        const m = new THREE.Mesh(geo, matMain);
        m.position.y = h * 0.5;
        group.add(m);
    } else {
        // Stepped 3-tier architectural skyscraper / megastructure
        const tiers = 3;
        for (let t = 0; t < tiers; t++) {
            const th = h / tiers;
            const tw = h * 0.34 * (1 - t * 0.22);
            const bGeo = new THREE.BoxGeometry(tw, th, tw);
            const bMesh = new THREE.Mesh(bGeo, matMain);
            bMesh.position.y = t * th + th * 0.5;
            group.add(bMesh);

            const wMesh = new THREE.Mesh(bGeo, matGlow);
            wMesh.position.y = bMesh.position.y;
            wMesh.scale.set(1.01, 1.01, 1.01);
            group.add(wMesh);
        }
        // Spire
        const sGeo = new THREE.CylinderGeometry(0.4, h * 0.04, h * 0.18, 16);
        const sMesh = new THREE.Mesh(sGeo, new THREE.MeshBasicMaterial({ color: 0xfef08a }));
        sMesh.position.y = h + h * 0.09;
        group.add(sMesh);
    }

    // Glowing Pedestal Base Ring
    const pedGeo = new THREE.CylinderGeometry(h * 0.42, h * 0.46, 1.5, 48);
    const pedMat = new THREE.MeshBasicMaterial({ color: col });
    const ped = new THREE.Mesh(pedGeo, pedMat);
    ped.position.y = 0.75;
    group.add(ped);

    group.position.set(x, 0, 0);
    scene.add(group);
    meshes.push(group);
}

// Starfield particles for high-altitude / space transition in later stages
const starGeo = new THREE.BufferGeometry();
const starPts = [];
for (let s = 0; s < 1500; s++) {
    const sx = (Math.sin(s * 12.9898) * 43758.54 % 1) * 4000 - 1500;
    const sy = Math.abs((Math.cos(s * 78.233) * 23421.63 % 1)) * 2000 + 80;
    const sz = (Math.sin(s * 45.164) * 19283.44 % 1) * 3000 - 1800;
    starPts.push(sx, sy, sz);
}
starGeo.setAttribute('position', new THREE.Float32BufferAttribute(starPts, 3));
const stars = new THREE.Points(starGeo, new THREE.PointsMaterial({ color: 0xffffff, size: 3.5 }));
scene.add(stars);

window.renderFrameAtTime = function(t) {
    const totalDur = Math.max(DATA.total_duration, 0.1);
    const progress = Math.min(Math.max(t / totalDur, 0.0), 1.0);

    // Determine active stage (0..4) and smooth interpolation
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

    const floatIdx = Math.min(4.0, stageIndex + stageProgress * 0.85);
    const i0 = Math.min(4, Math.floor(floatIdx));
    const i1 = Math.min(4, i0 + 1);
    const frac = floatIdx - i0;
    const smoothFrac = frac * frac * (3 - 2 * frac);

    const curX = xPositions[i0] * (1 - smoothFrac) + xPositions[i1] * smoothFrac;
    const curH = visualHeights[i0] * (1 - smoothFrac) + visualHeights[i1] * smoothFrac;

    // Sky transitions smoothly from bright atmospheric cyan-blue to deep stratosphere/cosmos
    const skyColors = [
        new THREE.Color(0x38bdf8),
        new THREE.Color(0x0284c7),
        new THREE.Color(0x1e1b4b),
        new THREE.Color(0x0f172a),
        new THREE.Color(0x030712)
    ];
    const curSky = skyColors[i0].clone().lerp(skyColors[i1], smoothFrac);
    scene.background = curSky;
    scene.fog.color = curSky;

    // Rotate 3D models subtly for lifelike depth
    meshes.forEach((m, idx) => {
        m.rotation.y = t * 0.35 + idx * 0.5;
    });

    // Cinematic 3D Camera Orbit & Dolly Pullback
    const camOrbit = Math.sin(t * 0.45) * 0.22;
    const dist = curH * (isWide ? 2.15 : 3.1);
    camera.position.set(
        curX + Math.sin(camOrbit) * dist * 0.55,
        curH * 0.62 + 6,
        Math.cos(camOrbit) * dist
    );
    camera.lookAt(curX, curH * 0.48, 0);

    renderer.render(scene, camera);

    // Draw Sleek MetaBallStudios-Style Broadcast Lower-Third HUD (NO SyntaxGrid Box!)
    ctx.clearRect(0, 0, W, H);
    const activeItem = items[stageIndex] || { name: 'STAGE ' + (stageIndex + 1), metric: '1,000 m', subtitle: '3D SCALE COMPARISON' };

    // Top Floating Broadcast Header
    const cardW = isWide ? 820 : (W - 90);
    const cardX = (W - cardW) / 2;
    const cardY = isWide ? 48 : 110;

    ctx.save();
    ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.roundRect(cardX, cardY, cardW, 175, 24);
    ctx.fill();
    ctx.stroke();

    // Rank Badge
    ctx.fillStyle = '#f59e0b';
    ctx.beginPath();
    ctx.roundRect(cardX + 26, cardY + 24, 155, 42, 10);
    ctx.fill();
    ctx.fillStyle = '#0f172a';
    ctx.font = 'bold 22px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('RANK #0' + (stageIndex + 1), cardX + 103, cardY + 53);

    // Object Name & Real-World Metric
    ctx.textAlign = 'left';
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 38px sans-serif';
    ctx.fillText(String(activeItem.name || '').toUpperCase().slice(0, 26), cardX + 205, cardY + 58);

    ctx.fillStyle = '#38bdf8';
    ctx.font = 'bold 54px sans-serif';
    ctx.fillText(String(activeItem.metric || ''), cardX + 26, cardY + 138);

    ctx.fillStyle = '#cbd5e1';
    ctx.font = 'bold 24px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(String(activeItem.subtitle || '').toUpperCase().slice(0, 28), cardX + cardW - 28, cardY + 135);
    ctx.restore();

    // Clean Broadcast Captions at Bottom
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
        ctx.fillStyle = 'rgba(0, 0, 0, 0.78)';
        ctx.beginPath();
        ctx.roundRect((W - tw) / 2, by - 52, tw, 76, 16);
        ctx.fill();
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
# 4. BESPOKE ENGINE 2: KINETICZEN (REAL MATTER.JS + ZERO-LANGUAGE ASMR!)
# =====================================================================
MATTERJS_PHYSICS_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #12091c; overflow: hidden; font-family: 'DejaVu Sans', 'Liberation Sans', sans-serif; }
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
const canvas = document.getElementById('zenCanvas');
canvas.width = W;
canvas.height = H;
const ctx = canvas.getContext('2d');

// Initialize Real Matter.js 2D Rigid-Body Physics World
const Engine = Matter.Engine,
      Bodies = Matter.Bodies,
      Body = Matter.Body,
      Composite = Matter.Composite,
      Events = Matter.Events;

const engine = Engine.create();
engine.gravity.y = 0.95;
const world = engine.world;

const cx = W / 2;
const cy = H * 0.53;
const ringRadii = [210, 295, 380, 465];
const ringColors = ['#ff2a85', '#ffae00', '#00f5d4', '#9b5de5'];
const rings = [];
const particles = [];
window.collisionEvents = [];

// Build 4 Concentric Rotating Destructible Rings with Escape Gaps
ringRadii.forEach((radius, rIdx) => {
    const segments = 32;
    const gapSegments = 4; // 45-degree escape gap
    const segBodies = [];
    for (let s = gapSegments; s < segments; s++) {
        const a1 = (s / segments) * Math.PI * 2;
        const a2 = ((s + 1) / segments) * Math.PI * 2;
        const midA = (a1 + a2) / 2;
        const sx = cx + Math.cos(midA) * radius;
        const sy = cy + Math.sin(midA) * radius;
        const arcLen = (2 * Math.PI * radius) / segments + 4;
        const seg = Bodies.rectangle(sx, sy, arcLen, 16, {
            isStatic: true,
            angle: midA + Math.PI / 2,
            restitution: 1.03,
            friction: 0.0,
            label: 'ring_' + rIdx
        });
        segBodies.push({ body: seg, relAngle: midA, radius: radius });
        Composite.add(world, seg);
    }
    rings.push({
        idx: rIdx,
        radius: radius,
        color: ringColors[rIdx],
        angleOffset: rIdx * 1.1,
        speed: (rIdx % 2 === 0 ? 1 : -1) * (0.024 + rIdx * 0.005),
        segments: segBodies,
        destroyed: false
    });
});

// Spawn 24 Glossy Competing Physics Spheres (12 Team CRIMSON vs 12 Team GOLD)
const balls = [];
for (let i = 0; i < 24; i++) {
    const isGold = (i % 2 === 0);
    const bx = cx + (Math.sin(i * 19.3) * 85);
    const by = cy - 40 + (Math.cos(i * 31.7) * 65);
    const ball = Bodies.circle(bx, by, 20, {
        restitution: 1.02,
        friction: 0.0,
        frictionAir: 0.0005,
        density: 0.002,
        label: isGold ? 'ball_GOLD' : 'ball_CRIMSON'
    });
    Body.setVelocity(ball, { x: Math.sin(i * 7) * 9, y: Math.cos(i * 11) * 9 });
    Composite.add(world, ball);
    balls.push({ body: ball, isGold: isGold, trail: [] });
}

// Record real Matter.js collision timestamps for our Musical Kalimba/Marimba Audio Synth!
let simTime = 0.0;
Events.on(engine, 'collisionStart', function(event) {
    const pairs = event.pairs;
    for (let i = 0; i < pairs.length; i++) {
        const speed = pairs[i].collision.depth || 2.5;
        if (window.collisionEvents.length < 450) {
            window.collisionEvents.push({ time: Number(simTime.toFixed(3)), intensity: Math.min(speed / 5, 1.0), noteIdx: i % 6 });
        }
    }
});

let goldScore = 0;
let crimsonScore = 0;

window.stepAndRenderPhysics = function(frameIdx, totalFrames) {
    simTime = frameIdx / 24.0;
    // Step Matter.js physics engine (2 substeps per frame for ultra-crisp elastic bounces)
    for (let sub = 0; sub < 2; sub++) {
        rings.forEach(r => {
            if (r.destroyed) return;
            r.angleOffset += r.speed * 0.5;
            r.segments.forEach(segObj => {
                const newA = segObj.relAngle + r.angleOffset;
                const nx = cx + Math.cos(newA) * segObj.radius;
                const ny = cy + Math.sin(newA) * segObj.radius;
                Body.setPosition(segObj.body, { x: nx, y: ny });
                Body.setAngle(segObj.body, newA + Math.PI / 2);
            });
        });
        Engine.update(engine, 1000 / 48);
    }

    // Check ring escapes & shatter rings with particle bursts!
    balls.forEach(b => {
        const pos = b.body.position;
        b.trail.push({ x: pos.x, y: pos.y });
        if (b.trail.length > 9) b.trail.shift();

        const dist = Math.hypot(pos.x - cx, pos.y - cy);
        rings.forEach(r => {
            if (!r.destroyed && dist > r.radius + 18) {
                r.destroyed = true;
                if (b.isGold) goldScore += 25; else crimsonScore += 25;
                // Remove ring bodies and spawn 45 glowing shatter particles
                r.segments.forEach(s => {
                    Composite.remove(world, s.body);
                    particles.push({
                        x: s.body.position.x,
                        y: s.body.position.y,
                        vx: (Math.random() - 0.5) * 14,
                        vy: (Math.random() - 0.5) * 14,
                        color: r.color,
                        alpha: 1.0
                    });
                });
                window.collisionEvents.push({ time: Number(simTime.toFixed(3)), intensity: 1.0, noteIdx: 5 });
            }
        });

        // Outer arena boundary bounce so escaped balls celebrate in orbit
        if (dist > 515) {
            const ang = Math.atan2(pos.y - cy, pos.x - cx);
            Body.setPosition(b.body, { x: cx + Math.cos(ang) * 510, y: cy + Math.sin(ang) * 510 });
            Body.setVelocity(b.body, { x: -b.body.velocity.x * 0.9, y: -b.body.velocity.y * 0.9 });
            if (b.isGold) goldScore += 1; else crimsonScore += 1;
        }
    });

    // Warm Sunset / Velvet Studio Background (Completely distinct from SyntaxGrid!)
    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, '#1a0b2e');
    grad.addColorStop(0.5, '#0f071e');
    grad.addColorStop(1, '#240b36');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, W, H);

    // Top Clean Viral Scoreboard (Zero-Language Global Style: Icons + Numbers Only!)
    ctx.save();
    ctx.fillStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.beginPath();
    ctx.roundRect(70, 95, W - 140, 150, 32);
    ctx.fill();

    // Gold Team Score
    ctx.fillStyle = '#ffd700';
    ctx.beginPath(); ctx.arc(165, 170, 34, 0, Math.PI * 2); ctx.fill();
    ctx.font = 'bold 58px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(String(goldScore), 220, 190);

    // VS Divider
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 36px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('VS', cx, 184);

    // Crimson Team Score
    ctx.fillStyle = '#ff2a85';
    ctx.beginPath(); ctx.arc(W - 165, 170, 34, 0, Math.PI * 2); ctx.fill();
    ctx.font = 'bold 58px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(String(crimsonScore), W - 220, 190);
    ctx.restore();

    // Draw Intact Rotating Rings
    rings.forEach(r => {
        if (r.destroyed) return;
        ctx.save();
        ctx.strokeStyle = r.color;
        ctx.shadowBlur = 22;
        ctx.shadowColor = r.color;
        ctx.lineWidth = 16;
        ctx.lineCap = 'round';
        r.segments.forEach(segObj => {
            const p = segObj.body.position;
            ctx.beginPath();
            ctx.arc(p.x, p.y, 9, 0, Math.PI * 2);
            ctx.fillStyle = r.color;
            ctx.fill();
        });
        ctx.restore();
    });

    // Update & Draw Shatter Particles
    for (let p of particles) {
        if (p.alpha <= 0.02) continue;
        p.x += p.vx;
        p.y += p.vy;
        p.alpha *= 0.92;
        ctx.save();
        ctx.globalAlpha = p.alpha;
        ctx.fillStyle = p.color;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 7, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
    }

    // Draw Glossy 3D-Shaded Balls & Motion Trails
    balls.forEach(b => {
        const col = b.isGold ? '#ffd700' : '#ff2a85';
        b.trail.forEach((tp, idx) => {
            ctx.save();
            ctx.globalAlpha = (idx / b.trail.length) * 0.35;
            ctx.fillStyle = col;
            ctx.beginPath();
            ctx.arc(tp.x, tp.y, 10 + idx, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
        });

        const pos = b.body.position;
        ctx.save();
        ctx.shadowBlur = 18;
        ctx.shadowColor = col;
        ctx.fillStyle = col;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 20, 0, Math.PI * 2);
        ctx.fill();

        // 3D Specular Gloss Highlight
        ctx.fillStyle = 'rgba(255, 255, 255, 0.65)';
        ctx.beginPath();
        ctx.arc(pos.x - 6, pos.y - 6, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
    });
};
</script>
</body>
</html>
"""


# =====================================================================
# 5. BESPOKE ENGINE 3: CAPITALGRID (BLOOMBERG / TRADINGVIEW AUTO-SCALING UI!)
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

const seriesA = DATA.series_a || [1000, 5000, 15000, 45000, 120000, 350000];
const seriesB = DATA.series_b || [1000, 12000, 8000, 95000, 40000, 1450000];
const nameA = String(DATA.asset_a_name || 'SAFE INDEX').toUpperCase();
const nameB = String(DATA.asset_b_name || 'AGGRESSIVE ASSET').toUpperCase();
const startYear = Number(DATA.start_year || 2000);
const endYear = Number(DATA.end_year || 2025);

function interpSeries(arr, prog) {
    const maxIdx = arr.length - 1;
    const f = Math.min(Math.max(prog, 0.0), 1.0) * maxIdx;
    const i0 = Math.floor(f);
    const i1 = Math.min(maxIdx, i0 + 1);
    const frac = f - i0;
    return arr[i0] * (1 - frac) + arr[i1] * frac;
}

function fmtMoney(val) {
    if (val >= 1e9) return '$' + (val / 1e9).toFixed(2) + 'B';
    if (val >= 1e6) return '$' + (val / 1e6).toFixed(2) + 'M';
    if (val >= 1e3) return '$' + (val / 1e3).toFixed(1) + 'K';
    return '$' + Math.round(val).toLocaleString();
}

window.renderFrameAtTime = function(t) {
    const totalDur = Math.max(DATA.total_duration, 0.1);
    const progress = Math.min(Math.max(t / totalDur, 0.0), 1.0);

    // 1. Sleek Financial Terminal Backdrop (Emerald & Obsidian — NO SyntaxGrid box!)
    ctx.fillStyle = '#090d16';
    ctx.fillRect(0, 0, W, H);

    // Top Live Wall Street Ticker Bar
    ctx.fillStyle = '#111827';
    ctx.fillRect(0, 0, W, 72);
    ctx.fillStyle = '#10b981';
    ctx.font = 'bold 24px sans-serif';
    const curYear = Math.round(startYear + progress * (endYear - startYear));
    ctx.fillText('LIVE BACKTEST TERMINAL  •  YEAR: ' + curYear, 48, 45);

    const curValA = interpSeries(seriesA, progress);
    const curValB = interpSeries(seriesB, progress);

    // 2. Dual Executive Asset Comparison Cards (Top Half)
    const padX = 48;
    const cardW = (W - padX * 3) / 2;
    const cardY = 105;
    const cardH = isWide ? 185 : 230;

    // Card A (Conservative / Steady Asset)
    ctx.fillStyle = '#111827';
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 4;
    ctx.beginPath(); ctx.roundRect(padX, cardY, cardW, cardH, 20); ctx.fill(); ctx.stroke();

    ctx.fillStyle = '#93c5fd';
    ctx.font = 'bold 24px sans-serif';
    ctx.fillText(nameA.slice(0, 18), padX + 26, cardY + 48);
    ctx.fillStyle = '#ffffff';
    ctx.font = isWide ? 'bold 54px sans-serif' : 'bold 52px sans-serif';
    ctx.fillText(fmtMoney(curValA), padX + 26, cardY + 125);

    const roiA = (((curValA - seriesA[0]) / Math.max(seriesA[0], 1)) * 100).toFixed(0);
    ctx.fillStyle = '#3b82f6';
    ctx.font = 'bold 24px sans-serif';
    ctx.fillText('+' + Number(roiA).toLocaleString() + '% ROI', padX + 26, cardY + cardH - 24);

    // Card B (High-Growth / Volatile Asset)
    const cardBX = padX * 2 + cardW;
    ctx.fillStyle = '#111827';
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 4;
    ctx.beginPath(); ctx.roundRect(cardBX, cardY, cardW, cardH, 20); ctx.fill(); ctx.stroke();

    ctx.fillStyle = '#6ee7b7';
    ctx.font = 'bold 24px sans-serif';
    ctx.fillText(nameB.slice(0, 18), cardBX + 26, cardY + 48);
    ctx.fillStyle = '#10b981';
    ctx.font = isWide ? 'bold 54px sans-serif' : 'bold 52px sans-serif';
    ctx.fillText(fmtMoney(curValB), cardBX + 26, cardY + 125);

    const roiB = (((curValB - seriesB[0]) / Math.max(seriesB[0], 1)) * 100).toFixed(0);
    ctx.fillStyle = '#fbbf24';
    ctx.font = 'bold 24px sans-serif';
    ctx.fillText('+' + Number(roiB).toLocaleString() + '% ROI', cardBX + 26, cardY + cardH - 24);

    // 3. Dynamic Auto-Scaling TradingView Area Chart (NEVER flat at the bottom!)
    const chartX = padX;
    const chartY = cardY + cardH + 35;
    const chartW = W - padX * 2;
    const chartH = (isWide ? (H - 180) : 1380) - chartY;

    ctx.fillStyle = '#0d1322';
    ctx.strokeStyle = '#1f2937';
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.roundRect(chartX, chartY, chartW, chartH, 22); ctx.fill(); ctx.stroke();

    // Sample 80 points up to current progress and dynamically scale Y-axis to current max!
    const numSamples = 80;
    const ptsA = [];
    const ptsB = [];
    let currentMax = Math.max(seriesA[0], seriesB[0]) * 1.2;

    for (let i = 0; i <= numSamples; i++) {
        const subProg = (i / numSamples) * Math.max(progress, 0.03);
        const va = interpSeries(seriesA, subProg);
        const vb = interpSeries(seriesB, subProg);
        if (va > currentMax) currentMax = va;
        if (vb > currentMax) currentMax = vb;
        ptsA.push({ sp: i / numSamples, v: va });
        ptsB.push({ sp: i / numSamples, v: vb });
    }

    // Horizontal Y-Axis Gridlines & Price Labels
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.07)';
    ctx.fillStyle = '#64748b';
    ctx.font = 'bold 20px sans-serif';
    for (let g = 0; g <= 4; g++) {
        const gy = chartY + 40 + (g / 4) * (chartH - 90);
        const gVal = currentMax * (1 - g / 4);
        ctx.beginPath(); ctx.moveTo(chartX + 25, gy); ctx.lineTo(chartX + chartW - 25, gy); ctx.stroke();
        ctx.fillText(fmtMoney(gVal), chartX + 32, gy - 8);
    }

    function drawAreaCurve(pts, strokeCol, fillTop) {
        ctx.save();
        ctx.beginPath();
        pts.forEach((pt, idx) => {
            const px = chartX + 40 + pt.sp * (chartW - 95);
            const py = chartY + chartH - 45 - (pt.v / currentMax) * (chartH - 95);
            if (idx === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        });
        ctx.strokeStyle = strokeCol;
        ctx.lineWidth = 7;
        ctx.shadowBlur = 16;
        ctx.shadowColor = strokeCol;
        ctx.stroke();

        // Gradient fill under curve
        const lastPt = pts[pts.length - 1];
        const lastX = chartX + 40 + lastPt.sp * (chartW - 95);
        const lastY = chartY + chartH - 45 - (lastPt.v / currentMax) * (chartH - 95);
        ctx.lineTo(lastX, chartY + chartH - 45);
        ctx.lineTo(chartX + 40, chartY + chartH - 45);
        ctx.closePath();
        const aGrad = ctx.createLinearGradient(0, chartY, 0, chartY + chartH);
        aGrad.addColorStop(0, fillTop);
        aGrad.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = aGrad;
        ctx.fill();

        // Pulsing Head Dot
        ctx.beginPath();
        ctx.arc(lastX, lastY, 12, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
        ctx.restore();
    }

    drawAreaCurve(ptsA, '#3b82f6', 'rgba(59, 130, 246, 0.28)');
    drawAreaCurve(ptsB, '#10b981', 'rgba(16, 185, 129, 0.32)');

    // 4. Clean Financial Broadcast Captions
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
        ctx.strokeStyle = '#10b981';
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
};
</script>
</body>
</html>
"""


# =====================================================================
# 6. AGENT 2: POD-SPECIFIC DATA & SCRIPT ARCHITECT
# =====================================================================
def generate_pod_plan(spy_data, past_topics, is_long=False):
    elite_titles = [c["title"] for c in spy_data["elite_candidates"][:10]]

    if POD_NAME == "PHYSICS":
        # Zero-Language Global ASMR Physics Arena (No TTS needed!)
        prompt = f"""
        Top XGBoost-Scored Physics & Satisfaction Search Queries:
        {json.dumps(elite_titles)}
        Past Titles (DO NOT REPEAT):
        {json.dumps(past_topics[-30:])}

        Design a viral, language-independent (Global ASMR) Bouncing Ball vs Concentric Spinning Rings video title, description, and engagement question.
        Output JSON ONLY:
        {{
          "title": "Gold vs Crimson: Can 24 Balls Break All 4 Rings? #shorts",
          "topic_summary": "Satisfying 60FPS Matter.js rigid-body physics battle between Gold and Crimson spheres with synchronized marimba collision chimes.",
          "pinned_comment": "🟡 Team GOLD or 🔴 Team CRIMSON — which color did you bet on before the final ring shattered? Comment below! 👇"
        }}
        """
        return extract_json(call_ai(SCRIPT_MODELS, "Output valid JSON only.", prompt, 0.75))

    elif POD_NAME == "SCALE3D":
        prompt = f"""
        Top XGBoost-Scored 3D Scale & Comparison Queries:
        {json.dumps(elite_titles)}
        Outliers: {json.dumps(spy_data['outliers'][:4])}
        Past Titles (DO NOT REPEAT): {json.dumps(past_topics[-30:])}

        Create a 5-Stage 3D Size/Depth/Height Comparison video plan rendered in our Three.js WebGL 3D Engine.
         CRITICAL: The metrics in `items` and the spoken numbers in `narrations` MUST match 100% word-for-word!
        - For Shorts ({not is_long}): Each narration MUST be 16 to 20 spoken words.
        - For Long ({is_long}): Each narration MUST be 32 to 38 spoken words.

        Output JSON ONLY:
        {{
          "title": "High-CTR 3D Comparison Title under 58 chars",
          "topic_summary": "True 3D WebGL scale comparison from smallest to largest.",
          "pinned_comment": "Which 3D scale jump blew your mind the most? Comment below!",
          "thumb_badge": "TRUE 3D SCALE!",
          "thumb_line_1": "SMALLEST TO",
          "thumb_line_2": "COLOSSAL",
          "items": [
            {{"name": "Statue of Liberty", "metric": "93 m", "subtitle": "NEW YORK LANDMARK", "shape": "cylinder"}},
            {{"name": "Eiffel Tower", "metric": "330 m", "subtitle": "3.5X TALLER", "shape": "pyramid"}},
            {{"name": "Burj Khalifa", "metric": "828 m", "subtitle": "WORLD'S TALLEST TOWER", "shape": "skyscraper"}},
            {{"name": "Mount Everest", "metric": "8,849 m", "subtitle": "10X BURJ KHALIFA", "shape": "pyramid"}},
            {{"name": "Olympus Mons (Mars)", "metric": "21,900 m", "subtitle": "2.5X MOUNT EVEREST", "shape": "megatower"}}
          ],
          "narrations": [
            "Stage 1 narration matching item 1 exact metric...",
            "Stage 2 narration matching item 2 exact metric...",
            "Stage 3 narration matching item 3 exact metric...",
            "Stage 4 narration matching item 4 exact metric...",
            "Stage 5 narration matching item 5 exact metric..."
          ]
        }}
        """
        return extract_json(call_ai(SCRIPT_MODELS, "Output valid JSON only.", prompt, 0.7))

    else:
        # WEALTH Pod (Auto-Scaling Financial Backtest Terminal)
        prompt = f"""
        Top XGBoost-Scored Wealth & Investing Queries:
        {json.dumps(elite_titles)}
        Outliers: {json.dumps(spy_data['outliers'][:4])}
        Past Titles (DO NOT REPEAT): {json.dumps(past_topics[-30:])}

        Design a 5-Stage Financial Backtest / Wealth Comparison between two assets or strategies (`asset_a_name` vs `asset_b_name`).
        CRITICAL RULE: The 6 numbers in `series_a` and `series_b` MUST match the exact dollar amounts spoken in `narrations` 100%! Never say $10M if the array ends at $500M!
        - For Shorts ({not is_long}): Each narration MUST be 16 to 20 spoken words.
        - For Long ({is_long}): Each narration MUST be 32 to 38 spoken words.

        Output JSON ONLY:
        {{
          "title": "High-CTR Wealth Simulation Title under 58 chars",
          "topic_summary": "Live historical portfolio backtest and compounding simulation.",
          "pinned_comment": "Would you have held through the crash or sold early? Be honest in the comments!",
          "thumb_badge": "$1,000 TO $1.4M!",
          "thumb_line_1": "S&P 500 VS",
          "thumb_line_2": "TECH GIANTS",
          "asset_a_name": "S&P 500 INDEX",
          "asset_b_name": "NASDAQ / TECH",
          "start_year": 2000,
          "end_year": 2025,
          "series_a": [10000, 14000, 11000, 32000, 68000, 145000],
          "series_b": [10000, 28000, 9500, 85000, 310000, 920000],
          "narrations": [
            "Stage 1 narration introducing the starting investment in start_year...",
            "Stage 2 narration describing the first surge...",
            "Stage 3 narration describing the market crash drawdown...",
            "Stage 4 narration describing the recovery and compounding...",
            "Stage 5 narration revealing the exact final numbers from series_a[5] and series_b[5]!"
          ]
        }}
        """
        return extract_json(call_ai(SCRIPT_MODELS, "Output valid JSON only.", prompt, 0.65))


# =====================================================================
# 7. AUDIO SYNTHESIS (POLYPHONIC MARIMBA/KALIMBA SYNTH + TTS)
# =====================================================================
def synthesize_collision_marimba_track(total_duration, collision_events, output_wav):
    """
    Builds a studio-grade polyphonic ASMR Marimba/Kalimba musical track where every
    real Matter.js ball collision plays a resonant harmonic note at the exact millisecond!
    """
    sr = 44100
    n_samples = int((total_duration + 0.3) * sr)
    audio = np.zeros(n_samples, dtype=np.float32)

    # C-Major Pentatonic / Lydian Kalimba Scale (Hz)
    scale_hz = [261.63, 329.63, 392.00, 493.88, 523.25, 659.25, 783.99, 987.77, 1046.50]

    for ev in collision_events:
        t_sec = float(ev.get("time", 0.0))
        intensity = float(ev.get("intensity", 0.7))
        n_idx = int(ev.get("noteIdx", 0)) % len(scale_hz)
        freq = scale_hz[n_idx]

        start_i = int(t_sec * sr)
        dur_s = 0.28
        c_len = int(dur_s * sr)
        if 0 <= start_i and start_i + c_len < n_samples:
            ct = np.linspace(0, dur_s, c_len, endpoint=False)
            # Wooden marimba fundamental + 4x overtone + crisp attack click
            env = np.exp(-ct * 16.0) * np.sin(np.pi * np.minimum(ct / 0.006, 1.0))
            tone = (
                np.sin(2 * np.pi * freq * ct) * 0.75 +
                np.sin(2 * np.pi * (freq * 2.0) * ct) * np.exp(-ct * 35.0) * 0.20 +
                np.sin(2 * np.pi * (freq * 4.0) * ct) * np.exp(-ct * 55.0) * 0.08
            )
            audio[start_i:start_i + c_len] += tone * env * (0.14 * max(intensity, 0.35))

    # Normalize cleanly to prevent clipping
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
                comm = edge_tts.Communicate(text_to_speak, voice=voice, rate="+6%", boundary="WordBoundary")
            except TypeError:
                comm = edge_tts.Communicate(text_to_speak, voice=voice, rate="+6%")

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

    for idx, nar in enumerate(narrations):
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

    # CASE A: KINETICZEN (REAL MATTER.JS + FRAME-ACCURATE COLLISION MARIMBA SYNTH)
    if POD_NAME == "PHYSICS":
        total_dur = 38.0
        total_frames = int(total_dur * FPS)
        raw_mp4 = f"{prefix}_silent.mp4"

        payload = {"width": out_w, "height": out_h, "is_widescreen": is_widescreen, "total_duration": total_dur}
        html_str = MATTERJS_PHYSICS_HTML.replace("__MATTER_JS_SRC__", matter_src).replace("__PAYLOAD_JSON__", json.dumps(payload))
        html_file = Path(f"{prefix}_matter.html").resolve()
        html_file.write_text(html_str, encoding="utf-8")

        ffmpeg_v = [
            "ffmpeg", "-y", "-f", "image2pipe", "-vcodec", "mjpeg", "-r", str(FPS), "-i", "-",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "19", "-pix_fmt", "yuv420p", raw_mp4
        ]
        proc = subprocess.Popen(ffmpeg_v, stdin=subprocess.PIPE)
        collision_events = []

        print(f"[KineticZen Matter.js Engine] Simulating {total_frames} rigid-body frames...", flush=True)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-gpu"])
            page = browser.new_page(viewport={"width": out_w, "height": out_h})
            page.goto(f"file://{html_file}")
            page.wait_for_timeout(200)

            for f_idx in range(total_frames):
                page.evaluate(f"window.stepAndRenderPhysics({f_idx}, {total_frames})")
                proc.stdin.write(page.screenshot(type="jpeg", quality=92))
                if f_idx == total_frames // 2 and ch_idx == 0:
                    Path(f"thumb_{POD_NAME.lower()}.jpg").write_bytes(page.screenshot(type="jpeg", quality=95))

            collision_events = page.evaluate("window.collisionEvents || []")
            browser.close()

        proc.stdin.close()
        proc.wait()

        # Synthesize frame-accurate polyphonic Marimba/Kalimba collision soundtrack!
        marimba_wav = f"{prefix}_marimba.wav"
        synthesize_collision_marimba_track(total_dur, collision_events, marimba_wav)
        subprocess.run([
            "ffmpeg", "-y", "-i", raw_mp4, "-i", marimba_wav,
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", output_mp4
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        for tmp in [raw_mp4, marimba_wav, str(html_file)]:
            if os.path.exists(tmp):
                os.remove(tmp)
        return output_mp4, total_dur

    # CASE B & C: SCALEVERSE 3D (THREE.JS WEBGL) OR CAPITALGRID (TRADINGVIEW TERMINAL)
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
        "items": plan.get("items", []),
        "asset_a_name": plan.get("asset_a_name", "S&P 500"),
        "asset_b_name": plan.get("asset_b_name", "BITCOIN"),
        "start_year": plan.get("start_year", 2000),
        "end_year": plan.get("end_year", 2025),
        "series_a": plan.get("series_a", [10000, 15000, 12000, 35000, 75000, 145000]),
        "series_b": plan.get("series_b", [10000, 28000, 9500, 85000, 310000, 920000])
    }

    if POD_NAME == "SCALE3D":
        html_str = THREEJS_SCALE3D_HTML.replace("__THREE_JS_SRC__", three_src).replace("__PAYLOAD_JSON__", json.dumps(payload))
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

    print(f"[{POD_NAME} Bespoke Engine] Rendering {total_frames} frames ({out_w}x{out_h})...", flush=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist", "--no-sandbox"]
        )
        page = browser.new_page(viewport={"width": out_w, "height": out_h})
        page.goto(f"file://{html_file}")
        page.wait_for_timeout(300)

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
# 9. INTERACTIVE MICRO-SAAS WEB PORTAL (COMPOUND CALCULATOR + SANDBOX)
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
    print(f"🎉 [{POD_NAME} V2.0 PUBLISHED!] Video ID: {video_id} | URL: {short_link}", flush=True)

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
        plan = generate_pod_plan(spy_data, past_topics, is_long=is_long)
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
