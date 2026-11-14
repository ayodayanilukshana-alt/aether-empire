import os
import re
import sys
import json
import math
import time
import wave
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
            "what happens if you invest",
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
            "largest asteroids and craters 3d",
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


def strip_stage_prefix(text):
    """Guarantees the narrator NEVER says 'Stage 1', 'Stage two', 'Step 1', or 'Rank 1' out loud!"""
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
# 3. BESPOKE ENGINE 1: SCALEVERSE 3D (10 CATEGORY-MATCHED 3D GENERATORS!)
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

// Environment Themes (Space, Ocean, Sunset City, Cyber Grid, Golden Desert)
const ENV_THEMES = [
    { sky: [0x090d16, 0x050814, 0x030712, 0x02040a, 0x000005], ground: 0x0b132b, grid: 0x38bdf8 },
    { sky: [0x0284c7, 0x0369a1, 0x1d4ed8, 0x1e3a8a, 0x0f172a], ground: 0x0f172a, grid: 0x34d399 },
    { sky: [0x3b0764, 0x4c1d95, 0x311068, 0x1e1b4b, 0x090d16], ground: 0x18092c, grid: 0xf472b6 },
    { sky: [0x064e3b, 0x065f46, 0x047857, 0x0f172a, 0x022c22], ground: 0x052e16, grid: 0xfacc15 }
];
const env = ENV_THEMES[(DATA.theme_seed || 0) % ENV_THEMES.length];

const scene = new THREE.Scene();
scene.background = new THREE.Color(env.sky[0]);
scene.fog = new THREE.FogExp2(env.sky[0], 0.0011);

const camera = new THREE.PerspectiveCamera(45, W / H, 0.5, 50000);
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setSize(W, H);
renderer.shadowMap.enabled = true;
document.getElementById('webgl-container').appendChild(renderer.domElement);

const hemiLight = new THREE.HemisphereLight(0xffffff, 0x60a5fa, 1.25);
hemiLight.position.set(0, 600, 0);
scene.add(hemiLight);

const dirLight = new THREE.DirectionalLight(0xfffbeb, 1.45);
dirLight.position.set(220, 480, 320);
scene.add(dirLight);

const rimLight = new THREE.DirectionalLight(0x38bdf8, 0.85);
rimLight.position.set(-220, 250, -240);
scene.add(rimLight);

function createPatternTexture(baseHex, accentHex, mode) {
    const c = document.createElement('canvas');
    c.width = 256; c.height = 512;
    const g = c.getContext('2d');
    g.fillStyle = baseHex;
    g.fillRect(0, 0, 256, 512);
    if (mode === 'windows') {
        for (let y = 10; y < 500; y += 16) {
            for (let x = 10; x < 246; x += 20) {
                g.fillStyle = Math.sin(x * 12.3 + y * 45.6) > -0.15 ? accentHex : '#090d16';
                g.fillRect(x, y, 12, 10);
            }
        }
    } else if (mode === 'planet') {
        for (let y = 0; y < 512; y += 8) {
            g.fillStyle = (y % 32 < 16) ? accentHex : baseHex;
            g.globalAlpha = 0.35 + 0.25 * Math.sin(y * 0.05);
            g.fillRect(0, y, 256, 8);
        }
    } else {
        g.strokeStyle = accentHex;
        g.lineWidth = 3;
        for (let y = 0; y < 512; y += 24) {
            g.beginPath(); g.moveTo(0, y); g.lineTo(256, y); g.stroke();
        }
    }
    return new THREE.CanvasTexture(c);
}

const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(40000, 40000),
    new THREE.MeshStandardMaterial({ color: env.ground, roughness: 0.4, metalness: 0.3 })
);
ground.rotation.x = -Math.PI / 2;
scene.add(ground);

const gridHelper = new THREE.GridHelper(12000, 240, env.grid, 0x1e293b);
gridHelper.position.y = 0.2;
scene.add(gridHelper);

const items = DATA.items || [];
const rawMeters = items.map((it, idx) => Math.max(Number(it.meters) || (idx + 1) * 100, 0.1));
const minM = Math.min(...rawMeters);
const maxM = Math.max(...rawMeters);

const visualHeights = rawMeters.map((m, idx) => {
    if (maxM / minM > 35) {
        const ratio = Math.log10(m / minM + 1) / Math.log10(maxM / minM + 1);
        return 22 + ratio * 235 + idx * 14;
    }
    return Math.max(22, (m / maxM) * 255);
});

const xPositions = [0, 75, 185, 340, 560];
const palettes = [
    ['#065f46', '#34d399', 0x10b981],
    ['#1e3a8a', '#60a5fa', 0x3b82f6],
    ['#78350f', '#fbbf24', 0xf59e0b],
    ['#831843', '#f472b6', 0xec4899],
    ['#4c1d95', '#c084fc', 0xa855f7]
];

// 10 True Category-Matched 3D Model Builders!
function buildCategoryMatched3DModel(modelType, h, baseHex, glowHex, hexNum, idx) {
    const group = new THREE.Group();
    const matMetal = new THREE.MeshStandardMaterial({ color: hexNum, roughness: 0.22, metalness: 0.72 });
    const matGlow = new THREE.MeshBasicMaterial({ color: glowHex, wireframe: true, transparent: true, opacity: 0.32 });

    if (modelType === 'planet' || modelType === 'star') {
        const tex = createPatternTexture(baseHex, glowHex, 'planet');
        const matPlanet = new THREE.MeshStandardMaterial({
            map: tex,
            roughness: modelType === 'star' ? 0.1 : 0.5,
            emissive: modelType === 'star' ? hexNum : 0x000000,
            emissiveIntensity: modelType === 'star' ? 0.65 : 0.0
        });
        const r = h * 0.46;
        const sphere = new THREE.Mesh(new THREE.SphereGeometry(r, 48, 48), matPlanet);
        sphere.position.y = r + 4;
        group.add(sphere);

        if (idx % 2 === 1 || modelType === 'planet') {
            const ring = new THREE.Mesh(
                new THREE.RingGeometry(r * 1.25, r * 1.68, 64),
                new THREE.MeshBasicMaterial({ color: glowHex, side: THREE.DoubleSide, transparent: true, opacity: 0.65 })
            );
            ring.rotation.x = Math.PI / 2.3;
            ring.position.y = r + 4;
            group.add(ring);
        }
    } else if (modelType === 'rocket' || modelType === 'vehicle') {
        const bodyH = h * 0.68;
        const rad = h * 0.12;
        const body = new THREE.Mesh(new THREE.CylinderGeometry(rad, rad * 1.15, bodyH, 32), matMetal);
        body.position.y = bodyH * 0.5 + h * 0.08;
        group.add(body);

        const nose = new THREE.Mesh(new THREE.ConeGeometry(rad, h * 0.24, 32), new THREE.MeshStandardMaterial({ color: 0xfef08a, metalness: 0.6, roughness: 0.2 }));
        nose.position.y = bodyH + h * 0.20;
        group.add(nose);

        for (let f = 0; f < 4; f++) {
            const a = (f * Math.PI) / 2;
            const booster = new THREE.Mesh(new THREE.CylinderGeometry(rad * 0.35, rad * 0.45, h * 0.38, 16), matMetal);
            booster.position.set(Math.cos(a) * rad * 1.2, h * 0.20, Math.sin(a) * rad * 1.2);
            group.add(booster);
        }
    } else if (modelType === 'mountain' || modelType === 'trench' || modelType === 'crater') {
        const peak = new THREE.Mesh(new THREE.ConeGeometry(h * 0.52, h * 0.85, 7), matMetal);
        peak.position.y = h * 0.425;
        group.add(peak);
        const snow = new THREE.Mesh(new THREE.ConeGeometry(h * 0.19, h * 0.30, 7), new THREE.MeshStandardMaterial({ color: 0xf8fafc, roughness: 0.3 }));
        snow.position.y = h * 0.72;
        group.add(snow);
        const wire = new THREE.Mesh(new THREE.ConeGeometry(h * 0.53, h * 0.86, 7), matGlow);
        wire.position.y = h * 0.43;
        group.add(wire);
    } else if (modelType === 'pyramid') {
        const pyr = new THREE.Mesh(new THREE.ConeGeometry(h * 0.55, h, 4), matMetal);
        pyr.position.y = h * 0.5;
        pyr.rotation.y = Math.PI / 4;
        group.add(pyr);
        const wire = new THREE.Mesh(new THREE.ConeGeometry(h * 0.555, h, 4, 10), matGlow);
        wire.position.y = h * 0.5;
        wire.rotation.y = Math.PI / 4;
        group.add(wire);
    } else if (modelType === 'statue') {
        const p1 = new THREE.Mesh(new THREE.BoxGeometry(h * 0.38, h * 0.22, h * 0.38), matMetal);
        p1.position.y = h * 0.11; group.add(p1);
        const p2 = new THREE.Mesh(new THREE.CylinderGeometry(h * 0.18, h * 0.24, h * 0.16, 24), matMetal);
        p2.position.y = h * 0.30; group.add(p2);
        const torso = new THREE.Mesh(new THREE.CylinderGeometry(h * 0.08, h * 0.13, h * 0.44, 24), new THREE.MeshStandardMaterial({ color: 0xfbbf24, metalness: 0.85, roughness: 0.18 }));
        torso.position.y = h * 0.60; group.add(torso);
        const head = new THREE.Mesh(new THREE.SphereGeometry(h * 0.065, 24, 24), new THREE.MeshStandardMaterial({ color: 0xfbbf24, metalness: 0.85, roughness: 0.18 }));
        head.position.y = h * 0.88; group.add(head);
        const halo = new THREE.Mesh(new THREE.RingGeometry(h * 0.09, h * 0.13, 32), new THREE.MeshBasicMaterial({ color: 0xfef08a, side: THREE.DoubleSide }));
        halo.position.set(0, h * 0.90, -h * 0.03); group.add(halo);
    } else {
        // Skyscraper / Tower with illuminated windows
        const tex = createPatternTexture(baseHex, glowHex, 'windows');
        const matWin = new THREE.MeshStandardMaterial({ map: tex, roughness: 0.25, metalness: 0.55 });
        const tiers = 4 + (idx % 3);
        for (let s = 0; s < tiers; s++) {
            const sh = (h * 0.86) / tiers;
            const sw = h * 0.28 * (1 - s * (0.65 / tiers));
            const seg = new THREE.Mesh(new THREE.BoxGeometry(sw, sh, sw), matWin);
            seg.position.y = s * sh + sh * 0.5;
            seg.rotation.y = (idx % 2 === 1) ? s * 0.15 : 0;
            group.add(seg);
        }
        const spire = new THREE.Mesh(new THREE.CylinderGeometry(0.2, h * 0.02, h * 0.16, 16), new THREE.MeshBasicMaterial({ color: 0xfef08a }));
        spire.position.y = h * 0.94;
        group.add(spire);
    }

    const ped = new THREE.Mesh(new THREE.CylinderGeometry(h * 0.40, h * 0.44, 1.5, 48), new THREE.MeshBasicMaterial({ color: hexNum }));
    ped.position.y = 0.75;
    group.add(ped);
    return group;
}

const meshes = [];
for (let i = 0; i < 5; i++) {
    const h = visualHeights[i];
    const x = xPositions[i];
    const [baseHex, glowHex, hexNum] = palettes[i % palettes.length];
    const mType = (items[i] && items[i].model_type) ? String(items[i].model_type).toLowerCase() : 'skyscraper';
    const grp = buildCategoryMatched3DModel(mType, h, baseHex, glowHex, hexNum, i);
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

    const floatIdx = Math.min(4.0, stageIndex + Math.pow(stageProgress, 1.4) * 0.92);
    const i0 = Math.min(4, Math.floor(floatIdx));
    const i1 = Math.min(4, i0 + 1);
    const frac = floatIdx - i0;
    const smoothFrac = frac * frac * (3 - 2 * frac);

    const curX = xPositions[i0] * (1 - smoothFrac) + xPositions[i1] * smoothFrac;
    const curH = visualHeights[i0] * (1 - smoothFrac) + visualHeights[i1] * smoothFrac;

    const c0 = new THREE.Color(env.sky[i0]);
    const c1 = new THREE.Color(env.sky[i1]);
    const curSky = c0.lerp(c1, smoothFrac);
    scene.background = curSky;
    scene.fog.color = curSky;

    meshes.forEach((m, idx) => {
        m.rotation.y = t * 0.42 + idx * 0.5;
    });

    const camOrbit = Math.sin(t * 0.45) * 0.24;
    const dist = Math.max(curH * (isWide ? 2.15 : 3.05), 48);
    camera.position.set(
        curX + Math.sin(camOrbit) * dist * 0.55 - dist * 0.16,
        curH * 0.58 + 8,
        Math.cos(camOrbit) * dist
    );
    camera.lookAt(curX, curH * 0.48, 0);

    renderer.render(scene, camera);

    ctx.clearRect(0, 0, W, H);
    const activeItem = items[stageIndex] || { name: 'OBJECT ' + (stageIndex + 1), metric: '100 m', subtitle: '3D SCALE' };

    const cardW = isWide ? 820 : (W - 90);
    const cardX = (W - cardW) / 2;
    const cardY = isWide ? 48 : 95;

    ctx.save();
    ctx.fillStyle = 'rgba(15, 23, 42, 0.90)';
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.roundRect(cardX, cardY, cardW, 175, 24);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#f59e0b';
    ctx.beginPath();
    ctx.roundRect(cardX + 26, cardY + 24, 155, 42, 10);
    ctx.fill();
    ctx.fillStyle = '#0f172a';
    ctx.font = 'bold 22px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('RANK #0' + (stageIndex + 1), cardX + 103, cardY + 53);

    ctx.textAlign = 'left';
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 36px sans-serif';
    ctx.fillText(String(activeItem.name || '').toUpperCase().slice(0, 26), cardX + 205, cardY + 58);

    ctx.fillStyle = '#38bdf8';
    ctx.font = 'bold 54px sans-serif';
    ctx.fillText(String(activeItem.metric || ''), cardX + 26, cardY + 138);

    ctx.fillStyle = '#cbd5e1';
    ctx.font = 'bold 24px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(String(activeItem.subtitle || '').toUpperCase().slice(0, 26), cardX + cardW - 28, cardY + 135);
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
        ctx.fillStyle = 'rgba(15, 23, 42, 0.90)';
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
# 4. BESPOKE ENGINE 2: KINETICZEN (5 DISTINCT MATTER.JS MODES + 6 PALETTES!)
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
const canvas = document.getElementById('zenCanvas');
canvas.width = W;
canvas.height = H;
const ctx = canvas.getContext('2d');

// 6 Distinct Studio Color Palettes so every KineticZen video looks unique!
const PALETTES = [
    { bg1: '#1a0b2e', bg2: '#0d051a', teamA: '#ffd700', teamB: '#ff2a85', nameA: 'GOLD', nameB: 'CRIMSON', rings: ['#ff2a85', '#ffae00', '#00f5d4', '#38bdf8', '#a855f7', '#10b981'] },
    { bg1: '#041f1e', bg2: '#021014', teamA: '#00ff9d', teamB: '#ff6b00', nameA: 'EMERALD', nameB: 'SOLAR', rings: ['#00ff9d', '#00e5ff', '#ff6b00', '#facc15', '#38bdf8', '#a855f7'] },
    { bg1: '#091931', bg2: '#040b18', teamA: '#00e5ff', teamB: '#f43f5e', nameA: 'CYAN', nameB: 'ROSE', rings: ['#00e5ff', '#f43f5e', '#fbbf24', '#8b5cf6', '#10b981', '#ec4899'] },
    { bg1: '#230b1e', bg2: '#11040e', teamA: '#bef264', teamB: '#d946ef', nameA: 'LIME', nameB: 'MAGENTA', rings: ['#bef264', '#d946ef', '#06b6d4', '#f59e0b', '#ec4899', '#6366f1'] },
    { bg1: '#18181b', bg2: '#09090b', teamA: '#facc15', teamB: '#38bdf8', nameA: 'GOLD', nameB: 'ICE', rings: ['#facc15', '#38bdf8', '#f43f5e', '#4ade80', '#c084fc', '#fb923c'] },
    { bg1: '#171035', bg2: '#080518', teamA: '#ff4d6d', teamB: '#4cc9f0', nameA: 'NEON RED', nameB: 'AQUA', rings: ['#ff4d6d', '#4cc9f0', '#f72585', '#fee440', '#7209b7', '#06d6a0'] }
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
const cy = H * 0.55;
const rings = [];
const pegs = [];
const spinners = [];
const particles = [];
const balls = [];
window.collisionEvents = [];

// =====================================================================
// 5 DISTINCT PROCEDURAL PHYSICS ARENA MODES!
// =====================================================================
if (modeIdx === 0 || modeIdx === 3 || modeIdx === 4) {
    // Mode 0: 6 Concentric Rings | Mode 3: Growing Balls in Concentric Rings | Mode 4: Counter-Rotating Vortex Rings
    const ringRadii = [155, 215, 275, 335, 395, 455];
    ringRadii.forEach((radius, rIdx) => {
        const segments = 72;
        const gapSegments = modeIdx === 4 ? 9 : 7;
        const segBodies = [];
        for (let s = gapSegments; s < segments; s++) {
            const midA = ((s + 0.5) / segments) * Math.PI * 2;
            const sx = cx + Math.cos(midA) * radius;
            const sy = cy + Math.sin(midA) * radius;
            const arcLen = ((2 * Math.PI * radius) / segments) * 1.45;
            const seg = Bodies.rectangle(sx, sy, arcLen, 26, {
                isStatic: true,
                angle: midA + Math.PI / 2,
                restitution: 1.02,
                friction: 0.0
            });
            segBodies.push({ body: seg, relAngle: midA, radius: radius });
            Composite.add(world, seg);
        }
        rings.push({
            idx: rIdx,
            radius: radius,
            color: pal.rings[rIdx % pal.rings.length],
            angleOffset: rIdx * 1.05,
            speed: (rIdx % 2 === 0 ? 1 : -1) * (0.024 + rIdx * 0.004),
            segments: segBodies,
            escapedCount: 0,
            destroyed: false,
            isPolygon: false
        });
    });

    const nBalls = (modeIdx === 3) ? 10 : 22;
    for (let i = 0; i < nBalls; i++) {
        const isA = (i % 2 === 0);
        const angle = (i / nBalls) * Math.PI * 2;
        const r = 30 + (i % 3) * 24;
        const bRad = (modeIdx === 3) ? 13 : 17;
        const ball = Bodies.circle(cx + Math.cos(angle) * r, cy + Math.sin(angle) * r, bRad, {
            restitution: 1.02, friction: 0.0, frictionAir: 0.0002
        });
        Body.setVelocity(ball, { x: Math.cos(angle * 2.1) * 9, y: Math.sin(angle * 1.7) * 9 });
        Composite.add(world, ball);
        balls.push({ body: ball, isA: isA, rad: bRad, ringStage: 0, trail: [] });
    }
} else if (modeIdx === 1) {
    // Mode 1: Nested Spinning Hexagon / Octagon Polygon Tumbler Cage!
    const polyRadii = [180, 265, 355, 445];
    const sides = 6;
    polyRadii.forEach((radius, rIdx) => {
        const segBodies = [];
        for (let s = 0; s < sides; s++) {
            const a1 = (s / sides) * Math.PI * 2;
            const a2 = ((s + 1) / sides) * Math.PI * 2;
            const midA = (a1 + a2) / 2;
            const sideLen = 2 * radius * Math.sin(Math.PI / sides);
            // Leave an escape opening on side 0
            const actualLen = (s === 0) ? sideLen * 0.38 : sideLen * 1.06;
            const rDist = radius * Math.cos(Math.PI / sides);
            const sx = cx + Math.cos(midA) * rDist;
            const sy = cy + Math.sin(midA) * rDist;
            const seg = Bodies.rectangle(sx, sy, actualLen, 26, {
                isStatic: true, angle: midA + Math.PI / 2, restitution: 1.03, friction: 0.0
            });
            segBodies.push({ body: seg, relAngle: midA, radius: rDist });
            Composite.add(world, seg);
        }
        rings.push({
            idx: rIdx,
            radius: radius,
            color: pal.rings[rIdx % pal.rings.length],
            angleOffset: rIdx * 0.7,
            speed: (rIdx % 2 === 0 ? 1 : -1) * 0.022,
            segments: segBodies,
            escapedCount: 0,
            destroyed: false,
            isPolygon: true
        });
    });

    for (let i = 0; i < 24; i++) {
        const isA = (i % 2 === 0);
        const a = (i / 24) * Math.PI * 2;
        const ball = Bodies.circle(cx + Math.cos(a) * 55, cy + Math.sin(a) * 55, 17, {
            restitution: 1.02, friction: 0.0, frictionAir: 0.0002
        });
        Body.setVelocity(ball, { x: Math.cos(a) * 9, y: Math.sin(a) * 9 });
        Composite.add(world, ball);
        balls.push({ body: ball, isA: isA, rad: 17, ringStage: 0, trail: [] });
    }
} else {
    // Mode 2: Galton Plinko Pegboard & Spinning Windmill Royale!
    const rows = 8;
    for (let r = 0; r < rows; r++) {
        const cols = (r % 2 === 0) ? 7 : 6;
        const spacingX = 115;
        const startX = cx - ((cols - 1) * spacingX) / 2;
        const py = 380 + r * 115;
        for (let c = 0; c < cols; c++) {
            const px = startX + c * spacingX;
            const peg = Bodies.circle(px, py, 14, { isStatic: true, restitution: 0.95, friction: 0.0 });
            Composite.add(world, peg);
            pegs.push({ x: px, y: py, r: 14, color: pal.rings[r % pal.rings.length] });
        }
    }
    // Side walls
    Composite.add(world, Bodies.rectangle(75, H / 2, 30, H, { isStatic: true, restitution: 0.9 }));
    Composite.add(world, Bodies.rectangle(W - 75, H / 2, 30, H, { isStatic: true, restitution: 0.9 }));

    for (let i = 0; i < 32; i++) {
        const isA = (i % 2 === 0);
        const bx = cx + (Math.sin(i * 17.3) * 260);
        const by = 290 + (i % 5) * 18;
        const ball = Bodies.circle(bx, by, 18, { restitution: 0.92, friction: 0.001, frictionAir: 0.001 });
        Body.setVelocity(ball, { x: Math.cos(i * 5) * 4, y: 2 });
        Composite.add(world, ball);
        balls.push({ body: ball, isA: isA, rad: 18, ringStage: 0, trail: [] });
    }
}

let simTime = 0.0;
let lastNoteTime = -1.0;
let noteCounter = 0;
Events.on(engine, 'collisionStart', function() {
    if (simTime - lastNoteTime >= 0.072) {
        lastNoteTime = simTime;
        noteCounter++;
        window.collisionEvents.push({
            time: Number(simTime.toFixed(3)),
            intensity: 0.85,
            noteIdx: noteCounter % 9
        });
    }
});

let scoreA = 0;
let scoreB = 0;

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
        if (spd < 5.2 && modeIdx !== 2) {
            Body.setVelocity(b.body, { x: vel.x * 1.35 + (Math.random() - 0.5) * 2, y: vel.y * 1.35 - 1.5 });
        }

        // Mode 3: Balls subtly grow over time!
        if (modeIdx === 3 && b.rad < 32 && frameIdx % 18 === 0) {
            b.rad += 0.45;
            Body.scale(b.body, 1.025, 1.025);
        }

        b.trail.push({ x: pos.x, y: pos.y });
        if (b.trail.length > 8) b.trail.shift();

        if (modeIdx === 2) {
            // Plinko mode: respawn balls at top when they hit the bottom jackpot zone!
            if (pos.y > 1420) {
                if (b.isA) scoreA += 15; else scoreB += 15;
                for (let k = 0; k < 10; k++) {
                    particles.push({
                        x: pos.x, y: pos.y,
                        vx: (Math.random() - 0.5) * 14, vy: -Math.random() * 12,
                        color: b.isA ? pal.teamA : pal.teamB, alpha: 1.0
                    });
                }
                Body.setPosition(b.body, { x: cx + (Math.random() - 0.5) * 280, y: 285 });
                Body.setVelocity(b.body, { x: (Math.random() - 0.5) * 5, y: 2 });
            }
        } else {
            const dist = Math.hypot(pos.x - cx, pos.y - cy);
            if (activeRingIdx < rings.length) {
                const curRing = rings[activeRingIdx];
                if (b.ringStage <= activeRingIdx && dist > curRing.radius + 14) {
                    b.ringStage = activeRingIdx + 1;
                    curRing.escapedCount += 1;
                    if (b.isA) scoreA += 10; else scoreB += 10;

                    const minTime = (activeRingIdx + 1) * 4.5;
                    if (curRing.escapedCount >= 3 && simTime >= minTime) {
                        curRing.destroyed = true;
                        if (b.isA) scoreA += 25; else scoreB += 25;
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

    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, pal.bg1);
    grad.addColorStop(1, pal.bg2);
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, W, H);

    // Top Scoreboard
    ctx.save();
    ctx.fillStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.beginPath();
    ctx.roundRect(70, 95, W - 140, 150, 32);
    ctx.fill();

    ctx.fillStyle = pal.teamA;
    ctx.beginPath(); ctx.arc(155, 170, 32, 0, Math.PI * 2); ctx.fill();
    ctx.font = 'bold 56px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(String(scoreA), 205, 190);

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 30px sans-serif';
    ctx.textAlign = 'center';
    const centerLabel = (modeIdx === 2) ? 'JACKPOT RACE' : ('STAGE: ' + rings.filter(r => !r.destroyed).length);
    ctx.fillText(centerLabel, cx, 182);

    ctx.fillStyle = pal.teamB;
    ctx.beginPath(); ctx.arc(W - 155, 170, 32, 0, Math.PI * 2); ctx.fill();
    ctx.font = 'bold 56px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(String(scoreB), W - 205, 190);
    ctx.restore();

    // Draw Rings / Polygons / Plinko Pegs
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
                const len = (sIdx === 0) ? r.radius * 0.35 : r.radius * 0.95;
                ctx.beginPath();
                ctx.moveTo(p.x - Math.cos(ang) * len * 0.5, p.y - Math.sin(ang) * len * 0.5);
                ctx.lineTo(p.x + Math.cos(ang) * len * 0.5, p.y + Math.sin(ang) * len * 0.5);
                ctx.stroke();
            });
        } else {
            const gapFrac = (modeIdx === 4) ? (9 / 72) : (7 / 72);
            ctx.beginPath();
            ctx.arc(cx, cy, r.radius, r.angleOffset + gapFrac * Math.PI * 2, r.angleOffset + Math.PI * 2);
            ctx.stroke();
        }
        ctx.restore();
    });

    pegs.forEach(pg => {
        ctx.save();
        ctx.fillStyle = pg.color;
        ctx.shadowBlur = 14;
        ctx.shadowColor = pg.color;
        ctx.beginPath();
        ctx.arc(pg.x, pg.y, pg.r, 0, Math.PI * 2);
        ctx.fill();
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
        b.trail.forEach((tp, idx) => {
            ctx.save();
            ctx.globalAlpha = (idx / b.trail.length) * 0.35;
            ctx.fillStyle = col;
            ctx.beginPath(); ctx.arc(tp.x, tp.y, 7 + idx, 0, Math.PI * 2); ctx.fill();
            ctx.restore();
        });

        const pos = b.body.position;
        ctx.save();
        ctx.shadowBlur = 16;
        ctx.shadowColor = col;
        ctx.fillStyle = col;
        ctx.beginPath(); ctx.arc(pos.x, pos.y, b.rad, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = 'rgba(255, 255, 255, 0.72)';
        ctx.beginPath(); ctx.arc(pos.x - b.rad * 0.3, pos.y - b.rad * 0.3, b.rad * 0.28, 0, Math.PI * 2); ctx.fill();
        ctx.restore();
    });
};
</script>
</body>
</html>
"""


# =====================================================================
# 5. BESPOKE ENGINE 3: CAPITALGRID (MULTI-PALETTE + VOLUME BARS TERMINAL)
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
    { bg: '#090d16', colA: '#3b82f6', colB: '#10b981', fillA: 'rgba(59,130,246,0.26)', fillB: 'rgba(16,185,129,0.32)' },
    { bg: '#110c1d', colA: '#f59e0b', colB: '#ec4899', fillA: 'rgba(245,158,11,0.26)', fillB: 'rgba(236,72,153,0.32)' },
    { bg: '#07161b', colA: '#06b6d4', colB: '#facc15', fillA: 'rgba(6,182,212,0.26)', fillB: 'rgba(250,204,21,0.32)' },
    { bg: '#140a0e', colA: '#a855f7', colB: '#22c55e', fillA: 'rgba(168,85,247,0.26)', fillB: 'rgba(34,197,94,0.32)' }
];
const theme = FIN_THEMES[(DATA.theme_seed || 0) % FIN_THEMES.length];

const waypointsA = DATA.series_a || [1000, 2500, 4200, 6800, 11000, 18500];
const waypointsB = DATA.series_b || [1000, 4800, 1900, 16000, 45000, 142000];
const nameA = String(DATA.asset_a_name || 'ASSET A').toUpperCase();
const nameB = String(DATA.asset_b_name || 'ASSET B').toUpperCase();
const startYear = Number(DATA.start_year || 2005);
const endYear = Number(DATA.end_year || 2025);

function sampleMarketValue(waypoints, exactStageFloat, isVolatile) {
    const clamped = Math.min(Math.max(exactStageFloat, 0.0), 5.0);
    const i0 = Math.min(4, Math.floor(clamped));
    const i1 = Math.min(5, i0 + 1);
    const frac = clamped - i0;
    const baseVal = waypoints[i0] * (1 - frac) + waypoints[i1] * frac;
    const pinEnvelope = Math.sin(frac * Math.PI);
    const wave = (Math.sin(clamped * 26.0) * 0.042 + Math.cos(clamped * 63.0) * 0.024) * (isVolatile ? 1.5 : 0.65);
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

    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    ctx.fillStyle = 'rgba(17, 24, 39, 0.92)';
    ctx.fillRect(0, 0, W, 72);
    ctx.fillStyle = theme.colB;
    ctx.font = 'bold 24px sans-serif';
    const curYear = Math.round(startYear + normProg * (endYear - startYear));
    ctx.fillText('CAPITALGRID TERMINAL  •  YEAR: ' + curYear, 48, 45);

    const curValA = sampleMarketValue(waypointsA, exactStageFloat, false);
    const curValB = sampleMarketValue(waypointsB, exactStageFloat, true);

    const padX = 48;
    const cardW = (W - padX * 3) / 2;
    const cardY = 105;
    const cardH = isWide ? 185 : 230;

    // Card A
    ctx.fillStyle = '#111827';
    ctx.strokeStyle = theme.colA;
    ctx.lineWidth = 4;
    ctx.beginPath(); ctx.roundRect(padX, cardY, cardW, cardH, 20); ctx.fill(); ctx.stroke();

    ctx.fillStyle = '#cbd5e1';
    ctx.font = 'bold 24px sans-serif';
    ctx.fillText(nameA.slice(0, 18), padX + 26, cardY + 48);
    ctx.fillStyle = '#ffffff';
    ctx.font = isWide ? 'bold 54px sans-serif' : 'bold 52px sans-serif';
    ctx.fillText(fmtMoney(curValA), padX + 26, cardY + 125);

    const roiA = (((curValA - waypointsA[0]) / Math.max(waypointsA[0], 1)) * 100).toFixed(0);
    ctx.fillStyle = Number(roiA) >= 0 ? theme.colA : '#ef4444';
    ctx.font = 'bold 24px sans-serif';
    ctx.fillText((Number(roiA) >= 0 ? '+' : '') + Number(roiA).toLocaleString() + '% ROI', padX + 26, cardY + cardH - 24);

    // Card B
    const cardBX = padX * 2 + cardW;
    ctx.fillStyle = '#111827';
    ctx.strokeStyle = theme.colB;
    ctx.lineWidth = 4;
    ctx.beginPath(); ctx.roundRect(cardBX, cardY, cardW, cardH, 20); ctx.fill(); ctx.stroke();

    ctx.fillStyle = '#cbd5e1';
    ctx.font = 'bold 24px sans-serif';
    ctx.fillText(nameB.slice(0, 18), cardBX + 26, cardY + 48);
    ctx.fillStyle = theme.colB;
    ctx.font = isWide ? 'bold 54px sans-serif' : 'bold 52px sans-serif';
    ctx.fillText(fmtMoney(curValB), cardBX + 26, cardY + 125);

    const roiB = (((curValB - waypointsB[0]) / Math.max(waypointsB[0], 1)) * 100).toFixed(0);
    ctx.fillStyle = Number(roiB) >= 0 ? '#fbbf24' : '#ef4444';
    ctx.font = 'bold 24px sans-serif';
    ctx.fillText((Number(roiB) >= 0 ? '+' : '') + Number(roiB).toLocaleString() + '% ROI', cardBX + 26, cardY + cardH - 24);

    const chartX = padX;
    const chartY = cardY + cardH + 35;
    const chartW = W - padX * 2;
    const chartH = (isWide ? (H - 180) : 1380) - chartY;

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

    // Bottom Volume Histogram Bars inside Chart
    for (let b = 0; b < 36; b++) {
        const bx = chartX + 40 + (b / 36) * (chartW - 95);
        const bh = 12 + Math.abs(Math.sin(b * 3.7 + exactStageFloat * 4)) * 45;
        ctx.fillStyle = (b % 3 === 0) ? 'rgba(239, 68, 68, 0.22)' : 'rgba(16, 185, 129, 0.22)';
        ctx.fillRect(bx, chartY + chartH - 12 - bh, 14, bh);
    }

    function drawAreaCurve(pts, strokeCol, fillTop) {
        ctx.save();
        ctx.beginPath();
        pts.forEach((pt, idx) => {
            const px = chartX + 40 + pt.sp * (chartW - 95);
            const py = chartY + chartH - 55 - (pt.v / currentMax) * (chartH - 105);
            if (idx === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        });
        ctx.strokeStyle = strokeCol;
        ctx.lineWidth = 6;
        ctx.shadowBlur = 14;
        ctx.shadowColor = strokeCol;
        ctx.stroke();

        const lastPt = pts[pts.length - 1];
        const lastX = chartX + 40 + lastPt.sp * (chartW - 95);
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
        ctx.arc(lastX, lastY, 11, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
        ctx.restore();
    }

    drawAreaCurve(ptsA, theme.colA, theme.fillA);
    drawAreaCurve(ptsB, theme.colB, theme.fillB);

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
};
</script>
</body>
</html>
"""


# =====================================================================
# 6. AGENT 2: INFINITE VARIETY TOPIC WHEELS (NEVER REPEATS!)
# =====================================================================
SCALE3D_THEME_WHEEL = [
    ("Planets, Moons & Stars of the Universe (from Moon to UY Scuti)", "planet"),
    ("Fastest Machines, Jets & Spacecraft Ever Built (from Bugatti to Parker Solar Probe)", "rocket"),
    ("Deepest Ocean Trenches, Submarines & Shipwrecks (from Scuba Dive to Mariana Trench)", "trench"),
    ("Tallest Mountains & Volcanoes in the Solar System (from Fuji to Olympus Mons)", "mountain"),
    ("Colossal Asteroid Impacts & Craters Compared to Cities", "crater"),
    ("Ancient Wonders vs Modern Megastructures (from Giza Pyramid to Jeddah Tower)", "pyramid"),
    ("Largest Rockets & Spaceships in History (from Falcon 1 to Starship & Sea Dragon)", "rocket"),
    ("Biggest Cruise Ships, Supertankers & Aircraft Carriers Compared", "vehicle")
]

WEALTH_THEME_WHEEL = [
    "Gold vs Real Estate: Investing $5,000 over 30 Years",
    "$5/Day Coffee Habit vs Investing in Nvidia/Apple Stock over 15 Years",
    "High-Yield Savings Account vs Dividend Aristocrats Snowball ($10,000 Start)",
    "Cash Under the Mattress (Inflation Erosion) vs Gold Bullion over 40 Years",
    "Warren Buffett Value Strategy vs Day Trading Meme Stocks ($10,000 Simulation)",
    "Renting & Investing the Difference vs Buying a Luxury House (25-Year Math)",
    "1929 Great Depression Crash vs 2008 Financial Crisis Recovery Simulation",
    "Investing $100/Month Starting at Age 20 vs Starting at Age 35 (Compounding Secret)"
]

PHYSICS_MODE_DESCRIPTIONS = [
    (0, "6 Concentric Spinning Neon Rings Escape Battle"),
    (1, "Spinning Hexagon Tumbler Cage Physics Royale"),
    (2, "32-Ball Galton Plinko Pegboard Jackpot Race"),
    (3, "Growing Bouncing Balls Collision Arena (Balls Grow on Every Hit!)"),
    (4, "High-Speed Counter-Rotating Vortex Rings Escape")
]


def generate_pod_plan(spy_data, past_topics, unit_idx=0, is_long=False):
    run_idx = spy_data.get("run_idx", 0) + unit_idx
    elite_titles = [c["title"] for c in spy_data["elite_candidates"][:10]]
    word_rule = "13 to 16 spoken words" if not is_long else "30 to 35 spoken words"

    if POD_NAME == "PHYSICS":
        mode_id, mode_desc = PHYSICS_MODE_DESCRIPTIONS[run_idx % len(PHYSICS_MODE_DESCRIPTIONS)]
        prompt = f"""
        We are generating a viral, zero-language ASMR Physics Short using Mode: "{mode_desc}".
        Top XGBoost Search Queries: {json.dumps(elite_titles[:6])}
        PAST TITLES (DO NOT REPEAT ANY OF THESE): {json.dumps(past_topics[-35:])}

        Write a fresh, high-CTR title (under 58 chars with #shorts), summary, and interactive pinned comment matching "{mode_desc}".
        Output JSON ONLY:
        {{
          "title": "Unique Viral Title for {mode_desc} #shorts",
          "topic_summary": "60FPS Matter.js simulation of {mode_desc}.",
          "pinned_comment": "Which color won the {mode_desc}? Drop your prediction below! 👇"
        }}
        """
        plan = extract_json(call_ai(SCRIPT_MODELS, "Output valid JSON only.", prompt, 0.8))
        plan["physics_mode"] = mode_id
        plan["palette_seed"] = random.randint(0, 999)
        return plan

    elif POD_NAME == "SCALE3D":
        theme_desc, default_model = SCALE3D_THEME_WHEEL[run_idx % len(SCALE3D_THEME_WHEEL)]
        prompt = f"""
        MANDATORY 3D COMPARISON THEME FOR THIS VIDEO: "{theme_desc}"
        (FORBIDDEN: Do NOT use Statues, Christ the Redeemer, or Statue of Liberty unless the theme above explicitly asks for statues!)
        PAST TITLES (DO NOT REPEAT): {json.dumps(past_topics[-35:])}

        Create a 5-step 3D scale comparison for "{theme_desc}".
        CRITICAL RULES:
        1. Order the 5 `items` strictly from SMALLEST `meters` (Rank #1) to LARGEST `meters` (Rank #5).
        2. Set `"model_type"` for each item to one of: `"planet"`, `"star"`, `"rocket"`, `"mountain"`, `"trench"`, `"crater"`, `"pyramid"`, `"skyscraper"`, `"statue"`, `"vehicle"`.
        3. Write 5 natural `narrations` ({word_rule} each) matching the 5 sorted items.
        4. NEVER write "Stage 1", "Stage 2", "Step 1", or "Rank 1" in the narration text! Start directly with the fascinating hook or object name!

        Output JSON ONLY (replace placeholders with real facts & numbers):
        {{
          "title": "<Viral 3D Comparison Title under 58 chars>",
          "topic_summary": "<1-sentence summary>",
          "pinned_comment": "<Interactive question>",
          "thumb_badge": "<SHORT BADGE>",
          "thumb_line_1": "<WORD 1>",
          "thumb_line_2": "<WORD 2>",
          "items": [
            {{"name": "<Smallest Object>", "meters": 10, "metric": "<Formatted Metric>", "subtitle": "<2-3 WORD FACT>", "model_type": "{default_model}"}},
            {{"name": "<2nd Object>", "meters": 50, "metric": "<Formatted Metric>", "subtitle": "<2-3 WORD FACT>", "model_type": "{default_model}"}},
            {{"name": "<3rd Object>", "meters": 250, "metric": "<Formatted Metric>", "subtitle": "<2-3 WORD FACT>", "model_type": "{default_model}"}},
            {{"name": "<4th Object>", "meters": 1200, "metric": "<Formatted Metric>", "subtitle": "<2-3 WORD FACT>", "model_type": "{default_model}"}},
            {{"name": "<5th Largest Object>", "meters": 8000, "metric": "<Formatted Metric>", "subtitle": "<2-3 WORD FACT>", "model_type": "{default_model}"}}
          ],
          "narrations": [
            "<Natural narration for item 1 without saying Stage 1>",
            "<Natural narration for item 2 without saying Stage 2>",
            "<Natural narration for item 3 without saying Stage 3>",
            "<Natural narration for item 4 without saying Stage 4>",
            "<Natural narration for item 5 without saying Stage 5>"
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
        plan["theme_seed"] = random.randint(0, 999)
        return plan

    else:
        fin_theme = WEALTH_THEME_WHEEL[run_idx % len(WEALTH_THEME_WHEEL)]
        prompt = f"""
        MANDATORY FINANCIAL SIMULATION THEME FOR THIS VIDEO: "{fin_theme}"
        (FORBIDDEN: Do NOT repeat "S&P 500 vs Nasdaq"!)
        PAST TITLES (DO NOT REPEAT): {json.dumps(past_topics[-35:])}

        Create a 5-part financial backtest comparing two assets or habits based on: "{fin_theme}".
        CRITICAL RULES:
        1. `series_a` and `series_b` MUST each contain 6 numbers (`[initial_value, point_1, point_2, point_3, point_4, final_value]`).
        2. Each of the 5 `narrations` ({word_rule} each) MUST state the exact dollar figures in `series_a[i+1]` and `series_b[i+1]`.
        3. NEVER write the words "Stage 1", "Stage 2", "Stage 3", "Stage 4", or "Stage 5" in `narrations`! Speak like a natural human storyteller!

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
          "start_year": 2004,
          "end_year": 2026,
          "series_a": [5000, 6200, 7800, 9500, 12400, 16800],
          "series_b": [5000, 9400, 6100, 22000, 54000, 118000],
          "narrations": [
            "<Natural narration 1 without saying Stage 1>",
            "<Natural narration 2 without saying Stage 2>",
            "<Natural narration 3 without saying Stage 3>",
            "<Natural narration 4 without saying Stage 4>",
            "<Natural narration 5 without saying Stage 5>"
          ]
        }}
        """
        plan = extract_json(call_ai(SCRIPT_MODELS, "Output valid JSON only.", prompt, 0.78))
        plan["narrations"] = [strip_stage_prefix(n) for n in plan.get("narrations", [])]
        plan["theme_seed"] = random.randint(0, 999)
        return plan


# =====================================================================
# 7. AUDIO SYNTHESIS (4 MUSICAL SCALES/TIMBRES FOR PHYSICS + TTS)
# =====================================================================
def synthesize_collision_marimba_track(total_duration, collision_events, output_wav, seed_val=0):
    sr = 44100
    n_samples = int((total_duration + 0.3) * sr)
    audio = np.zeros(n_samples, dtype=np.float32)

    # Rotate across 4 musical scales so every KineticZen video has its own musical vibe!
     musical_scales = [
        [261.63, 329.63, 392.00, 493.88, 523.25, 659.25, 783.99, 987.77, 1046.50], # C Lydian Kalimba
        [293.66, 349.23, 440.00, 523.25, 587.33, 698.46, 880.00, 1046.50, 1174.66], # D Dorian Crystal Harp
        [220.00, 261.63, 329.63, 392.00, 440.00, 523.25, 659.25, 783.99, 880.00],  # A Minor Pentatonic Marimba
        [329.63, 349.23, 440.00, 493.88, 659.25, 698.46, 880.00, 987.77, 1318.51]  # Japanese Hirajoshi Zen Pluck
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

        print(f"[KineticZen Matter.js V3.0] Simulating Mode {payload['physics_mode']} ({total_frames} frames)...", flush=True)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-gpu"])
            page = browser.new_page(viewport={"width": out_w, "height": out_h})
            page.goto(f"file://{html_file}")
            page.wait_for_timeout(200)

            for f_idx in range(total_frames):
                page.evaluate(f"window.stepAndRenderPhysics({f_idx})")
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
        "start_year": plan.get("start_year", 2004),
        "end_year": plan.get("end_year", 2026),
        "series_a": plan.get("series_a", [5000, 6200, 7800, 9500, 12400, 16800]),
        "series_b": plan.get("series_b", [5000, 9400, 6100, 22000, 54000, 118000])
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

    print(f"[{POD_NAME} Bespoke Engine V3.0] Rendering {total_frames} frames ({out_w}x{out_h})...", flush=True)
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
    print(f"🎉 [{POD_NAME} V3.0 PUBLISHED!] Video ID: {video_id} | URL: {short_link}", flush=True)

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
