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
# 1. ISOLATED POD CREDENTIALS & LOW-TOKEN AI ROUTER
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
        "brand_badge": "CAPITALGRID LAB",
        "voice": "en-US-ChristopherNeural",
        "theme_color": "#00ff9d",
        "accent_color": "#ffd700",
        "autocomplete_seeds": [
            "what happens if you invest 1000",
            "why 99 percent of traders lose",
            "compound interest simulation",
            "how inflation secretly destroys",
            "the math behind billionaire wealth",
            "if you bought bitcoin or s&p 500"
        ],
        "yt_search_query": "money wealth investing simulation explained"
    },
    "PHYSICS": {
        "client_id": os.environ.get("PHYSICS_CLIENT_ID", "").strip(),
        "client_secret": os.environ.get("PHYSICS_CLIENT_SECRET", "").strip(),
        "refresh_token": os.environ.get("PHYSICS_REFRESH_TOKEN", "").strip(),
        "brand_badge": "KINETICZEN ARENA",
        "voice": "en-US-GuyNeural",
        "theme_color": "#00e5ff",
        "accent_color": "#ff2a6d",
        "autocomplete_seeds": [
            "satisfying ball escape physics",
            "100 balls vs spinning cage",
            "pendulum wave alignment",
            "marble race survival simulation",
            "bouncing ball multiplies on hit",
            "polyrhythm physics satisfaction"
        ],
        "yt_search_query": "satisfying physics ball simulation viral"
    },
    "SCALE3D": {
        "client_id": os.environ.get("SCALE3D_CLIENT_ID", "").strip(),
        "client_secret": os.environ.get("SCALE3D_CLIENT_SECRET", "").strip(),
        "refresh_token": os.environ.get("SCALE3D_REFRESH_TOKEN", "").strip(),
        "brand_badge": "SCALEVERSE 3D",
        "voice": "en-US-EricNeural",
        "theme_color": "#38bdf8",
        "accent_color": "#f59e0b",
        "autocomplete_seeds": [
            "size comparison 3d animation",
            "how deep is the ocean really",
            "speed comparison 3d",
            "largest megastructures ever built",
            "smallest to largest in the universe",
            "what happens at the bottom of mariana trench"
        ],
        "yt_search_query": "3d size depth speed comparison animation"
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

CODE_MODELS = [
    "deepseek-v4-flash",       # x0.10
    "qwen3.7-max",             # x0.20
    "deepseek-v4.1-flash",     # x0.30
    "kimi-k2.7-code",          # x1.50
    "gemini-3.8-flash"         # x1.00
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


def extract_js_code(text):
    text = text.strip()
    if "```javascript" in text:
        return text.split("```javascript")[1].split("```")[0].strip()
    if "```js" in text:
        return text.split("```js")[1].split("```")[0].strip()
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text


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
    return {"seed_idx": 0, "topics": [], "playlist_id": "", "published_videos": [], "xgb_training_rows": []}


def save_pod_memory(mem):
    Path(f"history_{POD_NAME.lower()}.json").write_text(json.dumps(mem, indent=2), encoding="utf-8")


# =====================================================================
# 2. XGBOOST VIRAL PREDICTION ENGINE + LIVE YOUTUBE OUTLIER HARVESTER
# =====================================================================
def extract_topic_features(title_str, views=50000, subs=10000, likes=2000, comments=150):
    """Extracts 8 numerical features for XGBoost viral scoring."""
    t_low = title_str.lower()
    has_num = 1.0 if re.search(r"\d+", t_low) else 0.0
    power_words = ["secret", "never", "simulation", "100", "1000", "vs", "why", "how", "impossible", "deep", "million", "billion"]
    pw_count = sum(1.0 for w in power_words if w in t_low)
    char_len = float(len(title_str))
    optimal_len_score = 1.0 - min(abs(char_len - 52.0) / 52.0, 1.0)
    outlier_ratio = min(float(views) / max(float(subs), 500.0), 500.0)
    eng_rate = min((float(likes) + float(comments) * 4.0) / max(float(views), 100.0), 0.25)
    return [has_num, pw_count, optimal_len_score, outlier_ratio, eng_rate, char_len]


def run_xgboost_viral_harvester():
    """
    Scrapes YouTube Live Autocomplete + Competitor Outliers, trains an XGBoost Regressor
    on real outlier engagement metrics, and scores all candidate topics!
    """
    mem = load_pod_memory()
    s_idx = mem.get("seed_idx", 0)
    seeds = ACTIVE_POD["autocomplete_seeds"]
    mem["seed_idx"] = s_idx + 1
    save_pod_memory(mem)

    autocomplete_queries = []
    ac_base = mk_url("http", ["suggestqueries", "google", "com"], "/complete/search?client=youtube&ds=yt&q=")
    for pref in seeds:
        try:
            r = requests.get(ac_base + urllib.parse.quote(pref), headers=BROWSER_HEADERS, timeout=6)
            if r.status_code == 200:
                for m in re.findall(r'\["([^"]+)"', r.text):
                    if len(m) > 8 and m not in autocomplete_queries:
                        autocomplete_queries.append(m)
        except Exception:
            pass

    outliers = []
    hijacked_tags = []
    X_train, y_train = [], []

    try:
        yt = get_youtube_client()
        s_resp = yt.search().list(q=ACTIVE_POD["yt_search_query"], part="id,snippet", type="video", order="viewCount", maxResults=10).execute()
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

                feats = extract_topic_features(title, views, subs, likes, comments)
                viral_score = math.log10(max(views, 10)) * (1.0 + min(outlier_mult / 10.0, 5.0))
                X_train.append(feats)
                y_train.append(viral_score)

                outliers.append({
                    "title": title,
                    "views": views,
                    "outlier_multiplier": f"{outlier_mult}x",
                    "xgb_viral_score": round(viral_score, 2)
                })
    except Exception as e:
        print(f"[{POD_NAME} XGBoost Spy Note] {e}", flush=True)

    # Train XGBoost Regressor on harvested market data (with synthetic priors if API returned < 4 rows)
    while len(X_train) < 8:
        X_train.append([1.0, 2.0, 0.9, 15.0, 0.06, 52.0])
        y_train.append(18.5)

    xgb_model = XGBRegressor(n_estimators=25, max_depth=3, learning_rate=0.1, verbosity=0)
    xgb_model.fit(np.array(X_train), np.array(y_train))

    # Rank autocomplete queries using trained XGBoost model
    scored_queries = []
    for q in autocomplete_queries:
        f_vec = np.array([extract_topic_features(q, views=120000, subs=8000, likes=5000, comments=300)])
        pred_score = float(xgb_model.predict(f_vec)[0])
        scored_queries.append((pred_score, q))

    scored_queries.sort(key=lambda x: x[0], reverse=True)
    top_queries = [q for _, q in scored_queries[:20]]
    outliers.sort(key=lambda x: x.get("xgb_viral_score", 0), reverse=True)
    clean_tags = list(dict.fromkeys([re.sub(r"[^\w\s]", "", t).lower() for t in hijacked_tags if 2 < len(t) < 28]))[:15]

    print(f"✅ [{POD_NAME} XGBoost Engine] Ranked {len(scored_queries)} queries & {len(outliers)} viral outliers!", flush=True)
    return {
        "top_xgb_queries": top_queries,
        "outliers": outliers[:6],
        "hijacked_tags": clean_tags
    }


# =====================================================================
# 3. POD-SPECIFIC AI BLUEPRINT & BESPOKE JS VISUAL CODE GENERATOR
# =====================================================================
def generate_pod_experiment_unit(spy_data, past_topics, unit_idx=0, is_long=False):
    pod_instructions = {
        "WEALTH": """
        NICHE: High-RPM Wealth, Investing, Market Crash & Compounding Simulations (CapitalGrid).
        VISUAL STYLE: Animated multi-line portfolio growth/crash charts (Green vs Red curves), live compounding dollar counters ($1,000 -> $1,450,000), probability bars, and financial telemetry.
        """,
        "PHYSICS": """
        NICHE: Viral 60FPS ASMR Physics, Bouncing Ball Escapes, Pendulum Waves & Survival Arenas (KineticZen).
        VISUAL STYLE: Neon rotating circular gravity cages with escape gaps, 80-120 colorful colliding balls, pendulum waves, live team scoreboards (CYAN vs GOLD), and satisfying kinetic particle explosions!
        """,
        "SCALE3D": """
        NICHE: 3D Size, Depth, Speed & Megastructure Comparisons (ScaleVerse 3D).
        VISUAL STYLE: Pseudo-3D perspective camera fly-through (using 3D-to-2D perspective projection `scale = fov / (fov + z)`), wireframe 3D grids, rising 3D monoliths/depths/planets, live altitude/depth/speed meters (e.g. 0m -> 11,000m Mariana Trench or Mach 1 -> Light Speed).
        """
    }[POD_NAME]

    ideation_prompt = f"""
    {pod_instructions}

    Top XGBoost-Ranked Search Queries Today:
    {json.dumps(spy_data['top_xgb_queries'][:15])}

    Top Viral YouTube Outliers:
    {json.dumps(spy_data['outliers'][:5], indent=2)}

    PAST TOPICS (DO NOT REPEAT):
    {json.dumps(past_topics[-35:])}

    TASK:
    Design ONE extraordinary, high-retention video for {POD_NAME} with 5 continuous stages (Stage 0 -> Stage 4).
    - If Shorts mode ({not is_long}): Each of the 5 narrations MUST be 16 to 20 spoken words.
    - If Long mode ({is_long}): Each of the 5 narrations MUST be 32 to 38 spoken words.
    - Speak the core search keywords out loud in Stage 0!

    Output JSON ONLY:
    {{
      "topic_summary": "1-sentence description of the exact simulation",
      "title": "High-CTR Viral Title under 60 chars",
      "pinned_comment": "Interactive question forcing viewers to comment their prediction or score!",
      "thumb_badge": "99% GET THIS WRONG!",
      "thumb_line_1": "WATCH THE",
      "thumb_line_2": "SIMULATION",
      "stage_banners": [
        "STAGE 1: SHORT UPPERCASE BANNER",
        "STAGE 2: SHORT UPPERCASE BANNER",
        "STAGE 3: SHORT UPPERCASE BANNER",
        "STAGE 4: SHORT UPPERCASE BANNER",
        "STAGE 5: SHORT UPPERCASE BANNER (FINAL REVEAL)"
      ],
      "visual_simulation_spec": "Detailed specification of what the HTML5 Canvas simulation should animate across Stages 0, 1, 2, 3, and 4.",
      "narrations": [
        "Stage 0 narration...",
        "Stage 1 narration...",
        "Stage 2 narration...",
        "Stage 3 narration...",
        "Stage 4 narration..."
      ]
    }}
    """
    plan = extract_json(call_ai(SCRIPT_MODELS, "Output valid JSON only.", ideation_prompt, 0.75))

    js_prompt = f"""
    Write the pure JavaScript function `window.customSimDraw` for our {POD_NAME} channel:
    Topic: "{plan.get('topic_summary')}"
    Visual Specification: "{plan.get('visual_simulation_spec')}"
    5 Stage Banners: {json.dumps(plan.get('stage_banners', []))}

    ```javascript
    window.customSimDraw = function(ctx, t, progress, stageIndex, stageProgress, W, H, box) {{
        // Draw inside box.x, box.y, box.w, box.h (center is box.cx, box.cy)
    }};
    ```
    RULES:
    1. Must be 100% deterministic based on `t`, `progress` (0..1), `stageIndex` (0..4), and `stageProgress` (0..1).
    2. Use glowing neon visuals (`ctx.shadowBlur = 16`), bold live telemetry counters (`ctx.font = 'bold 34px sans-serif'`), and smooth 60fps math animations tailored to {POD_NAME}!
    3. Output ONLY valid JavaScript code defining `window.customSimDraw = ...`.
    """
    plan["custom_js"] = extract_js_code(call_ai(CODE_MODELS, "Output pure JavaScript only.", js_prompt, 0.55))
    return plan


# =====================================================================
# 4. PLAYWRIGHT CHROMIUM MULTI-ENGINE HARNESS
# =====================================================================
HTML_HARNESS = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #050811; overflow: hidden; font-family: 'DejaVu Sans', 'Liberation Sans', sans-serif; }
  canvas { display: block; }
</style>
</head>
<body>
<canvas id="simCanvas"></canvas>
<script>
const constPAYLOAD = __PAYLOAD_JSON__;
const canvas = document.getElementById('simCanvas');
const ctx = canvas.getContext('2d');
canvas.width = constPAYLOAD.width;
canvas.height = constPAYLOAD.height;

// Pod-specific built-in fallback engines (D3-style Wealth, Matter-style Physics, Three-style 3D Projection)
window.fallbackSimDraw = function(ctx, t, progress, stageIndex, stageProgress, W, H, box) {
    ctx.save();
    const pod = constPAYLOAD.pod_name;
    if (pod === 'WEALTH') {
        // Live Multi-Portfolio Compounding vs Crash Chart
        const steps = 60;
        const vis = Math.max(2, Math.floor(progress * steps));
        for (let p = 0; p < 18; p++) {
            ctx.beginPath();
            let val = 0.5;
            for (let s = 0; s < vis; s++) {
                const seed = Math.sin(p * 91.7 + s * 13.3) * 43758.54;
                const r = (seed - Math.floor(seed)) - 0.48;
                val = Math.max(0.05, Math.min(0.95, val + (p === 0 ? 0.012 : r * 0.045)));
                const x = box.x + 40 + (s / steps) * (box.w - 80);
                const y = box.y + box.h - 50 - val * (box.h - 120);
                if (s === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.strokeStyle = p === 0 ? '#00ff9d' : 'rgba(255, 60, 90, 0.45)';
            ctx.lineWidth = p === 0 ? 6 : 2;
            ctx.shadowBlur = p === 0 ? 15 : 0;
            ctx.shadowColor = '#00ff9d';
            ctx.stroke();
        }
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 38px sans-serif';
        const cash = Math.floor(1000 * Math.pow(1.18, progress * 40));
        ctx.fillText('PORTFOLIO VALUE: $' + cash.toLocaleString(), box.x + 50, box.y + 65);
    } else if (pod === 'SCALE3D') {
        // 3D Perspective Projection Tunnel & Scale Monoliths
        const fov = 350;
        for (let i = 0; i < 40; i++) {
            const z = ((i * 60 - t * 220) % 2400 + 2400) % 2400 + 40;
            const scale = fov / (fov + z);
            const rw = box.w * 0.85 * scale;
            const rh = box.h * 0.75 * scale;
            ctx.strokeStyle = `rgba(56, 189, 248, ${Math.min(1, scale * 1.4)})`;
            ctx.lineWidth = Math.max(1, 4 * scale);
            ctx.strokeRect(box.cx - rw / 2, box.cy - rh / 2, rw, rh);
        }
        ctx.fillStyle = '#f59e0b';
        ctx.font = 'bold 42px sans-serif';
        ctx.textAlign = 'center';
        const scaleVal = (Math.pow(10, progress * 6)).toFixed(0);
        ctx.fillText('SCALE MAGNITUDE: ' + Number(scaleVal).toLocaleString() + 'm', box.cx, box.y + 70);
    } else {
        // Rotating Gravity Ring & 70 Bouncing Balls
        const cageR = Math.min(box.w, box.h) * 0.36;
        ctx.strokeStyle = '#00e5ff';
        ctx.lineWidth = 10;
        ctx.beginPath();
        ctx.arc(box.cx, box.cy, cageR, t * 1.8 + 0.4, t * 1.8 + Math.PI * 2);
        ctx.stroke();
        for (let i = 0; i < 70; i++) {
            const a = t * (1.5 + (i % 7) * 0.3) + i * 2.1;
            const r = cageR * (0.85 * Math.abs(Math.sin(t * 1.3 + i)));
            const bx = box.cx + Math.cos(a) * r;
            const by = box.cy + Math.sin(a) * r;
            ctx.beginPath();
            ctx.arc(bx, by, 11, 0, Math.PI * 2);
            ctx.fillStyle = i % 2 === 0 ? '#ffd700' : '#ff2a6d';
            ctx.fill();
        }
    }
    ctx.restore();
};

try {
    __CUSTOM_AI_JS__
} catch (err) {
    console.error("Using pod fallback engine:", err);
}

function wrapText(ctx, text, maxWidth) {
    const words = String(text || '').toUpperCase().split(/\s+/);
    const lines = [];
    let current = '';
    for (let w of words) {
        const test = current ? (current + ' ' + w) : w;
        if (ctx.measureText(test).width <= maxWidth) {
            current = test;
        } else {
            if (current) lines.push(current);
            current = w;
        }
    }
    if (current) lines.push(current);
    return lines.slice(0, 2);
}

window.renderFrameAtTime = function(t) {
    const W = constPAYLOAD.width;
    const H = constPAYLOAD.height;
    const isWide = constPAYLOAD.is_widescreen;
    const totalDur = Math.max(constPAYLOAD.total_duration, 0.1);
    const progress = Math.min(Math.max(t / totalDur, 0.0), 1.0);

    const bgGrad = ctx.createRadialGradient(W/2, H/2, 50, W/2, H/2, Math.max(W, H) * 0.75);
    bgGrad.addColorStop(0, '#0d152b');
    bgGrad.addColorStop(1, '#04070f');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, W, H);

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let gx = 0; gx < W; gx += 72) {
        ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.stroke();
    }
    for (let gy = 0; gy < H; gy += 72) {
        ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke();
    }

    let stageIndex = 0;
    let stageProgress = 0.0;
    const timings = constPAYLOAD.scene_timings;
    for (let i = 0; i < timings.length; i++) {
        if (t >= timings[i].start && t <= timings[i].end + 0.05) {
            stageIndex = i;
            stageProgress = Math.min(Math.max((t - timings[i].start) / Math.max(timings[i].duration, 0.1), 0.0), 1.0);
            break;
        }
    }

    const padX = isWide ? 85 : 46;
    const topY = isWide ? 90 : 185;
    const botY = isWide ? (H - 175) : 1365;
    const arenaBox = {
        x: padX,
        y: topY + 105,
        w: W - padX * 2,
        h: (botY - (topY + 105)),
        cx: W / 2,
        cy: (topY + 105 + botY) / 2
    };

    ctx.save();
    ctx.fillStyle = 'rgba(10, 16, 32, 0.84)';
    ctx.strokeStyle = constPAYLOAD.theme_color;
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.roundRect(padX, topY, W - padX * 2, botY - topY, 28);
    ctx.fill();
    ctx.stroke();
    ctx.restore();

    ctx.save();
    ctx.fillStyle = constPAYLOAD.theme_color;
    ctx.beginPath();
    ctx.roundRect(padX, 32, 290, 44, 10);
    ctx.fill();
    ctx.fillStyle = '#050811';
    ctx.font = 'bold 20px sans-serif';
    ctx.fillText(constPAYLOAD.brand_badge, padX + 20, 61);

    ctx.fillStyle = 'rgba(18, 28, 56, 0.95)';
    ctx.beginPath();
    ctx.roundRect(padX + 22, topY + 18, W - (padX + 22) * 2, 82, 16);
    ctx.fill();

    const bannerRaw = (constPAYLOAD.stage_banners[stageIndex] || 'LIVE SIMULATION').toUpperCase();
    let fontSize = isWide ? 30 : 27;
    ctx.font = `bold ${fontSize}px sans-serif`;
    let lines = wrapText(ctx, bannerRaw, W - (padX + 45) * 2);
    ctx.fillStyle = stageIndex === 4 ? '#00ff9d' : constPAYLOAD.accent_color;
    ctx.textAlign = 'center';
    if (lines.length === 1) {
        ctx.fillText(lines[0], W / 2, topY + 69);
    } else {
        ctx.font = `bold ${fontSize - 4}px sans-serif`;
        ctx.fillText(lines[0], W / 2, topY + 52);
        ctx.fillText(lines[1], W / 2, topY + 84);
    }
    ctx.restore();

    ctx.save();
    ctx.beginPath();
    ctx.roundRect(arenaBox.x + 8, arenaBox.y + 4, arenaBox.w - 16, arenaBox.h - 12, 20);
    ctx.clip();
    try {
        if (typeof window.customSimDraw === 'function') {
            window.customSimDraw(ctx, t, progress, stageIndex, stageProgress, W, H, arenaBox);
        } else {
            window.fallbackSimDraw(ctx, t, progress, stageIndex, stageProgress, W, H, arenaBox);
        }
    } catch (simErr) {
        window.fallbackSimDraw(ctx, t, progress, stageIndex, stageProgress, W, H, arenaBox);
    }
    ctx.restore();

    const chunks = constPAYLOAD.chunks;
    let activeChunk = null;
    for (let c of chunks) {
        if (t >= c.start - 0.04 && t <= c.end + 0.08) {
            activeChunk = c;
            break;
        }
    }

    if (activeChunk) {
        ctx.save();
        const words = activeChunk.words;
        const spaceW = 22;
        const subFontNorm = isWide ? 'bold 44px sans-serif' : 'bold 46px sans-serif';
        const subFontAct = isWide ? 'bold 50px sans-serif' : 'bold 52px sans-serif';

        const measured = words.map(wObj => {
            const txt = String(wObj.word || '').toUpperCase();
            const isAct = (t >= wObj.start - 0.04 && t <= wObj.end + 0.06);
            ctx.font = isAct ? subFontAct : subFontNorm;
            return { txt, isAct, font: ctx.font, w: ctx.measureText(txt).width };
        });

        const subLines = [];
        let curLine = [];
        let curW = 0;
        for (let m of measured) {
            if (curLine.length > 0 && curW + spaceW + m.w > W - 130) {
                subLines.push(curLine);
                curLine = [m];
                curW = m.w;
            } else {
                curLine.push(m);
                curW += (curLine.length > 1 ? spaceW + m.w : m.w);
            }
        }
        if (curLine.length > 0) subLines.push(curLine);

        const baseY = isWide ? (H - 105) : (subLines.length > 1 ? 1475 : 1515);
        subLines.slice(0, 2).forEach((lineArr, lIdx) => {
            const totW = lineArr.reduce((acc, it) => acc + it.w, 0) + spaceW * Math.max(0, lineArr.length - 1);
            let xCur = (W - totW) / 2;
            const yCur = baseY + lIdx * 68;

            for (let item of lineArr) {
                ctx.font = item.font;
                ctx.textAlign = 'left';
                if (item.isAct) {
                    ctx.fillStyle = constPAYLOAD.accent_color;
                    ctx.strokeStyle = '#000000';
                    ctx.lineWidth = 4;
                    ctx.beginPath();
                    ctx.roundRect(xCur - 10, yCur - 46, item.w + 20, 62, 12);
                    ctx.fill();
                    ctx.stroke();
                    ctx.fillStyle = '#050811';
                    ctx.fillText(item.txt, xCur, yCur);
                } else {
                    ctx.strokeStyle = '#000000';
                    ctx.lineWidth = 8;
                    ctx.strokeText(item.txt, xCur, yCur);
                    ctx.fillStyle = '#ffffff';
                    ctx.fillText(item.txt, xCur, yCur);
                }
                xCur += item.w + spaceW;
            }
        });
        ctx.restore();
    }

    ctx.fillStyle = '#10182c';
    ctx.fillRect(0, H - 10, W, 10);
    ctx.fillStyle = constPAYLOAD.theme_color;
    ctx.fillRect(0, H - 10, Math.floor(progress * W), 10);
};
</script>
</body>
</html>
"""


# =====================================================================
# 5. AUDIO SYNTHESIS (WITH PENTATONIC ASMR CHIMES FOR PHYSICS POD!)
# =====================================================================
def generate_pod_soundtrack(total_duration, output_wav):
    sr = 44100
    n_samples = int((total_duration + 0.4) * sr)
    audio = np.zeros(n_samples, dtype=np.float32)

    # Ambient harmonic pad
    chord_freqs = [(110.0, 130.8, 164.8), (87.3, 110.0, 130.8), (130.8, 164.8, 196.0), (98.0, 123.5, 146.8)]
    seg_len = int(6.0 * sr)
    for start_idx in range(0, n_samples, seg_len):
        end_idx = min(start_idx + seg_len, n_samples)
        cur_len = end_idx - start_idx
        ct = np.linspace(0, cur_len / sr, cur_len, endpoint=False)
        f1, f2, f3 = chord_freqs[(start_idx // seg_len) % len(chord_freqs)]
        env = np.sin(np.pi * ct / max(cur_len / sr, 0.1)) ** 0.7
        audio[start_idx:end_idx] += (np.sin(2 * np.pi * f1 * ct) * 0.015 + np.sin(2 * np.pi * f2 * ct) * 0.012 + np.sin(2 * np.pi * f3 * ct) * 0.009) * env

    # For KineticZen (PHYSICS pod), layer crystalline pentatonic collision chimes!
    if POD_NAME == "PHYSICS":
        pentatonic_hz = [523.25, 587.33, 659.25, 783.99, 880.00, 1046.50]
        rng = np.random.default_rng(seed=42)
        t_cursor = 0.25
        while t_cursor < total_duration - 0.3:
            idx = int(t_cursor * sr)
            c_len = int(0.18 * sr)
            if idx + c_len < n_samples:
                freq = float(rng.choice(pentatonic_hz))
                ct = np.linspace(0, 0.18, c_len, endpoint=False)
                env = np.exp(-ct * 22.0)
                audio[idx:idx + c_len] += np.sin(2 * np.pi * freq * ct) * env * 0.045
            t_cursor += float(rng.uniform(0.16, 0.38))

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
        text_to_speak = "Watch closely as the simulation unfolds."

    voices = [ACTIVE_POD["voice"], "en-US-ChristopherNeural", "en-US-GuyNeural"]
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


def render_playwright_chapter(ch_idx, plan, is_widescreen, total_dur, scene_timings, word_timings, voice_wav):
    out_w = 1920 if is_widescreen else 1080
    out_h = 1080 if is_widescreen else 1920
    prefix = f"{POD_NAME.lower()}_ch_{ch_idx}"
    output_mp4 = f"{prefix}.mp4"

    bgm_wav = f"{prefix}_bgm.wav"
    generate_pod_soundtrack(total_dur, bgm_wav)

    mixed_audio = f"{prefix}_mixed.m4a"
    subprocess.run([
        "ffmpeg", "-y", "-i", voice_wav, "-i", bgm_wav,
        "-filter_complex", "[0:a]volume=1.55[v];[1:a]volume=0.65[b];[v][b]amix=inputs=2:duration=first:dropout_transition=2[aout]",
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
        "pod_name": POD_NAME,
        "brand_badge": ACTIVE_POD["brand_badge"],
        "theme_color": ACTIVE_POD["theme_color"],
        "accent_color": ACTIVE_POD["accent_color"],
        "width": out_w,
        "height": out_h,
        "is_widescreen": is_widescreen,
        "total_duration": total_dur,
        "scene_timings": scene_timings,
        "stage_banners": plan.get("stage_banners", [f"STAGE {i+1}" for i in range(5)]),
        "chunks": chunks
    }

    html_content = HTML_HARNESS.replace("__PAYLOAD_JSON__", json.dumps(payload)).replace("__CUSTOM_AI_JS__", plan.get("custom_js", ""))
    html_file = Path(f"{prefix}_sim.html").resolve()
    html_file.write_text(html_content, encoding="utf-8")

    total_frames = int(total_dur * FPS)
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "image2pipe", "-vcodec", "mjpeg", "-r", str(FPS), "-i", "-",
        "-i", mixed_audio,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", output_mp4
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    print(f"[{POD_NAME} Playwright Engine] Capturing {total_frames} frames ({out_w}x{out_h})...", flush=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--disable-gpu", "--no-sandbox"])
        page = browser.new_page(viewport={"width": out_w, "height": out_h})
        page.goto(f"file://{html_file}")
        page.wait_for_timeout(200)

        for f_idx in range(total_frames):
            t = f_idx / FPS
            page.evaluate(f"window.renderFrameAtTime({t:.4f})")
            proc.stdin.write(page.screenshot(type="jpeg", quality=90))

        if ch_idx == 0:
            page.evaluate(f"window.renderFrameAtTime({total_dur * 0.45:.4f})")
            Path(f"thumb_{POD_NAME.lower()}.jpg").write_bytes(page.screenshot(type="jpeg", quality=95))

        browser.close()

    proc.stdin.close()
    proc.wait()

    for tmp in [voice_wav, bgm_wav, mixed_audio, str(html_file)]:
        if os.path.exists(tmp):
            os.remove(tmp)

    return output_mp4, total_dur


# =====================================================================
# 6. AUTO-WEBSITE GENERATOR (docs/index.html) & YOUTUBE UPLOADER
# =====================================================================
def update_auto_website_portal(video_id, title, summary):
    """Automatically updates docs/index.html with interactive tools & latest video embeds!"""
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

    cards_html = ""
    yt_embed_base = mk_url("https", ["www", "youtube", "com"], "/embed/")
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
<title>AETHER Interactive Simulation & Wealth Lab</title>
<style>
  body {{ background: #060913; color: #f8fafc; font-family: system-ui, sans-serif; margin: 0; padding: 30px; }}
  h1 {{ text-align: center; color: #00e5ff; font-size: 2.2rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 24px; max-width: 1280px; margin: 30px auto; }}
  .card {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 16px; padding: 18px; }}
  .badge {{ background: #00e5ff; color: #060913; font-weight: 800; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; }}
  iframe {{ width: 100%; height: 220px; border: none; border-radius: 10px; margin-top: 12px; }}
</style>
</head>
<body>
  <h1>AETHER LIVE SIMULATION NETWORK</h1>
  <div class="grid">{cards_html}</div>
</body>
</html>"""
    Path("docs/index.html").write_text(html_page, encoding="utf-8")
    print(f"✅ [{POD_NAME} Web Portal] Updated docs/index.html with {video_id}!", flush=True)


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

    badge = clean_display_text(lead_plan.get("thumb_badge", "99% FAIL THIS!")).upper()[:18]
    l1 = clean_display_text(lead_plan.get("thumb_line_1", "WATCH THE")).upper()[:13]
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
    print(f"🎉 [{POD_NAME} PUBLISHED!] Video ID: {video_id} | URL: {short_link}", flush=True)

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
    update_auto_website_portal(video_id, title, description[:160])


def main():
    is_long = (VIDEO_MODE == "long")
    mem = load_pod_memory()
    past_topics = mem.get("topics", [])
    spy_data = run_xgboost_viral_harvester()

    num_chapters = 5 if is_long else 1
    chapter_files, chapter_timestamps, master_words = [], [], []
    elapsed_sec = 0.0
    lead_plan = None

    for idx in range(num_chapters):
        plan = generate_pod_experiment_unit(spy_data, past_topics, unit_idx=idx, is_long=is_long)
        if idx == 0:
            lead_plan = plan
        past_topics.append(plan.get("title", ""))

        nars = plan.get("narrations", [])
        dur, s_timings, w_timings, v_wav = asyncio.run(synthesize_chapter_audio(nars, prefix=f"{POD_NAME.lower()}_ch_{idx}"))
        for w in w_timings:
            master_words.append({"word": w["word"], "start": elapsed_sec + w["start"], "end": elapsed_sec + w["end"]})

        ch_mp4, dur = render_playwright_chapter(idx, plan, is_long, dur, s_timings, w_timings, v_wav)
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

    srt_file = build_seo_srt_file(master_words, f"captions_{POD_NAME.lower()}.srt")
    thumb_jpg = generate_pod_thumbnail(lead_plan, f"final_thumb_{POD_NAME.lower()}.jpg") if is_long else None
    tags = list(dict.fromkeys(spy_data["hijacked_tags"] + ["simulation", "experiment", POD_NAME.lower(), "viral"]))

    upload_pod_video(
        final_mp4,
        title=lead_plan.get("title", "Live Simulation Experiment"),
        description=f"{lead_plan.get('topic_summary', '')}\n\nSubscribe for daily {POD_NAME} simulations!",
        tags=tags,
        pinned_comment=lead_plan.get("pinned_comment", "What did you think of this simulation? Comment below!"),
        is_long=is_long,
        thumbnail_path=thumb_jpg,
        chapter_timestamps=chapter_timestamps,
        srt_path=srt_file
    )


if __name__ == "__main__":
    main()
