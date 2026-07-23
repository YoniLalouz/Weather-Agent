"""
Morning Weather + Hebrew Word Agent
------------------------------------
Agent loop:
  1. Tool call  -> fetch raw weather data (Open-Meteo, free, no API key)
  2. Reasoning  -> Claude writes a short weather briefing (Fahrenheit),
                   picks a Hebrew vocabulary word (bet+ level, with
                   present-tense m/f forms if it's a verb), and generates
                   a small custom cartoon SVG icon representing that
                   word's meaning
  3. Build      -> appends today's entry to docs/data.json (rolling
                   history) and renders docs/index.html as a vertically
                   scrolling page of daily cards, newest first, with an
                   animated cartoon SVG weather icon per day. Served
                   free via GitHub Pages.
  4. Action     -> push a notification to your phone (ntfy.sh) that opens
                   the page when tapped

Environment variables required (set as GitHub Actions secrets):
  ANTHROPIC_API_KEY  - your Claude API key
  NTFY_TOPIC         - a secret topic name you invent, e.g. "yoni-weather-8f2k"
  CITY               - city name to geocode, e.g. "Tel Aviv"
  PAGE_URL           - your GitHub Pages URL, e.g.
                        "https://yourusername.github.io/Weather-Agent/"
"""

import hashlib
import json
import os
import re
from datetime import date

import anthropic
import requests

CITY = os.environ.get("CITY", "Tel Aviv")
NTFY_TOPIC = os.environ["NTFY_TOPIC"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
PAGE_URL = os.environ.get("PAGE_URL", "")

DATA_PATH = "docs/data.json"
MAX_HISTORY_DAYS = 90  # keep the file from growing forever


def extract_text(message) -> str:
    """Claude's response can include non-text blocks (e.g. thinking) before
    the actual answer, so find the first text block instead of assuming
    content[0] is always text."""
    for block in message.content:
        if block.type == "text":
            return block.text.strip()
    raise ValueError("No text block found in Claude's response")


def geocode_city(city: str) -> dict:
    """Tool 1a: turn a city name into lat/lon."""
    resp = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
        timeout=10,
    )
    resp.raise_for_status()
    results = resp.json().get("results")
    if not results:
        raise ValueError(f"Could not geocode city: {city}")
    top = results[0]
    return {"lat": top["latitude"], "lon": top["longitude"], "name": top["name"]}


def fetch_weather(lat: float, lon: float) -> dict:
    """Tool 1b: fetch a 3-day high/low + condition forecast."""
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": (
                "temperature_2m_max,temperature_2m_min,"
                "precipitation_probability_max,weather_code"
            ),
            "temperature_unit": "fahrenheit",
            "timezone": "auto",
            "forecast_days": 3,
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["daily"]


def classify_condition(weather_code: int) -> str:
    """Map Open-Meteo's WMO weather code to a simple icon category."""
    if weather_code == 0:
        return "sunny"
    if weather_code in (1, 2):
        return "partly_cloudy"
    if weather_code == 3:
        return "cloudy"
    if weather_code in (45, 48):
        return "fog"
    if weather_code in (51, 53, 55, 56, 57, 61, 63, 65, 80, 81, 82):
        return "rain"
    if weather_code in (71, 73, 75, 77, 85, 86):
        return "snow"
    if weather_code in (95, 96, 99):
        return "storm"
    return "cloudy"


def summarize_with_claude(client: anthropic.Anthropic, city: str, daily: dict) -> str:
    """Reasoning step: Claude turns raw numbers into a short briefing."""
    raw = "\n".join(
        f"{d}: high {hi}\u00b0F, low {lo}\u00b0F, rain chance {rain}%"
        for d, hi, lo, rain in zip(
            daily["time"],
            daily["temperature_2m_max"],
            daily["temperature_2m_min"],
            daily["precipitation_probability_max"],
        )
    )
    message = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": (
                    f"You are a morning weather briefing assistant for {city}. "
                    f"Here is the raw 3-day forecast data (Fahrenheit):\n{raw}\n\n"
                    "Write a short (2-3 sentence), friendly push-notification-style "
                    "briefing covering today's high/low and anything notable "
                    "(rain, big swings). No markdown, no headers."
                ),
            }
        ],
    )
    return extract_text(message)


def hebrew_word_of_the_day(client: anthropic.Anthropic, recent_words: list) -> dict:
    """Reasoning step: Claude generates one new Hebrew vocabulary word,
    calibrated to intermediate (bet+) Ulpan level. If the word is a verb,
    also returns present-tense conjugations for masculine and feminine.
    Avoids repeating recently taught words."""
    avoid_note = ""
    if recent_words:
        avoid_note = (
            "Do not repeat any of these recently taught words: "
            + ", ".join(recent_words)
            + ".\n"
        )

    message = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=700,
        messages=[
            {
                "role": "user",
                "content": (
                    "Give me one new Hebrew vocabulary word for a language "
                    "learner at an intermediate level (Israeli Ulpan bet+ "
                    "level - comfortable with basics and simple conversation, "
                    "but not yet advanced). Pick something a bit beyond "
                    "beginner vocabulary - useful everyday verbs, adjectives, "
                    "or idioms rather than 'shalom'-tier basics. "
                    f"{avoid_note}"
                    "Respond with EXACTLY these 5 lines, no markdown, no "
                    "extra commentary, no labels, no numbering:\n"
                    "1) the word in Hebrew script with niqqud (dictionary/"
                    "infinitive form if it's a verb)\n"
                    "2) transliteration and English meaning\n"
                    "3) the part of speech - just one word: verb, noun, "
                    "adjective, or other\n"
                    "4) IF AND ONLY IF it's a verb: present tense forms "
                    "written as 'masc: <word> | fem: <word>' (both with "
                    "niqqud, transliteration in parentheses after each). "
                    "If it's not a verb, write exactly: N/A\n"
                    "5) one short example sentence in Hebrew, then its "
                    "English translation in parentheses"
                ),
            }
        ],
    )
    lines = [l.strip() for l in extract_text(message).splitlines() if l.strip()]
    while len(lines) < 5:
        lines.append("")
    return {
        "word": lines[0],
        "meaning": lines[1],
        "pos": lines[2],
        "conjugation": lines[3],
        "example": lines[4],
    }


SVG_FORBIDDEN_PATTERNS = ["<script", "onload=", "onerror=", "onclick=", "javascript:", "<foreignobject", "<image"]


def generate_word_illustration(client: anthropic.Anthropic, meaning: str, uid: str) -> str:
    """Reasoning step: Claude designs a small, unique, flat-cartoon-style
    SVG icon representing the word's meaning. Falls back to a simple
    generic icon if the response is missing or looks unsafe."""
    fallback = (
        f'<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">'
        f'<circle cx="100" cy="100" r="70" fill="#f0c869"/></svg>'
    )
    try:
        message = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=1800,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Design a small, friendly, flat-cartoon-style SVG icon "
                        f"that visually represents this concept: \"{meaning}\". "
                        "Think app-icon / sticker style: bold simple shapes, "
                        "soft gradients, rounded corners, a little personality "
                        "(e.g. a cute face, motion lines, or a playful detail) "
                        "if it fits naturally - but keep it simple and readable "
                        "at a small size.\n\n"
                        "Strict requirements:\n"
                        "- Root element: <svg viewBox=\"0 0 200 200\" "
                        "xmlns=\"http://www.w3.org/2000/svg\">\n"
                        "- Only use: circle, ellipse, rect, path, polygon, "
                        "line, g, linearGradient, radialGradient, stop, defs\n"
                        "- NO text elements, NO external images, NO <script>, "
                        "NO <foreignObject>, NO event handler attributes\n"
                        f"- Prefix every id you define with '{uid}_' so it "
                        "stays unique on a page with multiple icons\n"
                        "- Roughly 8-20 shape elements total\n"
                        "- Respond with ONLY the raw SVG markup, starting "
                        "with <svg and ending with </svg>. No markdown "
                        "fences, no explanation, nothing else."
                    ),
                }
            ],
        )
        svg = extract_text(message)
        # Strip markdown fences if the model added them anyway
        svg = re.sub(r"^```[a-zA-Z]*\n?", "", svg)
        svg = re.sub(r"\n?```$", "", svg).strip()

        lowered = svg.lower()
        if not lowered.startswith("<svg") or "</svg>" not in lowered:
            return fallback
        if any(pat in lowered for pat in SVG_FORBIDDEN_PATTERNS):
            return fallback
        return svg
    except Exception:
        return fallback


def seeded_hue(seed_str: str) -> int:
    """Turn a date string into a stable-for-the-day but different-each-day
    hue (0-359) so each day's card has a slightly different color scheme."""
    digest = hashlib.md5(seed_str.encode("utf-8")).hexdigest()
    return int(digest[:4], 16) % 360


# Static, detailed flat-illustration ("Canva sticker") style weather icons.
# Layered shapes, soft shading, and small decorative accents - no motion.
WEATHER_ICONS = {
    "sunny": """
        <g>
          <polygon points="100,4 112,50 100,40 88,50" fill="url(#rayGrad{i})"/>
          <polygon points="100,196 112,150 100,160 88,150" fill="url(#rayGrad{i})"/>
          <polygon points="4,100 50,112 40,100 50,88" fill="url(#rayGrad{i})"/>
          <polygon points="196,100 150,112 160,100 150,88" fill="url(#rayGrad{i})"/>
          <polygon points="30,30 66,54 52,56 44,66" fill="url(#rayGrad{i})"/>
          <polygon points="170,170 134,146 148,144 156,134" fill="url(#rayGrad{i})"/>
          <polygon points="170,30 134,54 148,56 156,66" fill="url(#rayGrad{i})"/>
          <polygon points="30,170 66,146 52,144 44,134" fill="url(#rayGrad{i})"/>
        </g>
        <ellipse cx="100" cy="150" rx="34" ry="8" fill="#000" opacity="0.07"/>
        <circle cx="100" cy="100" r="48" fill="url(#sunGrad{i})"/>
        <circle cx="86" cy="86" r="16" fill="#fff" opacity="0.25"/>
        <circle cx="87" cy="94" r="6" fill="#7a4a00"/>
        <circle cx="113" cy="94" r="6" fill="#7a4a00"/>
        <path d="M82 112 Q100 128 118 112" stroke="#7a4a00" stroke-width="5"
              fill="none" stroke-linecap="round"/>
        <circle cx="80" cy="102" r="6" fill="#ff9f7a" opacity="0.6"/>
        <circle cx="120" cy="102" r="6" fill="#ff9f7a" opacity="0.6"/>
    """,
    "partly_cloudy": """
        <ellipse cx="118" cy="150" rx="46" ry="9" fill="#000" opacity="0.07"/>
        <g>
          <polygon points="72,10 80,40 72,33 64,40" fill="url(#rayGrad{i})"/>
          <polygon points="30,52 58,60 51,52 58,44" fill="url(#rayGrad{i})"/>
          <polygon points="114,52 86,60 93,52 86,44" fill="url(#rayGrad{i})"/>
        </g>
        <circle cx="72" cy="72" r="30" fill="url(#sunGrad{i})"/>
        <circle cx="64" cy="64" r="10" fill="#fff" opacity="0.3"/>
        <g>
          <circle cx="85" cy="128" r="30" fill="url(#cloudGrad{i})"/>
          <circle cx="115" cy="118" r="38" fill="url(#cloudGrad{i})"/>
          <circle cx="145" cy="132" r="26" fill="url(#cloudGrad{i})"/>
          <ellipse cx="115" cy="148" rx="62" ry="20" fill="url(#cloudGrad{i})"/>
          <ellipse cx="100" cy="112" rx="20" ry="10" fill="#fff" opacity="0.35"/>
        </g>
        <circle cx="103" cy="132" r="5" fill="#5b6b8c"/>
        <circle cx="129" cy="132" r="5" fill="#5b6b8c"/>
        <path d="M105 145 Q116 153 127 145" stroke="#5b6b8c" stroke-width="4"
              fill="none" stroke-linecap="round"/>
        <circle cx="98" cy="140" r="4" fill="#ffb3a0" opacity="0.6"/>
        <circle cx="134" cy="140" r="4" fill="#ffb3a0" opacity="0.6"/>
    """,
    "cloudy": """
        <ellipse cx="100" cy="152" rx="58" ry="10" fill="#000" opacity="0.07"/>
        <circle cx="62" cy="118" r="26" fill="url(#cloudGrad{i})"/>
        <circle cx="90" cy="105" r="34" fill="url(#cloudGrad{i})"/>
        <circle cx="126" cy="112" r="32" fill="url(#cloudGrad{i})"/>
        <circle cx="152" cy="126" r="24" fill="url(#cloudGrad{i})"/>
        <ellipse cx="105" cy="140" rx="70" ry="22" fill="url(#cloudGrad{i})"/>
        <ellipse cx="85" cy="98" rx="22" ry="11" fill="#fff" opacity="0.35"/>
        <circle cx="88" cy="128" r="5" fill="#5b6b8c"/>
        <circle cx="116" cy="128" r="5" fill="#5b6b8c"/>
        <path d="M90 141 Q102 149 113 141" stroke="#5b6b8c" stroke-width="4"
              fill="none" stroke-linecap="round"/>
        <circle cx="82" cy="136" r="4" fill="#ffb3a0" opacity="0.5"/>
        <circle cx="122" cy="136" r="4" fill="#ffb3a0" opacity="0.5"/>
    """,
    "fog": """
        <ellipse cx="100" cy="152" rx="55" ry="9" fill="#000" opacity="0.06"/>
        <circle cx="75" cy="72" r="24" fill="url(#cloudGrad{i})"/>
        <circle cx="100" cy="64" r="28" fill="url(#cloudGrad{i})"/>
        <circle cx="125" cy="74" r="22" fill="url(#cloudGrad{i})"/>
        <ellipse cx="100" cy="86" rx="55" ry="16" fill="url(#cloudGrad{i})"/>
        <circle cx="88" cy="82" r="4" fill="#5b6b8c"/>
        <circle cx="112" cy="82" r="4" fill="#5b6b8c"/>
        <g fill="none" stroke="url(#cloudGrad{i})" stroke-width="9" stroke-linecap="round">
          <path d="M30 118 Q65 108 100 118 T170 118"/>
          <path d="M40 140 Q75 132 110 140 T175 140"/>
          <path d="M30 162 Q65 154 100 162 T165 162"/>
        </g>
    """,
    "rain": """
        <ellipse cx="100" cy="98" rx="55" ry="9" fill="#000" opacity="0.06"/>
        <circle cx="70" cy="62" r="26" fill="url(#cloudGrad{i})"/>
        <circle cx="98" cy="52" r="32" fill="url(#cloudGrad{i})"/>
        <circle cx="128" cy="64" r="24" fill="url(#cloudGrad{i})"/>
        <ellipse cx="98" cy="78" rx="62" ry="18" fill="url(#cloudGrad{i})"/>
        <ellipse cx="85" cy="52" rx="18" ry="9" fill="#fff" opacity="0.35"/>
        <circle cx="86" cy="72" r="5" fill="#5b6b8c"/>
        <circle cx="112" cy="72" r="5" fill="#5b6b8c"/>
        <path d="M84 87 Q97 80 110 87" stroke="#5b6b8c" stroke-width="4"
              fill="none" stroke-linecap="round"/>
        <path d="M62 120 Q56 138 62 150 Q68 138 62 120Z" fill="url(#rayGrad{i})"/>
        <path d="M100 128 Q94 146 100 158 Q106 146 100 128Z" fill="url(#rayGrad{i})"/>
        <path d="M138 120 Q132 138 138 150 Q144 138 138 120Z" fill="url(#rayGrad{i})"/>
    """,
    "snow": """
        <ellipse cx="100" cy="98" rx="55" ry="9" fill="#000" opacity="0.06"/>
        <circle cx="70" cy="62" r="26" fill="url(#cloudGrad{i})"/>
        <circle cx="98" cy="52" r="32" fill="url(#cloudGrad{i})"/>
        <circle cx="128" cy="64" r="24" fill="url(#cloudGrad{i})"/>
        <ellipse cx="98" cy="78" rx="62" ry="18" fill="url(#cloudGrad{i})"/>
        <ellipse cx="85" cy="52" rx="18" ry="9" fill="#fff" opacity="0.35"/>
        <circle cx="86" cy="72" r="5" fill="#5b6b8c"/>
        <circle cx="112" cy="72" r="5" fill="#5b6b8c"/>
        <path d="M84 87 Q97 80 110 87" stroke="#5b6b8c" stroke-width="4"
              fill="none" stroke-linecap="round"/>
        <g stroke="url(#rayGrad{i})" stroke-width="4" stroke-linecap="round">
          <g transform="translate(62,132)">
            <line x1="-10" y1="0" x2="10" y2="0"/><line x1="0" y1="-10" x2="0" y2="10"/>
            <line x1="-7" y1="-7" x2="7" y2="7"/><line x1="-7" y1="7" x2="7" y2="-7"/>
          </g>
          <g transform="translate(100,150)">
            <line x1="-10" y1="0" x2="10" y2="0"/><line x1="0" y1="-10" x2="0" y2="10"/>
            <line x1="-7" y1="-7" x2="7" y2="7"/><line x1="-7" y1="7" x2="7" y2="-7"/>
          </g>
          <g transform="translate(138,132)">
            <line x1="-10" y1="0" x2="10" y2="0"/><line x1="0" y1="-10" x2="0" y2="10"/>
            <line x1="-7" y1="-7" x2="7" y2="7"/><line x1="-7" y1="7" x2="7" y2="-7"/>
          </g>
        </g>
    """,
    "storm": """
        <ellipse cx="100" cy="96" rx="55" ry="9" fill="#000" opacity="0.07"/>
        <circle cx="70" cy="60" r="26" fill="url(#cloudGrad{i})"/>
        <circle cx="98" cy="50" r="32" fill="url(#cloudGrad{i})"/>
        <circle cx="128" cy="62" r="24" fill="url(#cloudGrad{i})"/>
        <ellipse cx="98" cy="76" rx="62" ry="18" fill="url(#cloudGrad{i})"/>
        <circle cx="84" cy="70" r="5" fill="#e9edf5"/>
        <circle cx="112" cy="70" r="5" fill="#e9edf5"/>
        <path d="M82 84 Q97 74 112 84" stroke="#e9edf5" stroke-width="4"
              fill="none" stroke-linecap="round"/>
        <polygon points="112,90 82,140 100,140 88,182 138,122 116,122"
                 fill="url(#rayGrad{i})"/>
        <polygon points="112,90 82,140 100,140 88,182 138,122 116,122"
                 fill="none" stroke="#fff" stroke-width="2" opacity="0.4"/>
    """,
}


def build_weather_icon_svg(condition: str, hue: int, uid: str) -> str:
    """Build an animated, illustrated SVG weather icon. uid keeps gradient
    IDs unique so multiple day-cards can coexist on one page."""
    inner = WEATHER_ICONS.get(condition, WEATHER_ICONS["cloudy"]).format(i=uid)
    return f"""
    <svg class="wx-icon" width="120" height="120" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="sunGrad{uid}" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="hsl({hue}, 90%, 65%)"/>
          <stop offset="100%" stop-color="hsl({(hue + 40) % 360}, 90%, 55%)"/>
        </linearGradient>
        <linearGradient id="cloudGrad{uid}" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="hsl({(hue + 200) % 360}, 60%, 92%)"/>
          <stop offset="100%" stop-color="hsl({(hue + 220) % 360}, 40%, 80%)"/>
        </linearGradient>
        <linearGradient id="rayGrad{uid}" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="hsl({(hue + 15) % 360}, 90%, 70%)"/>
          <stop offset="100%" stop-color="hsl({(hue + 45) % 360}, 90%, 58%)"/>
        </linearGradient>
      </defs>
      {inner}
    </svg>
    """


def render_day_card(entry: dict, index: int, is_today: bool) -> str:
    hue = entry["hue"]
    bg_start = f"hsl({hue}, 70%, 95%)"
    accent = f"hsl({(hue + 30) % 360}, 70%, 45%)"
    icon_svg = build_weather_icon_svg(entry["condition"], hue, f"d{index}")
    hebrew = entry["hebrew"]
    illustration = hebrew.get("illustration") or (
        '<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">'
        '<circle cx="100" cy="100" r="70" fill="#f0c869"/></svg>'
    )

    conjugation_html = ""
    if hebrew.get("conjugation") and hebrew["conjugation"].strip().upper() != "N/A":
        conjugation_html = f"""<div class="hebrew-conjugation">{hebrew['conjugation']}</div>"""

    today_badge = '<span class="today-badge">Today</span>' if is_today else ""

    return f"""
    <div class="card">
      <div class="card-header">
        <div>
          <h2>{entry['city']} {today_badge}</h2>
          <div class="date">{entry['date_label']}</div>
        </div>
        {icon_svg}
      </div>
      <div class="temps">{round(entry['hi'])}&deg;F <span class="lo">/ {round(entry['lo'])}&deg;F</span></div>
      <div class="briefing">{entry['briefing']}</div>
      <div class="hebrew-section" style="background: linear-gradient(135deg, {bg_start}, white);">
        <div class="hebrew-illustration">{illustration}</div>
        <div class="hebrew-word" style="color: {accent};">{hebrew['word']}</div>
        <div class="hebrew-meaning">{hebrew['meaning']} <span class="pos-tag">({hebrew.get('pos', '')})</span></div>
        {conjugation_html}
        <div class="hebrew-example">{hebrew['example']}</div>
      </div>
    </div>
    """


def render_html_page(history: list) -> str:
    """Render all entries (newest first) as a vertically scrolling page."""
    cards = "\n".join(
        render_day_card(entry, i, is_today=(i == 0))
        for i, entry in enumerate(history)
    )
    city = history[0]["city"] if history else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Daily Brief - {city}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fredoka:wght@500;700&family=Rubik:wght@400;700&display=swap" rel="stylesheet">
<style>
  body {{
    margin: 0;
    font-family: 'Fredoka', sans-serif;
    background: linear-gradient(160deg, #fdf6f0, #f2f0fb);
    padding: 24px 12px 60px;
  }}
  .page-title {{
    text-align: center;
    color: #444;
    font-size: 1.3rem;
    margin-bottom: 20px;
  }}
  .feed {{
    display: flex;
    flex-direction: column;
    gap: 22px;
    max-width: 460px;
    margin: 0 auto;
  }}
  .card {{
    background: white;
    border-radius: 26px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.10);
    padding: 24px 26px;
  }}
  .card-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}
  .card-header h2 {{
    margin: 0;
    font-size: 1.25rem;
    color: #333;
    display: inline;
  }}
  .today-badge {{
    display: inline-block;
    background: #ffe08a;
    color: #7a5b00;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 10px;
    margin-left: 8px;
    vertical-align: middle;
  }}
  .date {{
    color: #999;
    font-size: 0.85rem;
    margin-top: 2px;
  }}
  .temps {{
    font-size: 1.9rem;
    font-weight: 700;
    color: #333;
    margin: 6px 0;
  }}
  .temps span.lo {{ color: #999; font-weight: 500; }}
  .briefing {{
    color: #555;
    font-size: 0.95rem;
    line-height: 1.4;
    margin: 6px 0 18px;
  }}
  .hebrew-section {{
    border-radius: 18px;
    padding: 18px;
    text-align: center;
  }}
  .hebrew-illustration {{
    width: 110px;
    height: 110px;
    margin: 0 auto 6px;
  }}
  .hebrew-illustration svg {{ width: 100%; height: 100%; }}
  .hebrew-word {{
    font-family: 'Rubik', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    direction: rtl;
  }}
  .hebrew-meaning {{
    color: #555;
    font-size: 0.95rem;
    margin-top: 4px;
  }}
  .pos-tag {{
    color: #999;
    font-size: 0.8rem;
    font-style: italic;
  }}
  .hebrew-conjugation {{
    font-family: 'Rubik', sans-serif;
    color: #666;
    font-size: 0.85rem;
    margin-top: 8px;
    direction: rtl;
    text-align: center;
  }}
  .hebrew-example {{
    color: #777;
    font-size: 0.85rem;
    margin-top: 10px;
    font-style: italic;
  }}

  /* Weather icons are static flat-illustration style - no animation CSS needed */
  .wx-icon {{ overflow: visible; }}
</style>
</head>
<body>
  <div class="page-title">Your Daily Brief</div>
  <div class="feed">
    {cards}
  </div>
</body>
</html>"""


def load_history() -> list:
    if not os.path.exists(DATA_PATH):
        return []
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_history(history: list) -> None:
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def send_push(title: str, body: str, click_url: str = "") -> None:
    """Action step: push a notification to your phone via ntfy.sh.
    If click_url is set, tapping the notification opens that page."""
    safe_title = title.encode("ascii", "ignore").decode("ascii")
    headers = {"Title": safe_title}
    if click_url:
        headers["Click"] = click_url
    requests.post(
        f"https://ntfy.sh/{NTFY_TOPIC}",
        data=body.encode("utf-8"),
        headers=headers,
        timeout=10,
    )


def main():
    today = date.today()
    history = load_history()

    # If today's entry already exists, skip everything - this lets the
    # workflow run frequently (to land close to the target time even if
    # GitHub's scheduler is delayed) without wasting API calls or sending
    # duplicate notifications.
    if history and history[0].get("date_iso") == today.isoformat():
        print(f"Already generated today's brief ({today.isoformat()}) - skipping.")
        return

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    recent_words = [e["hebrew"]["word"] for e in history[:30] if e.get("hebrew", {}).get("word")]

    location = geocode_city(CITY)
    daily = fetch_weather(location["lat"], location["lon"])
    briefing = summarize_with_claude(client, location["name"], daily)
    hebrew = hebrew_word_of_the_day(client, recent_words)

    illustration = generate_word_illustration(client, hebrew["meaning"], f"w{today.isoformat()}")
    hebrew["illustration"] = illustration

    today_hi = daily["temperature_2m_max"][0]
    today_lo = daily["temperature_2m_min"][0]
    condition = classify_condition(daily["weather_code"][0])
    hue = seeded_hue(today.isoformat())

    entry = {
        "date_iso": today.isoformat(),
        "date_label": today.strftime("%A, %B %d"),
        "city": location["name"],
        "hi": today_hi,
        "lo": today_lo,
        "condition": condition,
        "hue": hue,
        "briefing": briefing,
        "hebrew": hebrew,
    }

    history.insert(0, entry)
    history = history[:MAX_HISTORY_DAYS]
    save_history(history)

    html = render_html_page(history)
    os.makedirs("docs", exist_ok=True)
    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write(html)

    combined = f"{briefing}\n\n\U0001F1EE\U0001F1F1 {hebrew['word']} - {hebrew['meaning']}"
    send_push(f"Morning Brief - {location['name']}", combined, click_url=PAGE_URL)
    print(combined)


if __name__ == "__main__":
    main()
