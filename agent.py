"""
Morning Weather Agent
----------------------
A minimal AI agent that demonstrates the core loop:
  1. Tool call  -> fetch raw weather data (Open-Meteo, free, no API key)
  2. Reasoning  -> Claude turns raw data into a short, friendly briefing
  3. Action     -> push the result to your phone via ntfy.sh

Environment variables required (set as GitHub Actions secrets):
  ANTHROPIC_API_KEY  - your Claude API key
  NTFY_TOPIC         - a secret topic name you invent, e.g. "yoni-weather-8f2k"
  CITY               - city name to geocode, e.g. "Tel Aviv"
"""

import os
import requests
import anthropic

CITY = os.environ.get("CITY", "Tel Aviv")
NTFY_TOPIC = os.environ["NTFY_TOPIC"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]


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
    """Tool 1b: fetch a 3-day high/low forecast."""
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": 3,
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["daily"]


def summarize_with_claude(city: str, daily: dict) -> str:
    """Reasoning step: Claude turns raw numbers into a short briefing."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    raw = "\n".join(
        f"{date}: high {hi}°C, low {lo}°C, rain chance {rain}%"
        for date, hi, lo, rain in zip(
            daily["time"],
            daily["temperature_2m_max"],
            daily["temperature_2m_min"],
            daily["precipitation_probability_max"],
        )
    )

    message = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": (
                    f"You are a morning weather briefing assistant for {city}. "
                    f"Here is the raw 3-day forecast data:\n{raw}\n\n"
                    "Write a short (2-3 sentence), friendly push-notification-style "
                    "briefing covering today's high/low and anything notable "
                    "(rain, big swings). No markdown, no headers."
                ),
            }
        ],
    )
    return message.content[0].text.strip()


def send_push(title: str, body: str) -> None:
    """Action step: push the briefing to your phone via ntfy.sh."""
    requests.post(
        f"https://ntfy.sh/{NTFY_TOPIC}",
        data=body.encode("utf-8"),
        headers={"Title": title},
        timeout=10,
    )


def main():
    location = geocode_city(CITY)
    daily = fetch_weather(location["lat"], location["lon"])
    briefing = summarize_with_claude(location["name"], daily)
    send_push(f"Weather — {location['name']}", briefing)
    print(briefing)  # also visible in GitHub Actions logs


if __name__ == "__main__":
    main()
