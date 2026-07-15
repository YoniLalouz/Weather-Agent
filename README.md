# Morning Weather Agent

A minimal agent: **tool call → Claude reasoning → action.**

## Setup (10 minutes)

1. **Create a GitHub repo** and push these files (`agent.py`, `.github/workflows/morning-weather.yml`).

2. **Install the ntfy app** on your phone (App Store / Play Store — free, no signup).
   - Open it, tap "+", subscribe to a topic name you invent — make it random/hard to guess,
     e.g. `yoni-weather-8f2k`. Anyone who knows your topic name can see your notifications,
     since ntfy.sh topics aren't private by default.

3. **Get an Anthropic API key**: console.anthropic.com → API Keys.

4. **Add repo secrets**: GitHub repo → Settings → Secrets and variables → Actions → New repository secret
   - `ANTHROPIC_API_KEY` → your key
   - `NTFY_TOPIC` → the topic name you picked, e.g. `yoni-weather-8f2k`
   - `CITY` → your city, e.g. `Tel Aviv`

5. **Test it**: Actions tab → "Morning Weather Agent" → "Run workflow" (manual trigger).
   You should get a push notification within ~30 seconds.

6. **Adjust the schedule**: edit the `cron` line in the workflow file. Cron times are UTC —
   figure out the UTC offset for your wake time and adjust.

## How this maps to the general agent pattern

| Step | In this project | General agent concept |
|---|---|---|
| Tool 1 | `geocode_city()` + `fetch_weather()` | External data retrieval |
| Reasoning | `summarize_with_claude()` | LLM call — turns raw data into judgment/output |
| Action | `send_push()` | Something that affects the outside world |
| Trigger | GitHub Actions cron | What kicks off the loop (could be a person, a schedule, an event) |

## Where to go next
- Add a second tool (e.g. calendar events) and have Claude decide what's worth mentioning
- Give Claude *choices* — e.g. skip the notification entirely if weather is unremarkable
- Add error handling / retry logic for a flaky API
