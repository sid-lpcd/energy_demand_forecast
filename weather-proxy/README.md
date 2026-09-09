# weather-proxy

A one-file Vercel serverless function that transparently proxies `api.open-meteo.com/v1/forecast`.

Why: the live demo (`src/app`, deployed on Render) was hitting Open-Meteo 429s in production
because Render's free-tier outbound traffic shares an IP pool with other Render customers, and
Open-Meteo rate-limits its free tier by client IP. Routing the live forecast call through a
function deployed on a different host gives it a different egress IP.

## Deploy (one-time)

From this directory:

```bash
npm i -g vercel   # if not already installed
vercel login      # interactive browser login -- run this yourself
vercel --prod     # deploys this directory as its own Vercel project
```

`vercel --prod` prints the deployed URL, e.g. `https://weather-proxy-xyz.vercel.app`. The live
endpoint is then `https://weather-proxy-xyz.vercel.app/api/forecast`.

## Wiring it into the app

Set `OPEN_METEO_LIVE_FORECAST_URL` to that endpoint in Render's dashboard (Environment tab) for
the `energy-demand-forecast` service. `src/edf/data/weather.py`'s live-forecast fetch reads this
env var and falls back to calling Open-Meteo directly when it's unset -- so local dev, tests, and
historical data pulls are unaffected.

No further code changes needed after redeploying Render with the env var set.
