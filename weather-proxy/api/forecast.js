// Transparent proxy in front of Open-Meteo's Forecast API.
//
// Why this exists: the live-inference demo (src/app) runs on Render, whose free/shared outbound
// IP pool is used by many unrelated Render customers. Open-Meteo's free tier rate-limits by
// client IP (600/min, 5k/hour, 10k/day), so this app's own (tiny) traffic can still get 429'd by
// other tenants' usage of the same shared IP. Deploying this as its own Vercel function gives the
// forecast call a different, separately-rate-limited egress IP -- Vercel's pool is shared too, but
// with a different population of traffic than Render's.
//
// Deliberately dependency-free: forwards every query param as-is (lat/lon, start_date, end_date,
// hourly, models, timezone, ...) so it works for both the single-city and multi-city (comma
// separated lat/lon) request shapes without needing to know Open-Meteo's param list.
export default async function handler(req, res) {
  const params = new URLSearchParams(req.query);
  const upstream = `https://api.open-meteo.com/v1/forecast?${params.toString()}`;

  const upstreamResponse = await fetch(upstream);
  const body = await upstreamResponse.text();

  res.status(upstreamResponse.status);
  res.setHeader("content-type", upstreamResponse.headers.get("content-type") ?? "application/json");
  res.send(body);
}
