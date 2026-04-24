"""
main.py
Minimal ASGI entrypoint for Vercel deployment.
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse


app = FastAPI(title="AI Powered Data Analyst")


@app.get("/", response_class=HTMLResponse)
async def home() -> str:
    return """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>AI Powered Data Analyst</title>
        <style>
          :root {
            color-scheme: dark;
            --bg: #0b0f17;
            --panel: #111827;
            --panel-2: #172033;
            --text: #f3f4f6;
            --muted: #9ca3af;
            --accent: #60a5fa;
          }
          body {
            margin: 0;
            font-family: Inter, system-ui, sans-serif;
            background: radial-gradient(circle at top, #172033 0%, var(--bg) 50%);
            color: var(--text);
            min-height: 100vh;
            display: grid;
            place-items: center;
            padding: 24px;
          }
          .card {
            max-width: 820px;
            width: 100%;
            background: rgba(17, 24, 39, 0.92);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 20px;
            padding: 32px;
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.35);
          }
          h1 { margin: 0 0 12px; font-size: 2.4rem; }
          p { color: var(--muted); line-height: 1.6; }
          .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 14px;
            margin-top: 24px;
          }
          .item {
            background: var(--panel-2);
            border-radius: 16px;
            padding: 16px;
          }
          .item strong { display: block; margin-bottom: 6px; }
          a { color: var(--accent); text-decoration: none; }
        </style>
      </head>
      <body>
        <main class="card">
          <h1>AI Powered Data Analyst</h1>
          <p>
            The interactive Streamlit dashboard runs locally from <a href="/docs">docs</a>
            in development. This Vercel entrypoint exists so the repository deploys cleanly.
          </p>
          <div class="grid">
            <div class="item"><strong>Status</strong>Python app entrypoint detected</div>
            <div class="item"><strong>Local UI</strong>Run <code>streamlit run app.py</code></div>
            <div class="item"><strong>API</strong><a href="/health">Health check</a></div>
          </div>
        </main>
      </body>
    </html>
    """


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})
