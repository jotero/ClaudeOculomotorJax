"""Local FastAPI server — natural-language oculomotor simulation.

Usage
-----
    python -X utf8 scripts/server.py
    python -X utf8 scripts/server.py --host 0.0.0.0 --port 8000

Then open http://localhost:8000 in your browser.
To share over your local network: run with --host 0.0.0.0 and give
colleagues your machine's local IP (e.g. http://192.168.1.42:8000).
For internet access: set up port forwarding on your router for port 8000,
then share your public IP.
"""

import sys
import os
import io
import base64
import argparse
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from oculomotor.scenario import SimulationScenario, SimulationComparison
from oculomotor.runner import run_scenario, run_comparison
from scripts.simulate import call_llm, _call_llm, _call_llm_comparison


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(title='OculomotorSim')


# ── Request / response models ─────────────────────────────────────────────────

class RunRequest(BaseModel):
    description: str
    model: str = 'claude-sonnet-4-6'


class RunResponse(BaseModel):
    image_b64: str
    mode: str        # 'single' or 'comparison'
    title: str
    detail_json: dict


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.post('/run', response_model=RunResponse)
async def run_endpoint(req: RunRequest):
    """LLM decides single simulation or comparison; runs and returns the figure."""
    try:
        result = call_llm(req.description, model=req.model)

        if isinstance(result, SimulationComparison):
            fig   = run_comparison(result)
            title = result.title
            mode  = 'comparison'
            detail = result.model_dump()
        else:
            fig   = run_scenario(result)
            title = result.description
            mode  = 'single'
            detail = result.model_dump()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        return RunResponse(
            image_b64  = base64.b64encode(buf.read()).decode('utf-8'),
            mode       = mode,
            title      = title,
            detail_json= detail,
        )

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={'error': str(e)})


# ── Frontend ──────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OculomotorSim</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0f1117; color: #e0e0e0;
    min-height: 100vh; display: flex; flex-direction: column;
    align-items: center; padding: 40px 20px;
  }
  h1 { font-size: 1.6rem; font-weight: 600; margin-bottom: 6px; color: #fff; }
  .subtitle { font-size: 0.85rem; color: #888; margin-bottom: 32px; }
  .card {
    background: #1a1d27; border: 1px solid #2a2d3a;
    border-radius: 12px; padding: 28px; width: 100%; max-width: 820px;
  }
  .input-row { display: flex; gap: 10px; }
  textarea {
    flex: 1; background: #0f1117; border: 1px solid #2a2d3a;
    border-radius: 8px; color: #e0e0e0; font-size: 0.95rem;
    padding: 12px 14px; resize: vertical; min-height: 80px;
    outline: none; transition: border-color 0.2s; font-family: inherit;
  }
  textarea:focus { border-color: #4a6cf7; }
  button {
    background: #4a6cf7; border: none; border-radius: 8px; color: #fff;
    cursor: pointer; font-size: 0.95rem; font-weight: 600; padding: 0 22px;
    transition: background 0.2s, opacity 0.2s; white-space: nowrap;
    align-self: flex-end; height: 44px;
  }
  button:hover { background: #3a5ce0; }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  .examples { margin-top: 16px; display: flex; flex-wrap: wrap; gap: 8px; }
  .example-chip {
    background: #22253a; border: 1px solid #2a2d3a; border-radius: 20px;
    color: #aaa; cursor: pointer; font-size: 0.78rem; padding: 5px 13px;
    transition: all 0.15s;
  }
  .example-chip:hover { border-color: #4a6cf7; color: #ccc; background: #1e2135; }
  .status { margin-top: 18px; font-size: 0.85rem; color: #888; min-height: 1.2em; }
  .status.error { color: #f07070; }
  .mode-badge {
    display: inline-block; font-size: 0.72rem; font-weight: 600;
    padding: 2px 8px; border-radius: 10px; margin-right: 6px;
    vertical-align: middle;
  }
  .mode-single     { background: #1e3a6e; color: #7ab3f7; }
  .mode-comparison { background: #1a3a28; color: #6fcf97; }
  .result { margin-top: 24px; display: none; }
  .result img { width: 100%; border-radius: 8px; border: 1px solid #2a2d3a; }
  details { margin-top: 16px; border: 1px solid #2a2d3a; border-radius: 8px; overflow: hidden; }
  summary { cursor: pointer; padding: 10px 14px; font-size: 0.82rem; color: #888; background: #151820; user-select: none; }
  summary:hover { color: #ccc; }
  pre { background: #0f1117; color: #a0c0a0; font-size: 0.75rem; overflow-x: auto; padding: 14px; max-height: 320px; }
  .spinner {
    display: inline-block; width: 14px; height: 14px;
    border: 2px solid #4a6cf7; border-top-color: transparent;
    border-radius: 50%; animation: spin 0.7s linear infinite;
    margin-right: 8px; vertical-align: middle;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>

<h1>OculomotorSim</h1>
<p class="subtitle">Describe a scenario or comparison — the model figures out what to simulate.</p>

<div class="card">
  <div class="input-row">
    <textarea id="prompt" placeholder="e.g. &quot;VOR in the dark with left vestibular neuritis&quot; or &quot;compare healthy vs cerebellar deficit on a head impulse test&quot;"></textarea>
    <button id="btn" onclick="run()">Run</button>
  </div>

  <div class="examples">
    <span class="example-chip" onclick="set('Healthy 20° saccade to the right')">Healthy 20° saccade</span>
    <span class="example-chip" onclick="set('Left vestibular neuritis doing a head impulse test to the right')">Left neuritis HIT</span>
    <span class="example-chip" onclick="set('OKN: 30 deg/s full-field scene motion for 20 s, then OKAN')">OKN + OKAN</span>
    <span class="example-chip" onclick="set('Smooth pursuit of a 20 deg/s ramp target')">Smooth pursuit</span>
    <span class="example-chip" onclick="set('Cerebellar gaze-holding deficit, gaze-evoked nystagmus at 15 degrees')">Cerebellar GEN</span>
    <span class="example-chip" onclick="set('Compare healthy vs left vestibular neuritis on a rightward head impulse test')">Compare: healthy vs neuritis</span>
    <span class="example-chip" onclick="set('Compare healthy vs cerebellar deficit vs bilateral vestibular loss on VOR in the dark')">Compare: 3-way VOR</span>
    <span class="example-chip" onclick="set('Compare healthy vs pursuit deficit on a 20 deg/s ramp target')">Compare: pursuit deficit</span>
  </div>

  <div class="status" id="status"></div>

  <div class="result" id="result">
    <img id="plot" src="" alt="Simulation plot">
    <details><summary>JSON</summary><pre id="json-out"></pre></details>
  </div>
</div>

<script>
function set(text) { document.getElementById('prompt').value = text; }

async function run() {
  const prompt = document.getElementById('prompt').value.trim();
  if (!prompt) return;
  const btn    = document.getElementById('btn');
  const status = document.getElementById('status');
  const result = document.getElementById('result');
  btn.disabled = true;
  result.style.display = 'none';
  status.className = 'status';
  status.innerHTML = '<span class="spinner"></span>Thinking…';
  try {
    const resp = await fetch('/run', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({description: prompt}),
    });
    if (!resp.ok) { const e = await resp.json(); throw new Error(e.error || `HTTP ${resp.status}`); }
    const data = await resp.json();
    document.getElementById('plot').src = 'data:image/png;base64,' + data.image_b64;
    document.getElementById('json-out').textContent = JSON.stringify(data.detail_json, null, 2);
    result.style.display = 'block';
    const badge = data.mode === 'comparison'
      ? '<span class="mode-badge mode-comparison">comparison</span>'
      : '<span class="mode-badge mode-single">single</span>';
    status.innerHTML = badge + data.title;
  } catch(e) {
    status.className = 'status error';
    status.textContent = 'Error: ' + e.message;
  } finally { btn.disabled = false; }
}

document.getElementById('prompt').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) run();
});
</script>
</body>
</html>
"""


@app.get('/', response_class=HTMLResponse)
async def index():
    return HTML


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0',
                        help='Bind host (default 0.0.0.0 = all interfaces)')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    print(f"\n  OculomotorSim running at http://localhost:{args.port}")
    print(f"  Local network:  http://<your-ip>:{args.port}")
    print(f"  Ctrl+C to stop\n")

    uvicorn.run(app, host=args.host, port=args.port)
