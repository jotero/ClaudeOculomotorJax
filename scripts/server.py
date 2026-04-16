"""Local FastAPI server — natural-language oculomotor simulation.

Usage
-----
    python -X utf8 scripts/server.py
    python -X utf8 scripts/server.py --host 0.0.0.0 --port 8000

Then open http://localhost:8000 in your browser.
"""

import sys
import os
import io
import csv
import uuid
import base64
import argparse
import traceback
import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from oculomotor import __version__ as _SIM_VERSION
from oculomotor.llm_pipeline.scenario import SimulationScenario, SimulationComparison
from oculomotor.llm_pipeline.runner import run_scenario, run_comparison
from oculomotor.llm_pipeline.simulate import call_llm


# ── Paths ─────────────────────────────────────────────────────────────────────

_OUTPUTS_DIR = Path(__file__).parent.parent / 'outputs'
_OUTPUTS_DIR.mkdir(exist_ok=True)

_FIGURES_DIR = _OUTPUTS_DIR / 'server_figures'
_FIGURES_DIR.mkdir(exist_ok=True)

_LOG_FILE = _OUTPUTS_DIR / 'simulation_log.csv'

_LOG_COLUMNS = [
    'timestamp', 'run_id', 'version', 'prompt', 'mode', 'title',
    'figure_file', 'looks_correct', 'feedback',
]


# ── In-memory state ───────────────────────────────────────────────────────────

_log_entries: dict[str, dict] = {}   # run_id → CSV row dict
_sim_cache:   dict[str, dict] = {}   # run_id → sim data arrays (numpy)


def _load_log() -> None:
    """Load existing log CSV into _log_entries on startup."""
    if _LOG_FILE.exists():
        with open(_LOG_FILE, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                _log_entries[row['run_id']] = dict(row)


def _append_log(row: dict) -> None:
    """Append one row to the log CSV and in-memory dict."""
    _log_entries[row['run_id']] = row
    write_header = not _LOG_FILE.exists()
    with open(_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=_LOG_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow(row)


def _rewrite_log() -> None:
    """Rewrite the full CSV (used after feedback updates)."""
    with open(_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=_LOG_COLUMNS)
        w.writeheader()
        for row in _log_entries.values():
            w.writerow(row)


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(title='OculomotorSim')

# Load any existing log at startup
_load_log()


# ── Request / response models ─────────────────────────────────────────────────

class RunRequest(BaseModel):
    description: str
    model: str = 'claude-sonnet-4-6'


class RunResponse(BaseModel):
    image_b64:  str
    mode:       str       # 'single' or 'comparison'
    title:      str
    detail_json: dict
    run_id:     str
    version:    str


class FeedbackRequest(BaseModel):
    run_id:        str
    looks_correct: bool
    comment:       str = ''


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.post('/run', response_model=RunResponse)
async def run_endpoint(req: RunRequest):
    """LLM decides single simulation or comparison; runs and returns the figure."""
    run_id = str(uuid.uuid4())
    try:
        result = call_llm(req.description, model=req.model)

        if isinstance(result, SimulationComparison):
            fig   = run_comparison(result)
            title = result.title
            mode  = 'comparison'
            detail = result.model_dump()
            sim_data = None    # comparison download not supported yet
        else:
            fig, sim_data = run_scenario(result, return_data=True)
            title  = result.description
            mode   = 'single'
            detail = result.model_dump()

        # Save figure to disk
        fig_name = f'{run_id}.png'
        fig_path = _FIGURES_DIR / fig_name
        fig.savefig(fig_path, dpi=130, bbox_inches='tight')

        # Encode for inline display
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')

        # Cache sim data for download
        if sim_data is not None:
            _sim_cache[run_id] = sim_data

        # Log
        _append_log({
            'timestamp':   datetime.datetime.utcnow().isoformat(timespec='seconds') + 'Z',
            'run_id':      run_id,
            'version':     _SIM_VERSION,
            'prompt':      req.description,
            'mode':        mode,
            'title':       title,
            'figure_file': str(fig_path),
            'looks_correct': '',
            'feedback':    '',
        })

        return RunResponse(
            image_b64   = img_b64,
            mode        = mode,
            title       = title,
            detail_json = detail,
            run_id      = run_id,
            version     = _SIM_VERSION,
        )

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={'error': str(e)})


@app.post('/feedback')
async def feedback_endpoint(req: FeedbackRequest):
    """Record user feedback (looks_correct + comment) for a run."""
    if req.run_id not in _log_entries:
        raise HTTPException(status_code=404, detail='run_id not found')
    _log_entries[req.run_id]['looks_correct'] = str(req.looks_correct)
    _log_entries[req.run_id]['feedback']      = req.comment
    _rewrite_log()
    return {'status': 'ok'}


@app.get('/download/{run_id}')
async def download_endpoint(run_id: str):
    """Return simulation data as a CSV file (eye + stimulus arrays)."""
    if run_id not in _sim_cache:
        raise HTTPException(status_code=404,
                            detail='Data not available (comparison runs or expired cache).')
    data = _sim_cache[run_id]

    # Build CSV in memory
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        't_s',
        'eye_pos_yaw_deg', 'eye_pos_pitch_deg', 'eye_pos_roll_deg',
        'eye_vel_yaw_degs', 'eye_vel_pitch_degs', 'eye_vel_roll_degs',
        'head_vel_yaw_degs', 'head_vel_pitch_degs', 'head_vel_roll_degs',
        'scene_vel_yaw_degs', 'scene_vel_pitch_degs', 'scene_vel_roll_degs',
        'target_vel_yaw_degs', 'target_vel_pitch_degs', 'target_vel_roll_degs',
    ])

    t          = np.array(data['t'])
    eye_pos    = np.array(data['eye_pos'])
    eye_vel    = np.array(data['eye_vel'])
    head_vel   = np.array(data['head_vel'])
    scene_vel  = np.array(data['scene_vel'])
    target_vel = np.array(data['target_vel'])

    for i in range(len(t)):
        writer.writerow([
            f'{t[i]:.4f}',
            *[f'{v:.4f}' for v in eye_pos[i]],
            *[f'{v:.4f}' for v in eye_vel[i]],
            *[f'{v:.4f}' for v in head_vel[i]],
            *[f'{v:.4f}' for v in scene_vel[i]],
            *[f'{v:.4f}' for v in target_vel[i]],
        ])

    csv_bytes = buf.getvalue().encode('utf-8')
    filename  = f'oculomotorsim_{run_id[:8]}.csv'

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type='text/csv',
        headers={'Content-Disposition': f'attachment; filename="{filename}"'},
    )


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
    margin-bottom: 20px;
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
  button.secondary {
    background: #22253a; border: 1px solid #2a2d3a; color: #aaa;
    font-weight: 500;
  }
  button.secondary:hover { background: #2a2e45; color: #ddd; border-color: #4a6cf7; }
  button.success { background: #1a6e3a; }
  button.success:hover { background: #16593a; }
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
  .version-tag {
    display: inline-block; font-size: 0.68rem; color: #555;
    font-family: monospace; margin-left: 8px; vertical-align: middle;
  }
  .result { margin-top: 24px; display: none; }
  .result img { width: 100%; border-radius: 8px; border: 1px solid #2a2d3a; }
  details { margin-top: 16px; border: 1px solid #2a2d3a; border-radius: 8px; overflow: hidden; }
  summary { cursor: pointer; padding: 10px 14px; font-size: 0.82rem; color: #888; background: #151820; user-select: none; }
  summary:hover { color: #ccc; }
  pre { background: #0f1117; color: #a0c0a0; font-size: 0.75rem; overflow-x: auto; padding: 14px; max-height: 320px; }

  /* Feedback section */
  .feedback-section {
    margin-top: 20px; padding: 18px; background: #151820;
    border: 1px solid #2a2d3a; border-radius: 10px; display: none;
  }
  .feedback-section h3 { font-size: 0.9rem; color: #ccc; margin-bottom: 12px; }
  .feedback-row { display: flex; align-items: flex-start; gap: 16px; flex-wrap: wrap; }
  .checkbox-label {
    display: flex; align-items: center; gap: 7px;
    font-size: 0.88rem; color: #aaa; cursor: pointer; user-select: none;
    white-space: nowrap; padding-top: 6px;
  }
  .checkbox-label input[type=checkbox] { width: 16px; height: 16px; accent-color: #4a6cf7; cursor: pointer; }
  .feedback-comment {
    flex: 1; min-width: 200px;
    background: #0f1117; border: 1px solid #2a2d3a; border-radius: 6px;
    color: #e0e0e0; font-size: 0.85rem; padding: 8px 10px;
    resize: vertical; min-height: 60px; outline: none; font-family: inherit;
    transition: border-color 0.2s;
  }
  .feedback-comment:focus { border-color: #4a6cf7; }
  .feedback-actions { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-top: 12px; }
  .feedback-disclaimer {
    font-size: 0.72rem; color: #555; margin-top: 8px; line-height: 1.5;
  }
  .feedback-confirm { font-size: 0.82rem; color: #6fcf97; display: none; margin-top: 6px; }

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

    <!-- Feedback section -->
    <div class="feedback-section" id="feedback-section">
      <h3>Feedback</h3>
      <div class="feedback-row">
        <label class="checkbox-label">
          <input type="checkbox" id="looks-correct"> Simulation looks correct
        </label>
        <textarea class="feedback-comment" id="feedback-comment"
                  placeholder="Optional: notes on what looks right or wrong…"></textarea>
      </div>
      <div class="feedback-actions">
        <button class="secondary" id="submit-feedback-btn" onclick="submitFeedback()">Submit feedback</button>
        <button class="secondary" id="download-btn" onclick="downloadData()" style="display:none">
          &#8681; Download data (CSV)
        </button>
      </div>
      <p class="feedback-disclaimer">
        Feedback is recorded anonymously (prompt, result, your notes). No IP address or personal
        information is stored. Data is used only for research quality assessment.
      </p>
      <p class="feedback-confirm" id="feedback-confirm">&#10003; Feedback saved — thank you!</p>
    </div>

    <details><summary>JSON</summary><pre id="json-out"></pre></details>
  </div>
</div>

<script>
let _currentRunId = null;
let _currentMode  = null;

function set(text) { document.getElementById('prompt').value = text; }

async function run() {
  const prompt = document.getElementById('prompt').value.trim();
  if (!prompt) return;
  const btn    = document.getElementById('btn');
  const status = document.getElementById('status');
  const result = document.getElementById('result');

  btn.disabled = true;
  result.style.display = 'none';
  document.getElementById('feedback-section').style.display = 'none';
  document.getElementById('feedback-confirm').style.display = 'none';
  document.getElementById('looks-correct').checked = false;
  document.getElementById('feedback-comment').value = '';
  status.className = 'status';
  status.innerHTML = '<span class="spinner"></span>Thinking…';

  try {
    const resp = await fetch('/run', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({description: prompt}),
    });
    if (!resp.ok) { const e = await resp.json(); throw new Error(e.error || 'HTTP ' + resp.status); }
    const data = await resp.json();

    _currentRunId = data.run_id;
    _currentMode  = data.mode;

    document.getElementById('plot').src = 'data:image/png;base64,' + data.image_b64;
    document.getElementById('json-out').textContent = JSON.stringify(data.detail_json, null, 2);
    result.style.display = 'block';

    const badge = data.mode === 'comparison'
      ? '<span class="mode-badge mode-comparison">comparison</span>'
      : '<span class="mode-badge mode-single">single</span>';
    const vtag = '<span class="version-tag">v' + data.version + '</span>';
    status.innerHTML = badge + data.title + vtag;

    // Show feedback section
    document.getElementById('feedback-section').style.display = 'block';
    // Show download button only for single runs
    document.getElementById('download-btn').style.display =
      (data.mode === 'single') ? 'inline-block' : 'none';

  } catch(e) {
    status.className = 'status error';
    status.textContent = 'Error: ' + e.message;
  } finally { btn.disabled = false; }
}

async function submitFeedback() {
  if (!_currentRunId) return;
  const btn     = document.getElementById('submit-feedback-btn');
  const confirm = document.getElementById('feedback-confirm');
  btn.disabled  = true;
  try {
    const resp = await fetch('/feedback', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        run_id:        _currentRunId,
        looks_correct: document.getElementById('looks-correct').checked,
        comment:       document.getElementById('feedback-comment').value,
      }),
    });
    if (!resp.ok) throw new Error('Failed to save feedback');
    confirm.style.display = 'block';
    btn.textContent = 'Feedback saved';
    btn.className   = 'secondary success';
  } catch(e) {
    confirm.style.display = 'block';
    confirm.style.color   = '#f07070';
    confirm.textContent   = 'Error saving feedback — please try again.';
    btn.disabled = false;
  }
}

function downloadData() {
  if (!_currentRunId) return;
  window.location.href = '/download/' + _currentRunId;
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

    print(f"\n  OculomotorSim {_SIM_VERSION} running at http://localhost:{args.port}")
    print(f"  Local network:  http://<your-ip>:{args.port}")
    print(f"  Log:            {_LOG_FILE}")
    print(f"  Ctrl+C to stop\n")

    uvicorn.run(app, host=args.host, port=args.port)
