"""Local FastAPI server — natural-language oculomotor simulation.

Usage
-----
    python -X utf8 scripts/server.py
    python -X utf8 scripts/server.py --host 0.0.0.0 --port 8000

Then open http://localhost:8000 in your browser.
"""

import os
import io
import csv
import uuid
import base64
import argparse
import traceback
import datetime
from pathlib import Path


from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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

# Allow requests from GitHub Pages (and any other origin) so the static
# frontend can call this local backend across origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['GET', 'POST'],
    allow_headers=['Content-Type'],
)

# Load any existing log at startup
_load_log()


# ── Request / response models ─────────────────────────────────────────────────

class RunRequest(BaseModel):
    description: str
    model: str = 'claude-sonnet-4-6'


class RunResponse(BaseModel):
    image_b64:       str
    mode:            str        # 'single' or 'comparison'
    title:           str
    detail_json:     dict
    run_id:          str
    version:         str
    eye_trajectory:  dict | None = None   # downsampled L/R eye positions for 3D animation


class FeedbackRequest(BaseModel):
    run_id:        str
    looks_correct: str | None = None   # 'correct', 'incorrect', or None (not rated)
    comment:       str = ''


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_eye_trajectory(sim_data: dict, fps: int = 60) -> dict | None:
    """Downsample per-eye position arrays for web animation.

    Returns a dict with 'fps', 'n_frames', 'duration_s', 'left', 'right'
    where left/right are lists of [yaw, pitch, roll] in degrees, rounded to
    2 decimal places.  Kept small enough to embed in the JSON response.
    """
    if sim_data is None:
        return None
    t  = np.array(sim_data['t'])
    dt = float(t[1] - t[0]) if len(t) > 1 else 0.001
    step = max(1, round(1.0 / (fps * dt)))

    L = np.array(sim_data.get('eye_pos_L', sim_data['eye_pos']))[::step]
    R = np.array(sim_data.get('eye_pos_R', sim_data['eye_pos']))[::step]
    t_ds = t[::step]

    # Integrate head velocity → head orientation (deg), then downsample
    dt_orig  = float(t[1] - t[0]) if len(t) > 1 else 0.001
    hv       = np.array(sim_data['head_vel'])            # (T, 3) deg/s
    head_pos = np.cumsum(hv * dt_orig, axis=0)           # (T, 3) deg
    head_ds  = head_pos[::step]

    return dict(
        fps        = fps,
        n_frames   = int(len(t_ds)),
        duration_s = round(float(t_ds[-1]), 3),
        left  = [[round(float(v), 2) for v in row] for row in L.tolist()],
        right = [[round(float(v), 2) for v in row] for row in R.tolist()],
        head  = [[round(float(v), 2) for v in row] for row in head_ds.tolist()],
    )


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.get('/version')
async def version_endpoint():
    """Return the running simulator version and git commit."""
    return JSONResponse({'version': _SIM_VERSION})


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
            image_b64      = img_b64,
            mode           = mode,
            title          = title,
            detail_json    = detail,
            run_id         = run_id,
            version        = _SIM_VERSION,
            eye_trajectory = _build_eye_trajectory(sim_data) if mode == 'single' else None,
        )

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={'error': str(e)})


@app.post('/feedback')
async def feedback_endpoint(req: FeedbackRequest):
    """Record user feedback (looks_correct + comment) for a run."""
    if req.run_id not in _log_entries:
        raise HTTPException(status_code=404, detail='run_id not found')
    _log_entries[req.run_id]['looks_correct'] = req.looks_correct or ''
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


# ── Serve docs/ as the frontend (must be last — catches all unmatched routes) ──

_DOCS_DIR = Path(__file__).parent.parent / 'docs'
app.mount('/', StaticFiles(directory=str(_DOCS_DIR), html=True), name='frontend')


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
