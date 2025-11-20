"""
frontend.py
-----------
Simple Flask-based frontend to browse miniWeather scaling CSVs and render quick
plots on localhost. Intended for lightweight verification of results when
developing or presenting without Slurm. The server reads CSV outputs produced
by `miniweather_mpi_omp`/`miniweather_mpi_cuda` and auto-generates sample data
if no files exist.

This version adds a Grinch-inspired theme so demos feel more playful while
still surfacing the HPC numbers that matter. Everything is contained in a
single file with inline CSS and dynamic plots, so no static assets are
required.
"""
import csv
import os
from typing import List, Dict

from flask import Flask, Response, render_template_string
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

STRONG_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "strong_scaling.csv")
WEAK_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "weak_scaling.csv")


def _load_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _example_rows(mode: str) -> List[Dict[str, str]]:
    if mode == "strong":
        return [
            {"nodes": "1", "ranks": "4", "threads": "8", "nx": "2048", "nz": "1024", "steps": "200", "total_time": "0.92", "time_per_step": "0.0046", "avg_comm_per_step": "0.0007", "avg_compute_per_step": "0.0037", "comm_fraction": "0.15", "throughput_mcells_s": "455.6", "use_gpu": "1"},
            {"nodes": "4", "ranks": "16", "threads": "8", "nx": "2048", "nz": "1024", "steps": "200", "total_time": "0.30", "time_per_step": "0.0015", "avg_comm_per_step": "0.0004", "avg_compute_per_step": "0.0011", "comm_fraction": "0.27", "throughput_mcells_s": "1399.5", "use_gpu": "1"},
        ]
    return [
        {"nodes": "1", "ranks": "4", "threads": "8", "nx": "1024", "nz": "1024", "steps": "200", "total_time": "0.48", "time_per_step": "0.0024", "avg_comm_per_step": "0.0006", "avg_compute_per_step": "0.0018", "comm_fraction": "0.25", "throughput_mcells_s": "437.0", "use_gpu": "1"},
        {"nodes": "4", "ranks": "16", "threads": "8", "nx": "4096", "nz": "1024", "steps": "200", "total_time": "0.53", "time_per_step": "0.00265", "avg_comm_per_step": "0.0008", "avg_compute_per_step": "0.00185", "comm_fraction": "0.30", "throughput_mcells_s": "395.6", "use_gpu": "1"},
    ]


@app.route("/")
def index():
    strong = _load_csv(STRONG_PATH) or _example_rows("strong")
    weak = _load_csv(WEAK_PATH) or _example_rows("weak")
    strong_cols = list(strong[0].keys()) if strong else []
    weak_cols = list(weak[0].keys()) if weak else []
    banner_line = "Even the Grinch loves great scaling curves."
    strong_best = min(strong, key=lambda r: float(r["time_per_step"])) if strong else None
    weak_best = min(weak, key=lambda r: float(r["comm_fraction"])) if weak else None
    tmpl = """
    <html>
    <head>
      <title>miniWeather Scaling: Grinch Edition</title>
      <style>
        :root {
          --grinch-green: #00a86b;
          --grinch-dark: #0d1f12;
          --grinch-accent: #c41e3a;
          --card-bg: rgba(255,255,255,0.08);
        }
        * { font-family: 'Segoe UI', Tahoma, sans-serif; }
        body {
          background: radial-gradient(circle at 20% 20%, #0b2e19, #041009 60%),
                      linear-gradient(135deg, #0d1f12, #06160c);
          color: #e6f4ec;
          margin: 0;
          padding: 0;
        }
        header {
          padding: 20px 30px;
          background: linear-gradient(90deg, var(--grinch-green), #5de37a);
          color: #051009;
          box-shadow: 0 4px 20px rgba(0,0,0,0.35);
          position: sticky;
          top: 0;
          z-index: 3;
        }
        h1 { margin: 0; font-size: 28px; letter-spacing: 0.5px; }
        h2 { color: #9ef7bf; margin-bottom: 8px; }
        p.lede { margin: 6px 0 0; font-weight: 600; }
        .layout { display: grid; gap: 18px; padding: 24px; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); }
        .card {
          background: var(--card-bg);
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 14px;
          padding: 16px 18px;
          box-shadow: 0 12px 35px rgba(0,0,0,0.25);
          backdrop-filter: blur(6px);
        }
        .badge {
          display: inline-block;
          padding: 4px 10px;
          border-radius: 12px;
          background: #0c3c24;
          border: 1px solid rgba(255,255,255,0.12);
          color: #d8ffec;
          font-size: 12px;
          letter-spacing: 0.2px;
        }
        table { width: 100%; border-collapse: collapse; color: #e6f4ec; }
        th, td { border-bottom: 1px solid rgba(255,255,255,0.08); padding: 8px; text-align: left; }
        th { color: #b5f5c8; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }
        tr:hover { background: rgba(255,255,255,0.05); }
        .plot { width: 100%; border-radius: 10px; box-shadow: 0 6px 18px rgba(0,0,0,0.35); background: #0b1e13; padding: 10px; }
        .pill {
          display: inline-block; margin-right: 8px; padding: 6px 10px; border-radius: 999px;
          background: rgba(0,168,107,0.12); border: 1px solid rgba(0,168,107,0.5);
          color: #d8ffe9; font-size: 13px;
        }
        .footer { text-align: center; padding: 16px; color: #8ad9ae; font-size: 13px; }
        .accent { color: var(--grinch-accent); font-weight: bold; }
      </style>
    </head>
    <body>
      <header>
        <h1>miniWeather Scaling — Grinch Edition</h1>
        <p class="lede">{{ banner_line }}</p>
      </header>
      <div class="layout">
        <section class="card">
          <h2>Strong Scaling</h2>
          <p class="pill">GPU-friendly, MPI + OpenMP/CUDA</p>
          <img class="plot" src="/plot/strong.png" alt="Strong scaling plot" />
          {% if strong_best %}
            <p class="badge">Fastest step: {{ strong_best.time_per_step }}s @ {{ strong_best.nodes }} nodes</p>
          {% endif %}
          <table>
            <tr>{% for key in strong_cols %}<th>{{ key }}</th>{% endfor %}</tr>
            {% for row in strong %}<tr>{% for key in strong_cols %}<td>{{ row[key] }}</td>{% endfor %}</tr>{% endfor %}
          </table>
        </section>
        <section class="card">
          <h2>Weak Scaling</h2>
          <p class="pill">Balanced per-rank work, halo-heavy</p>
          <img class="plot" src="/plot/weak.png" alt="Weak scaling plot" />
          {% if weak_best %}
            <p class="badge">Lowest comm fraction: {{ weak_best.comm_fraction }}</p>
          {% endif %}
          <table>
            <tr>{% for key in weak_cols %}<th>{{ key }}</th>{% endfor %}</tr>
            {% for row in weak %}<tr>{% for key in weak_cols %}<td>{{ row[key] }}</td>{% endfor %}</tr>{% endfor %}
          </table>
        </section>
        <section class="card">
          <h2>How to Run Locally</h2>
          <p>Serve the dashboard with <code>python src/frontend.py</code>. Replace sample
             rows by generating <code>results/strong_scaling.csv</code> and
             <code>results/weak_scaling.csv</code> from your Slurm runs.</p>
          <ul>
            <li><strong>CPU build:</strong> <code>cd src && make cpu</code></li>
            <li><strong>GPU build:</strong> <code>cd src && make gpu</code></li>
            <li><strong>Baseline:</strong> <code>srun -n 1 --gpus-per-task=1 ./miniweather_mpi_cuda --nx 256 --nz 128 --steps 50 --output results/gpu.csv</code></li>
            <li><strong>Plots:</strong> <code>python src/plot_scaling.py</code></li>
          </ul>
          <p class="accent">Pro tip: pair this with profiling logs from Nsight or perf to show the Grinch that your halos are optimized.</p>
        </section>
      </div>
      <div class="footer">Powered by Flask + Matplotlib · No roast beast required</div>
    </body>
    </html>
    """
    return render_template_string(
        tmpl,
        strong=strong,
        weak=weak,
        strong_cols=strong_cols,
        weak_cols=weak_cols,
        banner_line=banner_line,
        strong_best=strong_best,
        weak_best=weak_best,
    )


def _plot_series(rows: List[Dict[str, str]], mode: str) -> io.BytesIO:
    if not rows:
        rows = _example_rows(mode)
    x = [int(r.get("nodes") or r.get("ranks")) for r in rows]
    times = [float(r["time_per_step"]) for r in rows]
    fig, ax = plt.subplots(figsize=(5, 3))
    if mode == "strong":
        baseline = times[0]
        speedup = [baseline / t for t in times]
        ax.plot(x, speedup, marker="o", label="Speedup")
        ax.set_ylabel("Speedup vs 1 rank")
    else:
        ax.plot(x, times, marker="o", label="Time/step")
        ax.set_ylabel("Time per step (s)")
    ax.set_xlabel("Nodes (or ranks)")
    ax.grid(True)
    ax.legend()
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


@app.route("/plot/<mode>.png")
def plot(mode: str):
    rows = _load_csv(STRONG_PATH if mode == "strong" else WEAK_PATH)
    buffer = _plot_series(rows, mode if mode in {"strong", "weak"} else "strong")
    return Response(buffer.getvalue(), mimetype="image/png")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
