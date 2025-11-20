"""
frontend.py
-----------
Simple Flask-based frontend to browse miniWeather scaling CSVs and render quick
plots on localhost. Intended for lightweight verification of results when
developing or presenting without Slurm. The server reads CSV outputs produced
by `miniweather_mpi_omp`/`miniweather_mpi_cuda` and auto-generates sample data
if no files exist.
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
    tmpl = """
    <html>
    <head><title>miniWeather Scaling Dashboard</title></head>
    <body>
      <h1>miniWeather Scaling Dashboard</h1>
      <p>Simple local viewer for CSV outputs. Replace sample data by running the
         mini-app and saving CSVs to <code>results/</code>.</p>
      <h2>Strong Scaling</h2>
      <img src="/plot/strong.png" alt="Strong scaling plot" width="480" />
      <table border="1" cellpadding="4" cellspacing="0">
        <tr>{% for key in strong_cols %}<th>{{ key }}</th>{% endfor %}</tr>
        {% for row in strong %}<tr>{% for key in strong_cols %}<td>{{ row[key] }}</td>{% endfor %}</tr>{% endfor %}
      </table>
      <h2>Weak Scaling</h2>
      <img src="/plot/weak.png" alt="Weak scaling plot" width="480" />
      <table border="1" cellpadding="4" cellspacing="0">
        <tr>{% for key in weak_cols %}<th>{{ key }}</th>{% endfor %}</tr>
        {% for row in weak %}<tr>{% for key in weak_cols %}<td>{{ row[key] }}</td>{% endfor %}</tr>{% endfor %}
      </table>
      <p>Launch with <code>python src/frontend.py</code> and open http://127.0.0.1:5000/.</p>
    </body>
    </html>
    """
    return render_template_string(tmpl, strong=strong, weak=weak)


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
