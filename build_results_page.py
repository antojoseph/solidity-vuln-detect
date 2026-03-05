"""Build a self-contained HTML results page from benchmark JSON files."""

import json
import statistics
from collections import defaultdict
from pathlib import Path

RESULT_FILES = {
    "Claude Sonnet 4.6": "results/claude-sonnet-4-6_20260223_070032.json",
    "Claude Haiku 4.5": "results/claude-haiku-4-5-20251001_20260224_080612.json",
    "Claude Opus 4.6": "results/claude-opus-4-6_20260223_062030.json",
    "Qwen3.5 (SGLang)": "results/Qwen_Qwen3.5-397B-A17B-FP8_20260226_031433.json",
    "Qwen3.5 (vLLM)": "results/Qwen_Qwen3.5-397B-A17B-FP8_20260227_194405.json",
}

COLORS = {
    "Claude Sonnet 4.6": "#8b5cf6",
    "Claude Haiku 4.5": "#a78bfa",
    "Claude Opus 4.6": "#6366f1",
    "Qwen3.5 (SGLang)": "#fb923c",
    "Qwen3.5 (vLLM)": "#f97316",
}

SUBSCORE_WEIGHTS = {
    "category_match": 0.40,
    "explanation_quality": 0.25,
    "severity_match": 0.10,
    "line_accuracy": 0.15,
    "exploitability": 0.10,
}


def load_model_data(path):
    with open(path) as f:
        data = json.load(f)
    traces = [t for t in data["traces"] if t.get("error") is None]
    rewards = [t["reward"] for t in traces]

    by_diff = defaultdict(list)
    by_cat = defaultdict(list)
    subscores = defaultdict(list)
    steps_list = []
    severity_dist = defaultdict(int)

    for t in traces:
        info = t.get("info", {})
        by_diff[info.get("difficulty", "unknown")].append(t["reward"])
        by_cat[info.get("canonical_category", "unknown")].append(t["reward"])
        for k, v in t.get("subscores", {}).items():
            subscores[k].append(v)
        steps_list.append(t.get("steps", 0))
        sev = info.get("submitted_severity", "UNKNOWN")
        severity_dist[sev] += 1

    sorted_rewards = sorted(rewards)
    n = len(rewards)

    return {
        "n": n,
        "total": data["total_tasks"],
        "errors": len(data["traces"]) - n,
        "mean": round(statistics.mean(rewards), 4) if rewards else 0,
        "median": round(statistics.median(rewards), 4) if rewards else 0,
        "stdev": round(statistics.stdev(rewards), 4) if n > 1 else 0,
        "p25": round(sorted_rewards[n // 4], 4) if rewards else 0,
        "p75": round(sorted_rewards[3 * n // 4], 4) if rewards else 0,
        "min": round(min(rewards), 4) if rewards else 0,
        "max": round(max(rewards), 4) if rewards else 0,
        "zeros": sum(1 for r in rewards if r == 0.0),
        "highs": sum(1 for r in rewards if r >= 0.8),
        "high_rate": round(sum(1 for r in rewards if r >= 0.8) / n * 100, 1) if n else 0,
        "by_diff": {k: round(statistics.mean(v), 4) for k, v in by_diff.items()},
        "by_cat": {k: round(statistics.mean(v), 4) for k, v in sorted(by_cat.items())},
        "by_cat_n": {k: len(v) for k, v in sorted(by_cat.items())},
        "subscores": {k: round(statistics.mean(v), 4) for k, v in subscores.items()},
        "avg_steps": round(statistics.mean(steps_list), 1) if steps_list else 0,
        "rewards": rewards,
        "severity_dist": dict(severity_dist),
        "timestamp": data.get("timestamp", ""),
    }


def build_histogram_data(rewards, bins=20):
    """Return bin edges and counts for histogram."""
    min_val, max_val = 0.0, 1.0
    step = (max_val - min_val) / bins
    edges = [round(min_val + i * step, 3) for i in range(bins + 1)]
    counts = [0] * bins
    for r in rewards:
        idx = min(int((r - min_val) / step), bins - 1)
        counts[idx] += 1
    return edges, counts


def generate_html(models_data):
    model_names = list(models_data.keys())
    colors = [COLORS[m] for m in model_names]

    # Histogram data
    hist_datasets = []
    for name in model_names:
        edges, counts = build_histogram_data(models_data[name]["rewards"])
        hist_datasets.append({"label": name, "data": counts})
    hist_labels = [f"{edges[i]:.2f}" for i in range(len(edges) - 1)]

    # All categories across all models
    all_cats = sorted(set(
        cat for m in models_data.values() for cat in m["by_cat"]
    ))

    # Category heatmap data
    cat_data = []
    for cat in all_cats:
        row = {"category": cat}
        for name in model_names:
            row[name] = models_data[name]["by_cat"].get(cat, None)
            row[name + "_n"] = models_data[name]["by_cat_n"].get(cat, 0)
        cat_data.append(row)

    # Subscore names
    subscore_names = list(SUBSCORE_WEIGHTS.keys())
    subscore_labels = [
        f"{s.replace('_', ' ').title()} ({int(SUBSCORE_WEIGHTS[s]*100)}%)"
        for s in subscore_names
    ]

    # Embed data as JSON
    page_data = {
        "models": model_names,
        "colors": colors,
        "summary": {name: {k: v for k, v in d.items() if k != "rewards"} for name, d in models_data.items()},
        "hist_labels": hist_labels,
        "hist_datasets": hist_datasets,
        "difficulties": ["easy", "medium", "hard"],
        "diff_data": {name: [d["by_diff"].get(diff, 0) for diff in ["easy", "medium", "hard"]] for name, d in models_data.items()},
        "subscore_names": subscore_names,
        "subscore_labels": subscore_labels,
        "subscore_data": {name: [d["subscores"].get(s, 0) for s in subscore_names] for name, d in models_data.items()},
        "cat_data": cat_data,
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Solidity Vulnerability Detection Benchmark Results</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0f172a; --bg2: #1e293b; --bg3: #334155;
    --fg: #f1f5f9; --fg2: #94a3b8; --fg3: #64748b;
    --accent: #8b5cf6; --green: #22c55e; --red: #ef4444; --yellow: #eab308; --orange: #f97316;
    --border: #334155;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace; background: var(--bg); color: var(--fg); line-height: 1.6; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 2rem; }}
  h1 {{ font-size: 2rem; margin-bottom: 0.5rem; }}
  h2 {{ font-size: 1.4rem; margin: 2.5rem 0 1rem; color: var(--fg); border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }}
  h3 {{ font-size: 1.1rem; color: var(--fg2); margin-bottom: 0.5rem; }}
  .subtitle {{ color: var(--fg2); margin-bottom: 2rem; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
  .card {{ background: var(--bg2); border-radius: 12px; padding: 1.5rem; border: 1px solid var(--border); }}
  .card-title {{ font-size: 1.1rem; font-weight: 600; margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.5rem; }}
  .card-dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
  .card .big {{ font-size: 2rem; font-weight: 700; }}
  .card .stats {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.25rem 1rem; margin-top: 0.75rem; font-size: 0.85rem; color: var(--fg2); }}
  .card .stats .val {{ color: var(--fg); font-weight: 500; }}
  .chart-container {{ background: var(--bg2); border-radius: 12px; padding: 1.5rem; border: 1px solid var(--border); margin-bottom: 1.5rem; }}
  .chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
  @media (max-width: 900px) {{ .chart-row {{ grid-template-columns: 1fr; }} }}
  canvas {{ max-height: 400px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th, td {{ padding: 0.6rem 0.8rem; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ background: var(--bg3); color: var(--fg); font-weight: 600; cursor: pointer; user-select: none; }}
  th:hover {{ background: #475569; }}
  td {{ color: var(--fg2); }}
  tr:hover td {{ background: rgba(139,92,246,0.05); }}
  .best {{ color: var(--green); font-weight: 600; }}
  .worst {{ color: var(--red); }}
  .heatcell {{ padding: 0.4rem 0.6rem; border-radius: 4px; text-align: center; font-weight: 500; font-size: 0.85rem; }}
  .findings {{ background: var(--bg2); border-radius: 12px; padding: 1.5rem; border: 1px solid var(--border); margin-bottom: 1.5rem; }}
  .findings li {{ margin-bottom: 0.5rem; color: var(--fg2); }}
  .findings li strong {{ color: var(--fg); }}
  .tag {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }}
  .tag-green {{ background: rgba(34,197,94,0.15); color: var(--green); }}
  .tag-red {{ background: rgba(239,68,68,0.15); color: var(--red); }}
  .tag-yellow {{ background: rgba(234,179,8,0.15); color: var(--yellow); }}
  #catFilter {{ background: var(--bg3); color: var(--fg); border: 1px solid var(--border); padding: 0.5rem; border-radius: 6px; margin-bottom: 1rem; width: 300px; }}
</style>
</head>
<body>
<div class="container">

<h1>Solidity Vulnerability Detection Benchmark</h1>
<p class="subtitle">972 tasks &middot; 5 models &middot; Multi-turn tool-calling agent evaluation &middot; February 2026</p>

<!-- All chart sections laid out statically so innerHTML += doesn't destroy canvases -->
<div id="cards"></div>

<h2>Overall Performance</h2>
<div class="chart-row">
  <div class="chart-container"><canvas id="overallChart"></canvas></div>
  <div class="chart-container"><canvas id="histChart"></canvas></div>
</div>

<h2>Performance by Difficulty</h2>
<div class="chart-container"><canvas id="diffChart"></canvas></div>

<h2>Subscores</h2>
<div class="chart-row">
  <div class="chart-container"><canvas id="radarChart"></canvas></div>
  <div class="chart-container"><canvas id="subscoreBarChart"></canvas></div>
</div>

<h2>Scoring Methodology</h2>
<div class="chart-container" style="font-size:0.9rem;color:var(--fg2)">
  <p style="margin-bottom:1rem;color:var(--fg)">Each submission is scored by 5 deterministic subscores (no LLM judge), combined into a final reward:</p>
  <div style="background:var(--bg3);padding:1rem;border-radius:8px;margin-bottom:1.5rem;font-family:monospace;font-size:0.85rem;color:var(--fg);overflow-x:auto">
    reward = (0.40 &times; category_match + 0.25 &times; explanation_quality + 0.10 &times; severity_match + 0.15 &times; line_accuracy + 0.10 &times; exploitability) &times; schema_penalty &times; hint_penalty
  </div>

  <div style="display:grid;gap:1.25rem">
    <div>
      <h3 style="color:#8b5cf6;margin-bottom:0.3rem">Category Match (40% weight)</h3>
      <p><strong>What it measures:</strong> Did the model correctly identify the vulnerability type from a taxonomy of 41 categories (e.g. reentrancy, oracle-manipulation, access-control)?</p>
      <p style="margin-top:0.3rem"><strong>How it scores:</strong> Exact match on canonical category &rarr; 1.0 &middot; Keyword overlap &rarr; 0.6&ndash;0.9 &middot; Jaccard word similarity &rarr; 0.4&ndash;0.8 &middot; No match &rarr; 0.0</p>
      <p style="margin-top:0.3rem"><strong>Significance:</strong> The heaviest-weighted subscore. Low scores (like Qwen&rsquo;s 0.21) mean the model identifies real vulnerabilities but labels them differently than the ground truth taxonomy &mdash; a classification problem, not a comprehension one.</p>
    </div>

    <div>
      <h3 style="color:#8b5cf6;margin-bottom:0.3rem">Explanation Quality (25% weight)</h3>
      <p><strong>What it measures:</strong> How well does the model explain <em>why</em> the code is vulnerable?</p>
      <p style="margin-top:0.3rem"><strong>How it scores:</strong> References specific code identifiers (0&ndash;0.3) + Uses attack vector language like &ldquo;exploit&rdquo;, &ldquo;drain&rdquo;, &ldquo;manipulate&rdquo; (0&ndash;0.4) + Sufficient length &ge;50 words (0&ndash;0.15) + References preconditions (0&ndash;0.15)</p>
      <p style="margin-top:0.3rem"><strong>Significance:</strong> All models score 0.92&ndash;0.95, indicating this subscore is near-saturated and doesn&rsquo;t differentiate models well. All models write detailed, well-structured explanations.</p>
    </div>

    <div>
      <h3 style="color:#8b5cf6;margin-bottom:0.3rem">Line Accuracy (15% weight)</h3>
      <p><strong>What it measures:</strong> Did the model pinpoint the exact lines of code containing the root cause?</p>
      <p style="margin-top:0.3rem"><strong>How it scores:</strong> F1 score of submitted line numbers vs ground truth, with &plusmn;1 line tolerance. No ground truth lines &rarr; flat 0.5, no submission &rarr; 0.0.</p>
      <p style="margin-top:0.3rem"><strong>Significance:</strong> Scores of 0.32&ndash;0.41 indicate models identify the right code region but struggle with exact line precision. This is the most granular test of code understanding.</p>
    </div>

    <div>
      <h3 style="color:#8b5cf6;margin-bottom:0.3rem">Severity Match (10% weight)</h3>
      <p><strong>What it measures:</strong> Did the model correctly rate the impact severity?</p>
      <p style="margin-top:0.3rem"><strong>How it scores:</strong> Exact match &rarr; 1.0 &middot; Off by 1 level &rarr; 0.6 &middot; Off by 2 levels &rarr; 0.3 &middot; Scale: CRITICAL &gt; HIGH &gt; MEDIUM &gt; LOW &gt; NONE</p>
      <p style="margin-top:0.3rem"><strong>Significance:</strong> Scores of 0.66&ndash;0.79 mean models are usually close but tend to slightly over- or under-rate severity. Opus scores highest here (0.79), suggesting better calibration.</p>
    </div>

    <div>
      <h3 style="color:#8b5cf6;margin-bottom:0.3rem">Exploitability (10% weight)</h3>
      <p><strong>What it measures:</strong> Quality of the attack path, prerequisites, and impact descriptions.</p>
      <p style="margin-top:0.3rem"><strong>How it scores:</strong> Each field scored on word count + quality keywords (action verbs, sequencing terms, impact terms). Combined as 40% attack_path + 30% prerequisites + 30% impact.</p>
      <p style="margin-top:0.3rem"><strong>Significance:</strong> All models score 0.87&ndash;0.90, indicating consistently good exploit narratives. Like explanation quality, this subscore is near-saturated.</p>
    </div>

    <div style="border-top:1px solid var(--border);padding-top:1rem">
      <h3 style="color:var(--fg3);margin-bottom:0.3rem">Penalties</h3>
      <p><strong>Schema penalty</strong> (0.85&ndash;0.90&times;): Applied when optional fields (attack_path, prerequisites, impact) are all empty, or when the model submits multiple vulnerability labels separated by commas.</p>
      <p style="margin-top:0.3rem"><strong>Hint penalty</strong> (0.70&times;): Applied when the model called the <code>list_hints()</code> tool, which provides detection guidance but caps the maximum achievable reward.</p>
    </div>
  </div>
</div>

<h2>Subscore Details</h2>
<div class="chart-container"><table id="subscoreTable"><thead><tr id="subscoreHead"></tr></thead><tbody id="subscoreBody"></tbody></table></div>

<h2>Distribution Statistics</h2>
<div class="chart-container"><table id="distTable"><thead><tr id="distHead"></tr></thead><tbody id="distBody"></tbody></table></div>

<h2>Performance by Category</h2>
<input type="text" id="catFilter" placeholder="Filter categories...">
<div class="chart-container" style="overflow-x:auto"><table id="catTable"><thead><tr id="catHead"></tr></thead><tbody id="catBody"></tbody></table></div>

<div id="findings"></div>

<p style="text-align:center;color:var(--fg3);margin-top:3rem;font-size:0.85rem">Generated from benchmark results &middot; 972 Solidity vulnerability detection tasks &middot; February 2026</p>

<script>
const DATA = {json.dumps(page_data, indent=None)};

// --- Summary Cards ---
let cardsHtml = '<h2>Model Summary</h2><div class="cards">';
DATA.models.forEach((name, i) => {{
  const s = DATA.summary[name];
  const color = DATA.colors[i];
  cardsHtml += `
    <div class="card">
      <div class="card-title"><span class="card-dot" style="background:${{color}}"></span>${{name}}</div>
      <div class="big" style="color:${{color}}">${{s.mean.toFixed(4)}}</div>
      <div style="color:var(--fg2);font-size:0.85rem">mean reward &plusmn; ${{s.stdev.toFixed(4)}}</div>
      <div class="stats">
        <div>Median <span class="val">${{s.median.toFixed(4)}}</span></div>
        <div>Completed <span class="val">${{s.n}}/${{s.total}}</span></div>
        <div>High (&ge;0.8) <span class="val">${{s.highs}} (${{s.high_rate}}%)</span></div>
        <div>Zeros <span class="val">${{s.zeros}}</span></div>
        <div>Avg steps <span class="val">${{s.avg_steps}}</span></div>
        <div>Errors <span class="val">${{s.errors}}</span></div>
      </div>
    </div>`;
}});
cardsHtml += '</div>';
document.getElementById('cards').innerHTML = cardsHtml;

// --- Chart defaults ---
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = '#334155';
Chart.defaults.font.family = '-apple-system, monospace';

// --- Overall Performance ---
new Chart(document.getElementById('overallChart'), {{
  type: 'bar',
  data: {{
    labels: DATA.models,
    datasets: [
      {{ label: 'Mean', data: DATA.models.map(m => DATA.summary[m].mean), backgroundColor: DATA.colors.map(c => c + 'cc'), borderColor: DATA.colors, borderWidth: 1 }},
      {{ label: 'Median', data: DATA.models.map(m => DATA.summary[m].median), backgroundColor: DATA.colors.map(c => c + '66'), borderColor: DATA.colors, borderWidth: 1 }},
    ]
  }},
  options: {{ responsive: true, plugins: {{ title: {{ display: true, text: 'Mean & Median Reward' }} }}, scales: {{ y: {{ beginAtZero: true, max: 0.75 }} }} }}
}});

new Chart(document.getElementById('histChart'), {{
  type: 'bar',
  data: {{
    labels: DATA.hist_labels,
    datasets: DATA.hist_datasets.map((ds, i) => ({{
      label: ds.label, data: ds.data,
      backgroundColor: DATA.colors[i] + '88',
      borderColor: DATA.colors[i],
      borderWidth: 1,
    }}))
  }},
  options: {{ responsive: true, plugins: {{ title: {{ display: true, text: 'Reward Distribution' }} }}, scales: {{ x: {{ title: {{ display: true, text: 'Reward' }} }}, y: {{ title: {{ display: true, text: 'Count' }} }} }} }}
}});

// --- Difficulty ---
new Chart(document.getElementById('diffChart'), {{
  type: 'bar',
  data: {{
    labels: ['Easy', 'Medium', 'Hard'],
    datasets: DATA.models.map((m, i) => ({{
      label: m, data: DATA.diff_data[m],
      backgroundColor: DATA.colors[i] + 'cc',
      borderColor: DATA.colors[i], borderWidth: 1,
    }}))
  }},
  options: {{ responsive: true, scales: {{ y: {{ beginAtZero: true, max: 0.85 }} }}, plugins: {{ title: {{ display: true, text: 'Mean Reward by Difficulty' }} }} }}
}});

// --- Subscores ---
new Chart(document.getElementById('radarChart'), {{
  type: 'radar',
  data: {{
    labels: DATA.subscore_labels,
    datasets: DATA.models.map((m, i) => ({{
      label: m, data: DATA.subscore_data[m],
      borderColor: DATA.colors[i], backgroundColor: DATA.colors[i] + '22',
      pointBackgroundColor: DATA.colors[i], borderWidth: 2,
    }}))
  }},
  options: {{ responsive: true, scales: {{ r: {{ beginAtZero: true, max: 1.0, ticks: {{ stepSize: 0.2 }} }} }}, plugins: {{ title: {{ display: true, text: 'Subscore Radar' }} }} }}
}});

new Chart(document.getElementById('subscoreBarChart'), {{
  type: 'bar',
  data: {{
    labels: DATA.subscore_labels,
    datasets: DATA.models.map((m, i) => ({{
      label: m, data: DATA.subscore_data[m],
      backgroundColor: DATA.colors[i] + 'cc', borderColor: DATA.colors[i], borderWidth: 1,
    }}))
  }},
  options: {{ responsive: true, scales: {{ y: {{ beginAtZero: true, max: 1.1 }} }}, plugins: {{ title: {{ display: true, text: 'Subscore Breakdown' }} }} }}
}});

// --- Subscore Table ---
document.getElementById('subscoreHead').innerHTML = '<th>Subscore</th><th>Weight</th>' + DATA.models.map(m => '<th>' + m + '</th>').join('');
const stBody = document.getElementById('subscoreBody');
DATA.subscore_names.forEach((s, idx) => {{
  const vals = DATA.models.map(m => DATA.subscore_data[m][idx]);
  const maxVal = Math.max(...vals);
  const minVal = Math.min(...vals);
  const weight = [40, 25, 10, 15, 10][idx];
  let row = `<tr><td>${{s.replace(/_/g, ' ')}}</td><td>${{weight}}%</td>`;
  vals.forEach(v => {{
    const cls = v === maxVal ? 'best' : (v === minVal ? 'worst' : '');
    row += `<td class="${{cls}}">${{v.toFixed(4)}}</td>`;
  }});
  stBody.innerHTML += row + '</tr>';
}});

// --- Distribution Stats Table ---
document.getElementById('distHead').innerHTML = '<th>Metric</th>' + DATA.models.map(m => '<th>' + m + '</th>').join('');
const dtBody = document.getElementById('distBody');
const metrics = [
  ['Completed', m => `${{DATA.summary[m].n}}/${{DATA.summary[m].total}}`],
  ['Mean', m => DATA.summary[m].mean.toFixed(4)],
  ['Median', m => DATA.summary[m].median.toFixed(4)],
  ['Stdev', m => DATA.summary[m].stdev.toFixed(4)],
  ['P25', m => DATA.summary[m].p25.toFixed(4)],
  ['P75', m => DATA.summary[m].p75.toFixed(4)],
  ['Min', m => DATA.summary[m].min.toFixed(4)],
  ['Max', m => DATA.summary[m].max.toFixed(4)],
  ['Zero rewards', m => DATA.summary[m].zeros],
  ['High rewards (>=0.8)', m => `${{DATA.summary[m].highs}} (${{DATA.summary[m].high_rate}}%)`],
  ['Avg steps', m => DATA.summary[m].avg_steps],
];
metrics.forEach(([label, fn]) => {{
  let row = `<tr><td><strong>${{label}}</strong></td>`;
  DATA.models.forEach(m => {{ row += `<td>${{fn(m)}}</td>`; }});
  dtBody.innerHTML += row + '</tr>';
}});

// --- Category Heatmap ---
document.getElementById('catHead').innerHTML = '<th onclick="sortCatTable(0)">Category</th><th onclick="sortCatTable(1)">N</th>' + DATA.models.map((m, i) => `<th onclick="sortCatTable(${{i+2}})">${{m}}</th>`).join('');

function heatColor(val) {{
  if (val === null) return 'background:var(--bg3);color:var(--fg3)';
  const r = val < 0.5 ? 255 : Math.round(255 * (1 - val) * 2);
  const g = val > 0.5 ? 200 : Math.round(200 * val * 2);
  return `background:rgba(${{r}},${{g}},60,0.2);color:rgb(${{Math.min(r+40,255)}},${{Math.min(g+40,255)}},80)`;
}}

function renderCatTable(filter) {{
  const tbody = document.getElementById('catBody');
  tbody.innerHTML = '';
  DATA.cat_data.filter(c => !filter || c.category.includes(filter.toLowerCase())).forEach(c => {{
    const n = Math.max(...DATA.models.map(m => c[m + '_n'] || 0));
    let row = `<tr><td><strong>${{c.category}}</strong></td><td>${{n}}</td>`;
    DATA.models.forEach(m => {{
      const v = c[m];
      if (v === null) {{ row += '<td><span class="heatcell" style="' + heatColor(null) + '">-</span></td>'; }}
      else {{ row += `<td><span class="heatcell" style="${{heatColor(v)}}">${{v.toFixed(3)}}</span></td>`; }}
    }});
    tbody.innerHTML += row + '</tr>';
  }});
}}
renderCatTable('');
document.getElementById('catFilter').addEventListener('input', e => renderCatTable(e.target.value));

let sortDir = 1;
function sortCatTable(col) {{
  sortDir *= -1;
  DATA.cat_data.sort((a, b) => {{
    if (col === 0) return sortDir * a.category.localeCompare(b.category);
    if (col === 1) return sortDir * ((Math.max(...DATA.models.map(m => a[m + '_n'] || 0))) - (Math.max(...DATA.models.map(m => b[m + '_n'] || 0))));
    const m = DATA.models[col - 2];
    return sortDir * ((a[m] || 0) - (b[m] || 0));
  }});
  renderCatTable(document.getElementById('catFilter').value);
}}

// --- Key Findings ---
document.getElementById('findings').innerHTML = `
<h2>Key Findings</h2>
<div class="findings"><ul>
  <li><strong>Claude Sonnet 4.6 leads overall</strong> with a mean reward of ${{DATA.summary['Claude Sonnet 4.6'].mean.toFixed(4)}} and the most consistent performance across all difficulty levels.</li>
  <li><strong>Qwen3.5 vLLM improves dramatically over SGLang</strong> &mdash; mean reward jumped from ${{DATA.summary['Qwen3.5 (SGLang)'].mean.toFixed(4)}} to ${{DATA.summary['Qwen3.5 (vLLM)'].mean.toFixed(4)}} (+${{(DATA.summary['Qwen3.5 (vLLM)'].mean - DATA.summary['Qwen3.5 (SGLang)'].mean).toFixed(4)}}), now matching Claude Haiku 4.5 (${{DATA.summary['Claude Haiku 4.5'].mean.toFixed(4)}}).</li>
  <li><strong>Category matching remains Qwen3.5's main weakness</strong> (SGLang: ${{DATA.subscore_data['Qwen3.5 (SGLang)'][0].toFixed(4)}}, vLLM: ${{DATA.subscore_data['Qwen3.5 (vLLM)'][0].toFixed(4)}}) &mdash; the heaviest-weighted subscore at 40%.</li>
  <li><strong>Explanation quality is uniformly high</strong> across all models (${{Math.min(...DATA.models.map(m => DATA.subscore_data[m][1])).toFixed(3)}}&ndash;${{Math.max(...DATA.models.map(m => DATA.subscore_data[m][1])).toFixed(3)}}), suggesting this subscore may not differentiate models well.</li>
  <li><strong>vLLM improved Qwen3.5's zero-reward count</strong> from ${{DATA.summary['Qwen3.5 (SGLang)'].zeros}} to ${{DATA.summary['Qwen3.5 (vLLM)'].zeros}} failed submissions, indicating better tool-calling reliability.</li>
  <li><strong>Sonnet 4.6 is uniquely flat across difficulties</strong> (easy: ${{DATA.diff_data['Claude Sonnet 4.6'][0].toFixed(3)}}, hard: ${{DATA.diff_data['Claude Sonnet 4.6'][2].toFixed(3)}}), while other models drop significantly from easy to hard.</li>
</ul></div>
`;
</script>
</div>
</body>
</html>"""

    return html


def main():
    print("Loading result files...")
    models_data = {}
    for name, path in RESULT_FILES.items():
        print(f"  {name}: {path}")
        models_data[name] = load_model_data(path)

    print("Generating HTML...")
    html = generate_html(models_data)

    out_path = Path("results/index.html")
    out_path.write_text(html)
    print(f"Saved to {out_path} ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
