#!/bin/python
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import numpy as np


def _to_iso_list(values):
    out = []
    for value in values:
        if hasattr(value, 'strftime'):
            out.append(value.strftime('%Y-%m-%d'))
        elif isinstance(value, np.datetime64):
            out.append(np.datetime_as_string(value, unit='D'))
        else:
            out.append(str(value)[:10])
    return out


def _to_number_list(values):
    arr = np.asarray(values, dtype=float).reshape(-1)
    return [None if not np.isfinite(v) else float(v) for v in arr]


def _to_jsonable(value):
    if hasattr(value, 'strftime'):
        return value.strftime('%Y-%m-%d')
    if isinstance(value, np.datetime64):
        return np.datetime_as_string(value, unit='D')
    if isinstance(value, np.ndarray):
        return [_to_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _to_jsonable(value.item())
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def make_trace(name, x, y, mode='lines', color=None, dash=None,
               width=1.2, marker_size=4, error_y=None,
               customdata=None, hovertemplate=None, showlegend=True):
    trace = {
        'type': 'scatter',
        'mode': mode,
        'name': name,
        'legendgroup': name,
        'showlegend': showlegend,
        'x': _to_iso_list(x),
        'y': _to_number_list(y),
    }
    if 'lines' in mode:
        trace['line'] = {'width': width}
        if color is not None:
            trace['line']['color'] = color
        if dash is not None:
            trace['line']['dash'] = dash
    if 'markers' in mode:
        trace['marker'] = {'size': marker_size}
        if color is not None:
            trace['marker']['color'] = color
    if error_y is not None:
        trace['error_y'] = {
            'type': 'data',
            'array': _to_number_list(error_y),
            'visible': True,
            'color': color or 'gray',
            'thickness': 0.6,
        }
    if customdata is not None:
        trace['customdata'] = _to_number_list(customdata)
    if hovertemplate is not None:
        trace['hovertemplate'] = hovertemplate
    return trace


def write_plotly_panels(output_path, title, panels, columns=2, yaxis_title='', height=340):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_to_jsonable(panels), ensure_ascii=False, allow_nan=False)
    safe_title = json.dumps(title, ensure_ascii=False)
    safe_yaxis_title = json.dumps(yaxis_title, ensure_ascii=False)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    h1 {{ font-size: 22px; margin-bottom: 18px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat({columns}, minmax(360px, 1fr));
      gap: 18px;
    }}
    .plot {{ width: 100%; height: {height}px; }}
  </style>
</head>
<body>
  <h1 id="title"></h1>
  <div class="grid" id="grid"></div>
  <script>
    const title = {safe_title};
    const yaxisTitle = {safe_yaxis_title};
    const panels = {payload};
    document.getElementById('title').textContent = title;
    const grid = document.getElementById('grid');

    panels.forEach((panel, i) => {{
      const div = document.createElement('div');
      div.className = 'plot';
      div.id = `plot-${{i}}`;
      grid.appendChild(div);

      const shapes = [];
      (panel.vlines || []).forEach((x) => {{
        shapes.push({{
          type: 'line',
          xref: 'x',
          yref: 'paper',
          x0: x,
          x1: x,
          y0: 0,
          y1: 1,
          line: {{ color: 'gray', width: 1, dash: 'dot' }}
        }});
      }});
      (panel.hlines || []).forEach((y) => {{
        shapes.push({{
          type: 'line',
          xref: 'paper',
          yref: 'y',
          x0: 0,
          x1: 1,
          y0: y,
          y1: y,
          line: {{ color: 'black', width: 1 }}
        }});
      }});

      const layout = {{
        title: {{ text: panel.title, font: {{ size: 14 }} }},
        margin: {{ l: 62, r: 22, t: 48, b: 48 }},
        hovermode: 'x unified',
        xaxis: {{ title: 'Date' }},
        yaxis: {{ title: yaxisTitle }},
        shapes: shapes,
        legend: {{ orientation: 'h', y: -0.25 }}
      }};
      Plotly.newPlot(div.id, panel.traces, layout, {{ responsive: true }});
    }});
  </script>
</body>
</html>
"""
    output_path.write_text(html, encoding='utf-8')
    print('interactive html:', output_path)
