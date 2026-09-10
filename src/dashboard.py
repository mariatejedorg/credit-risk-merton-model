"""Dashboard HTML interactivo con Plotly: un único archivo autocontenido en
outputs/. Mismos tokens de color y helpers de layout que
proyecto-6-opciones-volatilidad-implicita/src/dashboard.py, copiados
literalmente para dar continuidad visual entre proyectos del portfolio.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"

BLUE = "#2a78d6"
GREEN = "#1baf7a"
RED = "#e34948"
ORANGE = "#e3a648"

SURFACE = "#fcfcfb"
PAGE_PLANE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BORDER = "rgba(11,11,11,0.10)"

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _base_layout(title: str, **extra) -> dict:
    layout = dict(
        title=dict(text=title, font=dict(family=FONT_FAMILY, size=15, color=INK_PRIMARY)),
        font=dict(family=FONT_FAMILY, size=12, color=INK_SECONDARY),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        legend=dict(font=dict(color=INK_SECONDARY, size=11), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=50, r=30, t=50, b=40),
    )
    layout.update(extra)
    return layout


def _axis(**extra) -> dict:
    axis = dict(
        gridcolor=GRIDLINE,
        gridwidth=1,
        linecolor=BASELINE,
        tickfont=dict(color=INK_MUTED, size=11),
        title_font=dict(color=INK_SECONDARY, size=12),
        zeroline=False,
    )
    axis.update(extra)
    return axis


def _assets_vs_equity_figure(V: pd.Series, E: pd.Series, D: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=V.index, y=V.values, mode="lines", name="Activos recuperados (V_t)",
                              line=dict(color=BLUE, width=2)))
    fig.add_trace(go.Scatter(x=E.index, y=E.values, mode="lines", name="Capitalización bursátil (E_t)",
                              line=dict(color=GREEN, width=2)))
    fig.add_hline(y=D, line_color=RED, line_width=1.5, line_dash="dot",
                  annotation_text="Punto de impago (D)", annotation_font_color=RED)

    fig.update_layout(
        **_base_layout(
            "Activos recuperados vs. equity observado",
            xaxis=_axis(title="Fecha"),
            yaxis=_axis(title="Valor"),
            hovermode="x unified",
            hoverlabel=dict(bgcolor=SURFACE, font=dict(color=INK_PRIMARY, family=FONT_FAMILY)),
            height=450,
        )
    )
    return fig


def _default_probability_figure(pd_real_world: float, pd_risk_neutral: float) -> go.Figure:
    # Para emisores de bajo riesgo la PD a 1 año suele ser una fracción de
    # 1% -- en puntos básicos (1 bp = 0.01%) se lee mucho mejor que en "%",
    # convención estándar en riesgo de crédito para cifras tan pequeñas.
    labels = ["PD real-world (con mu_V)", "PD neutral al riesgo (con r)"]
    values_bps = [pd_real_world * 10_000, pd_risk_neutral * 10_000]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values_bps, y=labels, orientation="h",
        marker=dict(color=[BLUE, ORANGE]),
        text=[f"{v:.1f} bps" for v in values_bps], textposition="outside",
    ))

    fig.update_layout(
        **_base_layout(
            "Probabilidad de impago a 1 año",
            xaxis=_axis(title="Probabilidad (puntos básicos)"),
            yaxis=_axis(),
            height=300,
        )
    )
    return fig


def _stat_tile(label: str, value: str, sublabel: str) -> str:
    return f"""<div class="tile">
  <div class="tile-label">{label}</div>
  <div class="tile-value">{value}</div>
  <div class="tile-sublabel">{sublabel}</div>
</div>"""


def _kpi_tiles_html(ticker: str, E: float, D: float, V: float, sigma_V: float, dd: float) -> str:
    tiles = [
        _stat_tile("Empresa", ticker, f"Capitalización: {E:,.0f}"),
        _stat_tile("Punto de impago (D)", f"{D:,.0f}", "pasivo corriente + 0.5 x deuda LP"),
        _stat_tile("Activos estimados (V)", f"{V:,.0f}", f"ratio V/D: {V / D:.2f}x"),
        _stat_tile("Volatilidad de activos", f"{sigma_V * 100:.1f}%", "recuperada (KMV)"),
        _stat_tile("Distance to Default", f"{dd:.2f}", "desviaciones típicas hasta D"),
    ]
    return '<div class="tiles">' + "".join(tiles) + "</div>"


def build_dashboard(
    ticker: str,
    V: pd.Series,
    E: pd.Series,
    D: float,
    sigma_V: float,
    dd: float,
    pd_real_world: float,
    pd_risk_neutral: float,
) -> Path:
    assets_html = pio.to_html(
        _assets_vs_equity_figure(V, E, D), full_html=False, include_plotlyjs="cdn", config={"displaylogo": False}
    )
    pd_html = pio.to_html(
        _default_probability_figure(pd_real_world, pd_risk_neutral), full_html=False, include_plotlyjs=False, config={"displaylogo": False}
    )
    tiles_html = _kpi_tiles_html(ticker, float(E.iloc[-1]), D, float(V.iloc[-1]), sigma_V, dd)

    page = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Proyecto 7 — Credit Risk: Merton Structural Model</title>
<style>
  :root {{
    --surface: {SURFACE};
    --page-plane: {PAGE_PLANE};
    --ink-primary: {INK_PRIMARY};
    --ink-secondary: {INK_SECONDARY};
    --ink-muted: {INK_MUTED};
    --gridline: {GRIDLINE};
    --border: {BORDER};
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: {FONT_FAMILY};
    margin: 0;
    background: var(--page-plane);
    color: var(--ink-primary);
  }}
  .wrap {{ max-width: 1080px; margin: 0 auto; padding: 0 24px 64px; }}

  .hero {{
    background: linear-gradient(135deg, #123a2e 0%, #1baf7a 100%);
    color: #ffffff;
    padding: 48px 24px 40px;
    margin-bottom: 28px;
  }}
  .hero-inner {{ max-width: 1080px; margin: 0 auto; }}
  .hero h1 {{ font-size: 1.75rem; margin: 0 0 8px; font-weight: 700; }}
  .hero p {{ margin: 0; color: rgba(255,255,255,0.85); font-size: 0.95rem; }}
  .hero .meta {{ margin-top: 18px; font-size: 0.8rem; color: rgba(255,255,255,0.65); letter-spacing: 0.02em; }}

  .tiles {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 14px;
    margin: 0 0 28px;
  }}
  .tile {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 18px;
  }}
  .tile-label {{ font-size: 0.72rem; color: var(--ink-muted); text-transform: uppercase; letter-spacing: 0.04em; }}
  .tile-value {{ font-size: 1.55rem; font-weight: 700; color: var(--ink-primary); margin: 6px 0 2px; }}
  .tile-sublabel {{ font-size: 0.8rem; color: var(--ink-secondary); }}

  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 22px;
  }}
  .card h2 {{ font-size: 1.05rem; margin: 0 0 16px; color: var(--ink-primary); font-weight: 600; }}

  footer {{ text-align: center; font-size: 0.78rem; color: var(--ink-muted); padding-top: 8px; }}
</style>
</head>
<body>

<div class="hero">
  <div class="hero-inner">
    <h1>Credit Risk: Merton Structural Model</h1>
    <p>El equity de {ticker} tratado como una opción call sobre sus activos — reutilizando el motor de Black-Scholes del Proyecto 6 para estimar su probabilidad de impago.</p>
    <div class="meta">Datos en vivo vía yfinance (equity + balance) · metodología KMV</div>
  </div>
</div>

<div class="wrap">

{tiles_html}

<div class="card">
  <h2>Activos recuperados vs. equity observado</h2>
  {assets_html}
</div>

<div class="card">
  <h2>Probabilidad de impago a 1 año</h2>
  {pd_html}
</div>

<footer>Proyecto 7 · Roadmap Quant · Python (numpy, scipy, pandas, yfinance, Plotly)</footer>

</div>
</body>
</html>"""

    OUTPUTS_DIR.mkdir(exist_ok=True)
    out_path = OUTPUTS_DIR / "dashboard.html"
    out_path.write_text(page, encoding="utf-8")
    return out_path
