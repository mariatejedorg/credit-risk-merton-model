"""Visualización estática: valor de los activos recuperado (V_t) frente a
la capitalización bursátil observada (E_t), y un resumen de Distance to
Default / probabilidad de impago."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


def plot_assets_vs_equity(V: pd.Series, E: pd.Series, D: float, ticker: str) -> None:
    """V_t siempre por encima de E_t: el equity es solo la parte "opción"
    del valor de los activos, la diferencia es la deuda que los acreedores
    tienen prioridad para cobrar."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(V.index, V.values, color="tab:blue", linewidth=1.8, label="Activos recuperados (V_t)")
    ax.plot(E.index, E.values, color="tab:green", linewidth=1.8, label="Capitalización bursátil (E_t)")
    ax.axhline(D, color="tab:red", linewidth=1.2, linestyle="--", label=f"Punto de impago (D = {D:,.0f})")

    ax.set_title(f"Merton: activos recuperados vs. equity observado — {ticker}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    OUTPUTS_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUTS_DIR / "assets_vs_equity.png", dpi=150)
    plt.close(fig)


def plot_default_probability_gauge(pd_real_world: float, pd_risk_neutral: float, dd: float, ticker: str) -> None:
    """Barra horizontal comparando la probabilidad de impago real-world y
    neutral al riesgo, con la Distance to Default como referencia numérica."""
    fig, ax = plt.subplots(figsize=(10, 4))

    # Puntos básicos (1 bp = 0.01%): para PD tan bajas como las de un
    # emisor de bajo riesgo, se lee mejor que un porcentaje con muchos ceros.
    labels = ["PD real-world\n(con mu_V)", "PD neutral al riesgo\n(con r)"]
    values = [pd_real_world * 10_000, pd_risk_neutral * 10_000]
    colors = ["tab:blue", "tab:orange"]

    bars = ax.barh(labels, values, color=colors)
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{value:.1f} bps", va="center")

    ax.set_xlabel("Probabilidad de impago a 1 año (puntos básicos)")
    ax.set_title(f"Merton — {ticker}: Distance to Default = {dd:.2f}")
    ax.set_xlim(0, max(values) * 1.3 if max(values) > 0 else 1)
    ax.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    OUTPUTS_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUTS_DIR / "default_probability.png", dpi=150)
    plt.close(fig)
