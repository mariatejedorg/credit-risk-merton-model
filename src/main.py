"""Punto de entrada: descarga equity y balance reales, recupera el valor y
la volatilidad de los activos con el procedimiento iterativo de Merton/KMV,
calcula la Distance to Default y la probabilidad de impago, y genera las
visualizaciones."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.credit import RISK_FREE_RATE, TICKER, TIME_HORIZON_YEARS

from dashboard import build_dashboard
from data import get_default_point, get_market_cap_series, save_snapshot
from merton import (
    distance_to_default,
    estimate_assets,
    log_returns,
    probability_of_default,
    probability_of_default_risk_neutral,
    TRADING_DAYS_PER_YEAR,
)
from visualize import plot_assets_vs_equity, plot_default_probability_gauge


def main() -> None:
    E = get_market_cap_series()
    D = get_default_point()
    save_snapshot(E)

    print("=== Datos descargados ===")
    print(f"Ticker: {TICKER}")
    print(f"Capitalización bursátil actual (E): {E.iloc[-1]:,.0f}")
    print(f"Punto de impago (D): {D:,.0f}")
    print(f"Ratio D/E actual: {D / E.iloc[-1]:.2f}x")

    V, sigma_V = estimate_assets(E, D, TIME_HORIZON_YEARS, RISK_FREE_RATE)
    mu_V = float(log_returns(V).mean() * TRADING_DAYS_PER_YEAR)

    print("\n=== Activos recuperados (procedimiento iterativo KMV) ===")
    print(f"Valor de activos actual (V): {V.iloc[-1]:,.0f}")
    print(f"Volatilidad de activos (sigma_V): {sigma_V:.2%}")
    print(f"Deriva de activos estimada (mu_V): {mu_V:+.2%}")

    dd = distance_to_default(V.iloc[-1], D, TIME_HORIZON_YEARS, mu_V, sigma_V)
    pd_real_world = probability_of_default(dd)
    pd_risk_neutral = probability_of_default_risk_neutral(V.iloc[-1], D, TIME_HORIZON_YEARS, RISK_FREE_RATE, sigma_V)

    print("\n=== Riesgo de crédito (horizonte 1 año) ===")
    print(f"Distance to Default: {dd:.2f}")
    print(f"Probabilidad de impago real-world: {pd_real_world:.2%}")
    print(f"Probabilidad de impago neutral al riesgo: {pd_risk_neutral:.2%}")

    plot_assets_vs_equity(V, E, D, TICKER)
    plot_default_probability_gauge(pd_real_world, pd_risk_neutral, dd, TICKER)
    dashboard_path = build_dashboard(TICKER, V, E, D, sigma_V, dd, pd_real_world, pd_risk_neutral)

    print(f"\nGráficos guardados en outputs/ (dashboard interactivo: {dashboard_path})")


if __name__ == "__main__":
    main()
