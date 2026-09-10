"""Modelo estructural de Merton (1974) para riesgo de crédito: trata el
equity de una empresa como una opción call sobre el valor de sus activos,
con precio de ejercicio igual al punto de impago (deuda).

    E = V x N(d1) - D x e^(-rT) x N(d2)   =   bs_price(S=V, K=D, T, r, sigma_V, "call")

V (valor de los activos) y sigma_V (su volatilidad) no se observan
directamente -- solo se observan E (capitalización bursátil) y sigma_E
(volatilidad del equity, estimable del histórico de precios). Se recuperan
con el procedimiento iterativo estándar de la industria (metodología KMV):

    1. sigma_V inicial = sigma_E * E / (E + D)
    2. Con sigma_V fijo, invertir E_t = BS(V_t) día a día -> serie V_t
    3. Reestimar sigma_V = volatilidad anualizada de los retornos de V_t
    4. Repetir 2-3 hasta que sigma_V converja

Reutiliza bs_price y delta de black_scholes.py (Proyecto 6) sin
modificarlos: la única diferencia con la inversión de volatilidad implícita
de implied_vol.py es que aquí la incógnita del paso 2 es el SUBYACENTE (V),
no la volatilidad -- Black-Scholes sigue siendo la misma fórmula.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.credit import (
    MAX_ITERATIONS,
    NEWTON_MAX_ITERATIONS,
    NEWTON_TOLERANCE,
    SIGMA_V_TOLERANCE,
)


def _load_from_proyecto(relative_project: str, module_filename: str, module_alias: str):
    """Carga un módulo de otro proyecto del portfolio por ruta de archivo,
    sin contaminar sys.path ni sys.modules con los paquetes "config" de ese
    otro proyecto -- mismo mecanismo (y mismo motivo) que
    proyecto-6-opciones-volatilidad-implicita/src/monte_carlo_check.py, que
    ya tuvo que resolver exactamente este problema de colisión de nombres.
    """
    module_path = Path(__file__).resolve().parents[2] / relative_project / "src" / module_filename

    saved_config_modules = {
        name: mod for name, mod in sys.modules.items()
        if name == "config" or name.startswith("config.")
    }
    for name in saved_config_modules:
        del sys.modules[name]

    try:
        spec = importlib.util.spec_from_file_location(module_alias, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name in [n for n in sys.modules if n == "config" or n.startswith("config.")]:
            del sys.modules[name]
        sys.modules.update(saved_config_modules)

    return module


_black_scholes = _load_from_proyecto(
    "proyecto-6-opciones-volatilidad-implicita", "black_scholes.py", "proyecto6_black_scholes"
)
bs_price = _black_scholes.bs_price
delta = _black_scholes.delta

_estimation = _load_from_proyecto(
    "proyecto-3-monte-carlo", "estimation.py", "proyecto3_estimation"
)
log_returns = _estimation.log_returns
TRADING_DAYS_PER_YEAR = _estimation.TRADING_DAYS_PER_YEAR


def solve_asset_value(E: float, D: float, T: float, r: float, sigma_V: float, V_guess: float) -> float:
    """Newton-Raphson: recupera el valor de los activos V tal que
    bs_price(V, D, T, r, sigma_V, "call") = E.

        V_(n+1) = V_n - (BS(V_n) - E) / Delta(V_n)

    Delta = dPrecio/dV (misma función que en black_scholes.py; aquí V hace
    el papel de "spot" del subyacente, y E el papel de "precio de la opción").
    """
    V = V_guess

    for _ in range(NEWTON_MAX_ITERATIONS):
        price_error = bs_price(V, D, T, r, sigma_V, "call") - E

        if abs(price_error) < NEWTON_TOLERANCE:
            return V

        d = delta(V, D, T, r, sigma_V, "call")
        if d < 1e-8:
            break

        V = V - price_error / d

    return V


def estimate_assets(equity_series: pd.Series, D: float, T: float, r: float) -> tuple[pd.Series, float]:
    """Procedimiento iterativo KMV completo: devuelve la serie diaria de
    valor de activos V_t y la volatilidad anualizada de activos sigma_V,
    ambas consistentes entre sí (el punto fijo del sistema E = BS(V, sigma_V)
    y sigma_V = vol(V_t)).
    """
    E = equity_series.to_numpy()
    E_returns_sigma = float(log_returns(equity_series).std() * np.sqrt(TRADING_DAYS_PER_YEAR))

    # Arranque: aproximación de libro de texto para sigma_V (KMV la usa como
    # punto de partida, no como resultado final).
    sigma_V = E_returns_sigma * E[-1] / (E[-1] + D)

    V = E + D  # primera aproximación de V_t: equity + deuda, se refina abajo

    for _ in range(MAX_ITERATIONS):
        V_new = np.empty_like(E)
        V_guess = V[0]
        for i in range(len(E)):
            V_guess = solve_asset_value(E[i], D, T, r, sigma_V, V_guess)
            V_new[i] = V_guess

        V_series = pd.Series(V_new, index=equity_series.index)
        sigma_V_new = float(log_returns(V_series).std() * np.sqrt(TRADING_DAYS_PER_YEAR))

        relative_change = abs(sigma_V_new - sigma_V) / sigma_V
        V, sigma_V = V_new, sigma_V_new

        if relative_change < SIGMA_V_TOLERANCE:
            break

    return pd.Series(V, index=equity_series.index), sigma_V


def distance_to_default(V: float, D: float, T: float, mu_V: float, sigma_V: float) -> float:
    """Distance to Default: cuántas desviaciones típicas separan el valor
    esperado de los activos en T del punto de impago D. Mismo numerador que
    d2 de Black-Scholes, pero con la deriva REAL de los activos (mu_V) en
    vez del tipo libre de riesgo -- es una medida real-world, no neutral al
    riesgo (ver probability_of_default_risk_neutral para la versión con r).
    """
    return (np.log(V / D) + (mu_V - 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))


def probability_of_default(dd: float) -> float:
    """PD real-world = N(-DD): probabilidad de que el valor de los activos
    caiga por debajo del punto de impago en el horizonte T, bajo la medida
    real (no neutral al riesgo)."""
    return float(norm.cdf(-dd))


def probability_of_default_risk_neutral(V: float, D: float, T: float, r: float, sigma_V: float) -> float:
    """PD neutral al riesgo = N(-d2), el mismo d2 que en el precio de
    Black-Scholes (con r en vez de mu_V). Sistemáticamente mayor que la PD
    real-world porque incorpora una prima de riesgo -- la misma relación
    que entre probabilidades de ejercicio real y neutrales al riesgo de
    cualquier opción."""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(V / D) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * sqrt_T)
    d2 = d1 - sigma_V * sqrt_T
    return float(norm.cdf(-d2))


if __name__ == "__main__":
    # Verificación round-trip: genero una serie sintética de equity a partir
    # de un V_t simulado (GBM con sigma_V_real conocida) y compruebo que
    # estimate_assets recupera tanto la serie V_t como su volatilidad.
    #
    # Ojo: el objetivo correcto de comparación NO es sigma_V_real (el
    # parámetro poblacional del GBM), sino la volatilidad REALIZADA de ese
    # único path simulado -- con 252 observaciones, la vol realizada de una
    # trayectoria concreta se desvía de su parámetro generador por puro
    # ruido de muestreo (error estándar relativo ~ 1/sqrt(2*252) ~ 4-5%),
    # igual que en cualquier estimación de volatilidad histórica. Por eso
    # se imprimen los dos: sigma_V_real (el parámetro) da contexto, pero la
    # comprobación real es "sigma_V recuperada ~= vol. realizada del path".
    rng = np.random.default_rng(42)
    n_days = 252
    D, T, r = 800.0, 1.0, 0.045
    sigma_V_real = 0.25
    mu_V_real = 0.06

    dt = 1 / TRADING_DAYS_PER_YEAR
    log_returns_sim = (mu_V_real - 0.5 * sigma_V_real**2) * dt + sigma_V_real * np.sqrt(dt) * rng.standard_normal(n_days)
    V_real = 1000.0 * np.exp(np.cumsum(log_returns_sim))
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")

    E_synthetic = pd.Series(
        [bs_price(v, D, T, r, sigma_V_real, "call") for v in V_real], index=dates
    )

    V_recovered, sigma_V_recovered = estimate_assets(E_synthetic, D, T, r)

    V_series_real = pd.Series(V_real, index=dates)
    realized_vol = float(log_returns(V_series_real).std() * np.sqrt(TRADING_DAYS_PER_YEAR))

    print(f"sigma_V (parámetro poblacional del GBM) = {sigma_V_real:.4f}")
    print(f"sigma_V (volatilidad realizada de ESTE path) = {realized_vol:.4f}  <- objetivo real de la comparación")
    print(f"sigma_V recuperada por estimate_assets()      = {sigma_V_recovered:.4f}")
    print(f"V real (último día) = {V_real[-1]:,.2f} | V recuperado = {V_recovered.iloc[-1]:,.2f}")
