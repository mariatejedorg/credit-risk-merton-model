"""Descarga de los tres datos de mercado/balance que alimentan el modelo de
Merton: la serie histórica de capitalización bursátil (equity observado),
el punto de impago (deuda) del último balance publicado, y las acciones en
circulación.
"""

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.credit import HISTORICAL_PERIOD, TICKER

# Mismo mecanismo que en los Proyectos 1-6: si un antivirus que inspecciona
# tráfico HTTPS (p. ej. Norton) rompe la verificación por defecto de yfinance,
# se usa un bundle de certificados local si existe.
_CUSTOM_CA_BUNDLE = Path(__file__).resolve().parent.parent / ".certs" / "cacert.pem"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Los nombres exactos de fila en yf.Ticker(...).balance_sheet varían entre
# tickers y versiones de yfinance. Se prueban por orden de preferencia y se
# usa el primero disponible con un valor no nulo en el balance más reciente.
_CURRENT_LIABILITIES_ALIASES = ["Current Liabilities", "Total Current Liabilities"]
_LONG_TERM_DEBT_ALIASES = ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"]
_TOTAL_DEBT_ALIASES = ["Total Debt"]


def _build_session():
    if not _CUSTOM_CA_BUNDLE.exists():
        return None
    from curl_cffi import requests as curl_requests

    return curl_requests.Session(impersonate="chrome", verify=str(_CUSTOM_CA_BUNDLE))


def _first_available(balance_sheet: pd.DataFrame, aliases: list[str], column) -> float | None:
    for name in aliases:
        if name in balance_sheet.index:
            value = balance_sheet.loc[name, column]
            if pd.notna(value):
                return float(value)
    return None


def get_market_cap_series(ticker: str = TICKER, period: str = HISTORICAL_PERIOD) -> pd.Series:
    """Serie histórica de capitalización bursátil: precio de cierre x
    acciones en circulación actuales.

    Simplificación: se usa el número de acciones en circulación ACTUAL para
    todo el histórico (no varía día a día en los datos gratuitos de
    yfinance). Para horizontes de 1-2 años y empresas sin recompras/
    emisiones masivas, el error introducido es pequeño frente al objetivo
    del modelo (una estimación de sigma_V, no una contabilidad exacta).
    """
    yf_ticker = yf.Ticker(ticker, session=_build_session())
    prices = yf_ticker.history(period=period, auto_adjust=True)["Close"]
    shares = yf_ticker.fast_info["shares"]
    return (prices * shares).dropna()


def get_default_point(ticker: str = TICKER) -> float:
    """Punto de impago D, convención estándar de la metodología KMV:

        D = pasivo corriente + 0.5 x deuda a largo plazo

    (no la deuda total): la deuda a largo plazo no vence dentro del
    horizonte de 1 año del modelo, así que solo se cuenta a medias como
    presión de impago a corto plazo -- es la aproximación práctica de KMV
    a "cuánta deuda hay que poder pagar en el próximo año".

    Si el balance no desglosa pasivo corriente / deuda a largo plazo (varía
    según el ticker), se cae a la deuda total (.info/balance_sheet) como
    aproximación menos precisa pero siempre disponible.
    """
    yf_ticker = yf.Ticker(ticker, session=_build_session())
    balance_sheet = yf_ticker.balance_sheet
    latest_period = balance_sheet.columns[0]

    current_liabilities = _first_available(balance_sheet, _CURRENT_LIABILITIES_ALIASES, latest_period)
    long_term_debt = _first_available(balance_sheet, _LONG_TERM_DEBT_ALIASES, latest_period)

    if current_liabilities is not None and long_term_debt is not None:
        return current_liabilities + 0.5 * long_term_debt

    total_debt = _first_available(balance_sheet, _TOTAL_DEBT_ALIASES, latest_period)
    if total_debt is not None:
        return total_debt

    raise ValueError(f"No se encontró desglose de deuda ni deuda total en el balance de {ticker}")


def save_snapshot(market_cap: pd.Series, ticker: str = TICKER) -> Path:
    """Vuelca la serie de capitalización bursátil a data/ como registro de
    la ejecución (para inspección posterior), no como caché de lectura --
    igual que option_chain_<TICKER>.csv en Proyecto 6."""
    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / f"market_cap_{ticker}.csv"
    market_cap.to_frame(name="market_cap").to_csv(out_path)
    return out_path


if __name__ == "__main__":
    market_cap = get_market_cap_series()
    default_point = get_default_point()

    print(f"Capitalización bursátil actual: {market_cap.iloc[-1]:,.0f}")
    print(f"Punto de impago (D): {default_point:,.0f}")
    print(f"Ratio D / E actual: {default_point / market_cap.iloc[-1]:.2f}x")
