"""
Groww API Client
================
Interfaces with Groww's unofficial mobile API endpoints for stock trading.
Supports both LIVE trading and PAPER (simulated) trading modes.

WARNING: Use PAPER mode for testing. Live trading involves real financial risk.
Groww does not provide an official public API. This client uses the same
endpoints as the Groww mobile app. Usage is subject to Groww's Terms of Service.
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Optional
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class GrowwPaperClient:
    """
    Paper trading client that simulates Groww trades locally.
    No real money involved — safe for testing and development.
    """

    def __init__(self, initial_balance: float = 100000.0):
        self.balance = initial_balance
        self.holdings: dict[str, dict] = {}  # symbol -> {qty, avg_price}
        self.trade_history: list[dict] = []
        self.initial_balance = initial_balance
        logger.info(f"[PAPER] Paper trading initialized with ₹{initial_balance:,.2f}")

    def get_quote(self, symbol: str) -> Optional[dict]:
        """Fetch live quote from NSE via public API."""
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}.NS"
            resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            data = resp.json()
            meta = data["chart"]["result"][0]["meta"]
            return {
                "symbol": symbol,
                "price": meta.get("regularMarketPrice", 0),
                "prev_close": meta.get("chartPreviousClose", 0),
                "volume": meta.get("regularMarketVolume", 0),
                "exchange": meta.get("exchangeName", "NSE"),
                "currency": meta.get("currency", "INR"),
            }
        except Exception as e:
            logger.error(f"[PAPER] Quote fetch failed for {symbol}: {e}")
            return None

    def buy(self, symbol: str, quantity: int) -> dict:
        """Simulate a buy order."""
        quote = self.get_quote(symbol)
        if not quote:
            return {"success": False, "error": "Could not fetch price"}

        price = quote["price"]
        total_cost = price * quantity
        brokerage = max(20, total_cost * 0.0003)  # Groww-like brokerage
        total_with_charges = total_cost + brokerage

        if total_with_charges > self.balance:
            return {
                "success": False,
                "error": f"Insufficient funds. Need ₹{total_with_charges:,.2f}, have ₹{self.balance:,.2f}",
            }

        self.balance -= total_with_charges
        if symbol in self.holdings:
            existing = self.holdings[symbol]
            new_qty = existing["qty"] + quantity
            new_avg = (existing["avg_price"] * existing["qty"] + price * quantity) / new_qty
            self.holdings[symbol] = {"qty": new_qty, "avg_price": new_avg}
        else:
            self.holdings[symbol] = {"qty": quantity, "avg_price": price}

        trade = {
            "type": "BUY",
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "brokerage": brokerage,
            "total": total_with_charges,
            "timestamp": datetime.now().isoformat(),
            "mode": "PAPER",
        }
        self.trade_history.append(trade)
        logger.info(f"[PAPER] BUY {quantity} x {symbol} @ ₹{price:.2f} | Total: ₹{total_with_charges:.2f}")
        return {"success": True, "trade": trade, "remaining_balance": self.balance}

    def sell(self, symbol: str, quantity: int) -> dict:
        """Simulate a sell order."""
        if symbol not in self.holdings or self.holdings[symbol]["qty"] < quantity:
            held = self.holdings.get(symbol, {}).get("qty", 0)
            return {"success": False, "error": f"Insufficient holdings. Have {held}, want to sell {quantity}"}

        quote = self.get_quote(symbol)
        if not quote:
            return {"success": False, "error": "Could not fetch price"}

        price = quote["price"]
        proceeds = price * quantity
        brokerage = max(20, proceeds * 0.0003)
        net_proceeds = proceeds - brokerage
        avg_price = self.holdings[symbol]["avg_price"]
        pnl = (price - avg_price) * quantity - brokerage

        self.balance += net_proceeds
        self.holdings[symbol]["qty"] -= quantity
        if self.holdings[symbol]["qty"] == 0:
            del self.holdings[symbol]

        trade = {
            "type": "SELL",
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "avg_buy_price": avg_price,
            "brokerage": brokerage,
            "net_proceeds": net_proceeds,
            "pnl": pnl,
            "timestamp": datetime.now().isoformat(),
            "mode": "PAPER",
        }
        self.trade_history.append(trade)
        logger.info(f"[PAPER] SELL {quantity} x {symbol} @ ₹{price:.2f} | P&L: ₹{pnl:.2f}")
        return {"success": True, "trade": trade, "pnl": pnl, "remaining_balance": self.balance}

    def get_portfolio(self) -> dict:
        """Return current portfolio state with live P&L."""
        portfolio = []
        total_invested = 0
        total_current = 0

        for symbol, holding in self.holdings.items():
            quote = self.get_quote(symbol)
            if quote:
                current_price = quote["price"]
                invested = holding["avg_price"] * holding["qty"]
                current_val = current_price * holding["qty"]
                pnl = current_val - invested
                pnl_pct = (pnl / invested) * 100 if invested > 0 else 0
                portfolio.append({
                    "symbol": symbol,
                    "qty": holding["qty"],
                    "avg_price": holding["avg_price"],
                    "current_price": current_price,
                    "invested": invested,
                    "current_value": current_val,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                })
                total_invested += invested
                total_current += current_val

        overall_pnl = total_current - total_invested
        total_value = self.balance + total_current
        return_pct = ((total_value - self.initial_balance) / self.initial_balance) * 100

        return {
            "mode": "PAPER",
            "cash_balance": self.balance,
            "holdings": portfolio,
            "total_invested": total_invested,
            "total_current_value": total_current,
            "overall_pnl": overall_pnl,
            "total_portfolio_value": total_value,
            "return_pct": return_pct,
        }


class GrowwLiveClient:
    """
    Live trading client for Groww using the unofficial mobile API.
    Requires a valid Groww auth token.

    DANGER: This executes REAL trades with REAL money.
    Only use after thorough testing in PAPER mode.
    """

    BASE_URL = "https://groww.in/v1/api"
    SEARCH_URL = "https://groww.in/v1/api/search/v3/query/global/st-query"

    def __init__(self, auth_token: str):
        if not auth_token:
            raise ValueError("Groww auth token is required for live trading")
        self.auth_token = auth_token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
            "User-Agent": "Groww/6.0.0 (Android)",
            "x-app-source": "ANDROID",
        })
        logger.warning("[LIVE] Live Groww client initialized — REAL MONEY AT RISK")

    def _request(self, method: str, endpoint: str, **kwargs) -> Optional[dict]:
        url = f"{self.BASE_URL}{endpoint}"
        try:
            resp = self.session.request(method, url, timeout=15, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"[LIVE] HTTP {e.response.status_code} for {endpoint}: {e}")
            return None
        except Exception as e:
            logger.error(f"[LIVE] Request failed for {endpoint}: {e}")
            return None

    def get_portfolio(self) -> Optional[dict]:
        """Fetch live portfolio from Groww."""
        data = self._request("GET", "/stocks/portfolio/v2/holdings")
        return data

    def get_quote(self, symbol: str) -> Optional[dict]:
        """Get live stock quote (uses Yahoo Finance as fallback)."""
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}.NS"
            resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            data = resp.json()
            meta = data["chart"]["result"][0]["meta"]
            return {
                "symbol": symbol,
                "price": meta.get("regularMarketPrice", 0),
                "prev_close": meta.get("chartPreviousClose", 0),
                "volume": meta.get("regularMarketVolume", 0),
            }
        except Exception as e:
            logger.error(f"[LIVE] Quote fetch failed for {symbol}: {e}")
            return None

    def get_funds(self) -> Optional[dict]:
        """Get available funds."""
        return self._request("GET", "/stocks/user/funds")

    def place_order(self, symbol: str, quantity: int, order_type: str,
                    transaction_type: str, price: Optional[float] = None) -> Optional[dict]:
        """
        Place a stock order on Groww.

        Args:
            symbol: NSE symbol (e.g., 'RELIANCE')
            quantity: Number of shares
            order_type: 'MARKET' or 'LIMIT'
            transaction_type: 'BUY' or 'SELL'
            price: Required for LIMIT orders
        """
        payload = {
            "exchange": "NSE",
            "tradingsymbol": symbol,
            "quantity": quantity,
            "order_type": order_type,
            "transaction_type": transaction_type,
            "product": "CNC",  # Cash and Carry (delivery)
            "validity": "DAY",
        }
        if order_type == "LIMIT" and price:
            payload["price"] = price

        logger.warning(f"[LIVE] Placing {transaction_type} order: {quantity} x {symbol}")
        return self._request("POST", "/stocks/order/place", json=payload)

    def buy(self, symbol: str, quantity: int) -> dict:
        result = self.place_order(symbol, quantity, "MARKET", "BUY")
        if result:
            return {"success": True, "order": result}
        return {"success": False, "error": "Order placement failed"}

    def sell(self, symbol: str, quantity: int) -> dict:
        result = self.place_order(symbol, quantity, "MARKET", "SELL")
        if result:
            return {"success": True, "order": result}
        return {"success": False, "error": "Order placement failed"}


def create_groww_client(mode: str = "paper", initial_balance: float = 100000.0):
    """
    Factory function to create the appropriate Groww client.

    Args:
        mode: 'paper' for simulation, 'live' for real trading
        initial_balance: Starting balance for paper trading
    """
    if mode.lower() == "live":
        token = os.getenv("GROWW_AUTH_TOKEN")
        if not token:
            raise ValueError(
                "GROWW_AUTH_TOKEN environment variable required for live trading. "
                "Get it from the Groww app's developer settings or network requests."
            )
        logger.warning("=" * 60)
        logger.warning("  LIVE TRADING MODE — REAL MONEY WILL BE USED")
        logger.warning("  Ensure you have tested thoroughly in PAPER mode first")
        logger.warning("=" * 60)
        return GrowwLiveClient(token)
    else:
        return GrowwPaperClient(initial_balance)
