"""
Risk Manager
============
Enforces trading guardrails to prevent catastrophic losses:
- Maximum position size per stock (% of portfolio)
- Maximum daily loss limit
- Maximum total open positions
- Stop-loss enforcement
- Trade frequency limits
"""

import logging
from datetime import datetime, date
from typing import Optional

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Enforces risk management rules before any trade is executed.

    Default conservative settings suitable for retail investors.
    """

    def __init__(
        self,
        max_position_pct: float = 10.0,       # Max % of portfolio in one stock
        max_daily_loss_pct: float = 3.0,       # Stop trading if daily loss > X%
        max_open_positions: int = 10,          # Max concurrent stock positions
        max_trades_per_day: int = 20,          # Max orders per day
        min_trade_value: float = 500.0,        # Minimum order value in INR
        max_trade_value: float = 50000.0,      # Maximum single order value in INR
        stop_loss_pct: float = 5.0,            # Auto stop-loss at X% below avg price
    ):
        self.max_position_pct = max_position_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_open_positions = max_open_positions
        self.max_trades_per_day = max_trades_per_day
        self.min_trade_value = min_trade_value
        self.max_trade_value = max_trade_value
        self.stop_loss_pct = stop_loss_pct

        self._daily_trades: list[dict] = []
        self._daily_loss: float = 0.0
        self._last_reset_date: date = date.today()

    def _reset_if_new_day(self):
        today = date.today()
        if today != self._last_reset_date:
            logger.info(f"New trading day {today} — resetting daily counters")
            self._daily_trades = []
            self._daily_loss = 0.0
            self._last_reset_date = today

    def check_buy(
        self,
        symbol: str,
        quantity: int,
        price: float,
        portfolio_value: float,
        current_holdings: dict,
    ) -> tuple[bool, str]:
        """
        Validate a BUY order against all risk rules.

        Returns:
            (allowed: bool, reason: str)
        """
        self._reset_if_new_day()
        trade_value = quantity * price

        # 1. Minimum trade size
        if trade_value < self.min_trade_value:
            return False, f"Trade value ₹{trade_value:.0f} below minimum ₹{self.min_trade_value:.0f}"

        # 2. Maximum single trade size
        if trade_value > self.max_trade_value:
            return False, f"Trade value ₹{trade_value:.0f} exceeds max ₹{self.max_trade_value:.0f}"

        # 3. Maximum position concentration
        position_pct = (trade_value / portfolio_value) * 100 if portfolio_value > 0 else 100
        existing_val = 0
        if symbol in current_holdings:
            h = current_holdings[symbol]
            existing_val = h.get("qty", 0) * price
        total_position_val = existing_val + trade_value
        total_position_pct = (total_position_val / portfolio_value) * 100 if portfolio_value > 0 else 100
        if total_position_pct > self.max_position_pct:
            return (
                False,
                f"Position in {symbol} would be {total_position_pct:.1f}% of portfolio "
                f"(max {self.max_position_pct}%)",
            )

        # 4. Maximum open positions
        positions_count = len(current_holdings)
        if symbol not in current_holdings and positions_count >= self.max_open_positions:
            return (
                False,
                f"Already at max open positions ({self.max_open_positions}). "
                "Sell something before opening a new position.",
            )

        # 5. Daily loss limit
        if self._daily_loss_pct(portfolio_value) >= self.max_daily_loss_pct:
            return (
                False,
                f"Daily loss limit reached ({self._daily_loss:.0f} INR). "
                "No new buys today.",
            )

        # 6. Daily trade limit
        if len(self._daily_trades) >= self.max_trades_per_day:
            return (
                False,
                f"Daily trade limit ({self.max_trades_per_day}) reached.",
            )

        return True, "OK"

    def check_sell(
        self,
        symbol: str,
        quantity: int,
        price: float,
        current_holdings: dict,
    ) -> tuple[bool, str]:
        """Validate a SELL order."""
        self._reset_if_new_day()

        if symbol not in current_holdings:
            return False, f"No holdings in {symbol}"

        held_qty = current_holdings[symbol].get("qty", 0)
        if quantity > held_qty:
            return False, f"Cannot sell {quantity} shares of {symbol}, only hold {held_qty}"

        if len(self._daily_trades) >= self.max_trades_per_day:
            return False, f"Daily trade limit ({self.max_trades_per_day}) reached"

        return True, "OK"

    def check_stop_loss(self, symbol: str, current_price: float, avg_buy_price: float) -> bool:
        """
        Returns True if stop-loss should be triggered.
        Stop-loss fires when current price is X% below average buy price.
        """
        if avg_buy_price <= 0:
            return False
        loss_pct = ((avg_buy_price - current_price) / avg_buy_price) * 100
        if loss_pct >= self.stop_loss_pct:
            logger.warning(
                f"[STOP-LOSS] {symbol}: down {loss_pct:.1f}% from avg "
                f"₹{avg_buy_price:.2f} → current ₹{current_price:.2f}"
            )
            return True
        return False

    def record_trade(self, trade: dict):
        """Record a completed trade for daily tracking."""
        self._reset_if_new_day()
        self._daily_trades.append({**trade, "timestamp": datetime.now().isoformat()})
        pnl = trade.get("pnl", 0)
        if pnl < 0:
            self._daily_loss += abs(pnl)

    def _daily_loss_pct(self, portfolio_value: float) -> float:
        if portfolio_value <= 0:
            return 0
        return (self._daily_loss / portfolio_value) * 100

    def get_status(self, portfolio_value: float) -> dict:
        """Get current risk management status."""
        self._reset_if_new_day()
        return {
            "daily_trades": len(self._daily_trades),
            "max_daily_trades": self.max_trades_per_day,
            "daily_loss_inr": self._daily_loss,
            "daily_loss_pct": self._daily_loss_pct(portfolio_value),
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "trading_allowed": (
                len(self._daily_trades) < self.max_trades_per_day
                and self._daily_loss_pct(portfolio_value) < self.max_daily_loss_pct
            ),
            "limits": {
                "max_position_pct": self.max_position_pct,
                "max_open_positions": self.max_open_positions,
                "min_trade_value": self.min_trade_value,
                "max_trade_value": self.max_trade_value,
                "stop_loss_pct": self.stop_loss_pct,
            },
        }
