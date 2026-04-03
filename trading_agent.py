"""
Autonomous Stock Trading Agent
================================
Uses Claude Opus 4.6 with tool use to analyze financial news and
autonomously decide whether to buy, sell, or hold stocks on Groww.

Usage:
    python trading_agent.py [--mode paper|live] [--interval 30] [--once]

    --mode      Trading mode: 'paper' (default) or 'live'
    --interval  Minutes between analysis cycles (default: 30)
    --once      Run one analysis cycle then exit

Environment variables:
    ANTHROPIC_API_KEY     Required. Your Anthropic API key.
    GROWW_AUTH_TOKEN      Required for live mode. Your Groww JWT token.
    ALPHA_VANTAGE_API_KEY Optional. For sentiment-scored news.
    TRADING_MODE          Override: 'paper' or 'live'
    INITIAL_BALANCE       Starting paper balance in INR (default: 100000)

DISCLAIMER: This is for educational purposes. Stock trading involves
substantial financial risk. Past performance does not guarantee future results.
Always conduct your own research before investing.
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from typing import Any

import anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.logging import RichHandler

from groww_client import create_groww_client
from news_fetcher import fetch_all_news, fetch_stock_news, get_market_summary
from risk_manager import RiskManager

load_dotenv()

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
logger = logging.getLogger("trading_agent")

console = Console()

# ── Anthropic Client ──────────────────────────────────────────────────────────
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool Definitions (what Claude can do)
# ═══════════════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "name": "get_latest_news",
        "description": (
            "Fetch the latest Indian stock market news from multiple financial sources "
            "(Economic Times, Moneycontrol, Business Standard, Livemint, etc.). "
            "Returns articles with titles, summaries, publish times, and which NSE stocks are mentioned."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "max_hours": {
                    "type": "integer",
                    "description": "Only fetch news from the last N hours (default: 12)",
                    "default": 12,
                },
                "max_articles": {
                    "type": "integer",
                    "description": "Maximum number of articles to return (default: 30)",
                    "default": 30,
                },
            },
        },
    },
    {
        "name": "get_stock_news",
        "description": "Fetch news articles specifically mentioning a given NSE stock symbol.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "NSE stock symbol (e.g., RELIANCE, TCS, INFY)",
                },
                "max_hours": {
                    "type": "integer",
                    "description": "Look back window in hours (default: 24)",
                    "default": 24,
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_market_summary",
        "description": (
            "Get current values and day change for major Indian market indices: "
            "NIFTY 50, SENSEX, NIFTY Bank. Shows overall market sentiment."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_stock_quote",
        "description": "Get the current live price, volume, and day change for a specific NSE stock.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "NSE stock symbol (e.g., RELIANCE, TCS, INFY)",
                }
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_portfolio",
        "description": (
            "View the current Groww portfolio: cash balance, all holdings with "
            "average buy price, current price, P&L, and overall return."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "buy_stock",
        "description": (
            "Buy shares of a stock on Groww. "
            "Risk checks are enforced automatically — the trade may be rejected if it "
            "violates position limits, daily loss limits, or insufficient funds."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "NSE stock symbol to buy (e.g., RELIANCE)",
                },
                "quantity": {
                    "type": "integer",
                    "description": "Number of shares to buy",
                    "minimum": 1,
                },
                "rationale": {
                    "type": "string",
                    "description": "Brief explanation of why this stock is being bought based on news",
                },
            },
            "required": ["symbol", "quantity", "rationale"],
        },
    },
    {
        "name": "sell_stock",
        "description": (
            "Sell shares of a stock on Groww. "
            "Risk checks are enforced. Will reject if not enough shares are held."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "NSE stock symbol to sell (e.g., RELIANCE)",
                },
                "quantity": {
                    "type": "integer",
                    "description": "Number of shares to sell",
                    "minimum": 1,
                },
                "rationale": {
                    "type": "string",
                    "description": "Brief explanation of why selling (bad news, stop-loss, take-profit, etc.)",
                },
            },
            "required": ["symbol", "quantity", "rationale"],
        },
    },
    {
        "name": "get_risk_status",
        "description": (
            "Get the current risk management status: daily trade count, "
            "daily loss so far, and whether trading is still allowed today."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "check_stop_losses",
        "description": (
            "Check all current holdings for stop-loss triggers "
            "(positions down more than the stop-loss threshold). "
            "Returns any stocks that should be sold immediately."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Trading Agent Class
# ═══════════════════════════════════════════════════════════════════════════════

class TradingAgent:
    """
    Autonomous stock trading agent powered by Claude Opus 4.6.
    Analyzes news → decides trades → executes on Groww (paper or live).
    """

    SYSTEM_PROMPT = """You are an autonomous Indian stock market trading agent operating on Groww.
Your goal is to make profitable trades by analyzing financial news and market conditions.

## Your Responsibilities
1. **Monitor news** — Fetch and analyze Indian financial news to identify trading opportunities
2. **Assess market sentiment** — Check NIFTY/SENSEX trends before any trade
3. **Make informed decisions** — Use news analysis to decide BUY, SELL, or HOLD
4. **Manage risk** — Always check risk status; never exceed position limits
5. **Check stop-losses** — Immediately sell any position triggering stop-loss

## Decision Framework
**BUY signals** (positive catalysts):
- Strong quarterly results (revenue growth >15%, profit growth >20%)
- New major contracts, partnerships, or government orders
- Product launches with strong market reception
- Sector tailwinds (policy support, commodity price drops for input-cost reduction)
- Analyst upgrades or institutional buying

**SELL signals** (negative catalysts):
- Weak quarterly results or profit warnings
- Regulatory action, fraud allegations, or management changes
- Loss of major contracts
- Sector headwinds (competition, rising input costs)
- Stop-loss triggered (>5% below average buy price)

## Trading Rules
- Only trade Nifty 50 or large-cap stocks for liquidity
- Always check get_portfolio before buying (ensure funds available)
- Check get_risk_status to confirm trading is allowed
- Maximum 2-3 new positions per analysis cycle
- Provide clear rationale for every trade decision
- When uncertain, prefer HOLD — capital preservation matters

## Important
- You are running in {mode} mode
- Every buy/sell call goes through automatic risk checks
- Be systematic: news → market check → portfolio check → risk check → decide → act"""

    def __init__(self, mode: str = "paper"):
        self.mode = mode
        initial_balance = float(os.getenv("INITIAL_BALANCE", "100000"))
        self.groww = create_groww_client(mode=mode, initial_balance=initial_balance)
        self.risk = RiskManager(
            max_position_pct=float(os.getenv("MAX_POSITION_PCT", "10")),
            max_daily_loss_pct=float(os.getenv("MAX_DAILY_LOSS_PCT", "3")),
            max_open_positions=int(os.getenv("MAX_OPEN_POSITIONS", "10")),
            max_trades_per_day=int(os.getenv("MAX_TRADES_PER_DAY", "20")),
            stop_loss_pct=float(os.getenv("STOP_LOSS_PCT", "5")),
        )
        self.trade_log: list[dict] = []
        console.print(
            Panel.fit(
                f"[bold green]Trading Agent Started[/bold green]\n"
                f"Mode: [bold {'red' if mode == 'live' else 'yellow'}]{mode.upper()}[/bold {'red' if mode == 'live' else 'yellow'}]\n"
                f"Model: claude-opus-4-6",
                title="🤖 Groww AI Trader",
                border_style="green",
            )
        )

    # ── Tool Handlers ─────────────────────────────────────────────────────────

    def _handle_tool(self, tool_name: str, tool_input: dict) -> Any:
        """Dispatch tool calls to the appropriate handler."""
        handlers = {
            "get_latest_news": self._tool_get_latest_news,
            "get_stock_news": self._tool_get_stock_news,
            "get_market_summary": self._tool_get_market_summary,
            "get_stock_quote": self._tool_get_stock_quote,
            "get_portfolio": self._tool_get_portfolio,
            "buy_stock": self._tool_buy_stock,
            "sell_stock": self._tool_sell_stock,
            "get_risk_status": self._tool_get_risk_status,
            "check_stop_losses": self._tool_check_stop_losses,
        }
        handler = handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}
        return handler(tool_input)

    def _tool_get_latest_news(self, inp: dict) -> dict:
        articles = fetch_all_news(
            max_hours=inp.get("max_hours", 12),
            max_articles=inp.get("max_articles", 30),
        )
        console.print(f"[dim]📰 Fetched {len(articles)} news articles[/dim]")
        return {"article_count": len(articles), "articles": articles}

    def _tool_get_stock_news(self, inp: dict) -> dict:
        symbol = inp["symbol"].upper()
        articles = fetch_stock_news(symbol, max_hours=inp.get("max_hours", 24))
        console.print(f"[dim]📰 Found {len(articles)} articles about {symbol}[/dim]")
        return {"symbol": symbol, "article_count": len(articles), "articles": articles}

    def _tool_get_market_summary(self, _inp: dict) -> dict:
        summary = get_market_summary()
        console.print("[dim]📊 Fetched market indices[/dim]")
        return summary

    def _tool_get_stock_quote(self, inp: dict) -> dict:
        symbol = inp["symbol"].upper()
        quote = self.groww.get_quote(symbol)
        if not quote:
            return {"error": f"Could not fetch quote for {symbol}"}
        console.print(f"[dim]💹 Quote {symbol}: ₹{quote['price']:.2f}[/dim]")
        return quote

    def _tool_get_portfolio(self, _inp: dict) -> dict:
        portfolio = self.groww.get_portfolio()
        # Display as table
        table = Table(title="Current Portfolio", box=box.SIMPLE)
        table.add_column("Symbol", style="cyan")
        table.add_column("Qty", justify="right")
        table.add_column("Avg Price", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("P&L %", justify="right")
        for h in portfolio.get("holdings", []):
            pnl_color = "green" if h["pnl"] >= 0 else "red"
            table.add_row(
                h["symbol"],
                str(h["qty"]),
                f"₹{h['avg_price']:.2f}",
                f"₹{h['current_price']:.2f}",
                f"[{pnl_color}]₹{h['pnl']:+.2f}[/{pnl_color}]",
                f"[{pnl_color}]{h['pnl_pct']:+.1f}%[/{pnl_color}]",
            )
        console.print(table)
        console.print(
            f"💰 Cash: ₹{portfolio.get('cash_balance', 0):,.2f} | "
            f"Total: ₹{portfolio.get('total_portfolio_value', 0):,.2f} | "
            f"Return: {portfolio.get('return_pct', 0):+.2f}%"
        )
        return portfolio

    def _tool_buy_stock(self, inp: dict) -> dict:
        symbol = inp["symbol"].upper()
        quantity = inp["quantity"]
        rationale = inp.get("rationale", "")
        console.print(f"[yellow]🛒 Attempting BUY: {quantity} × {symbol}[/yellow]")
        console.print(f"   [dim]Rationale: {rationale}[/dim]")

        # Get quote for risk check
        quote = self.groww.get_quote(symbol)
        if not quote:
            return {"success": False, "error": "Cannot fetch price for risk check"}
        price = quote["price"]

        # Get portfolio for risk check
        portfolio = self.groww.get_portfolio()
        portfolio_value = portfolio.get("total_portfolio_value", 0)
        holdings_dict = {h["symbol"]: h for h in portfolio.get("holdings", [])}

        # Risk check
        allowed, reason = self.risk.check_buy(
            symbol, quantity, price, portfolio_value, holdings_dict
        )
        if not allowed:
            console.print(f"[red]❌ Risk check failed: {reason}[/red]")
            return {"success": False, "risk_rejected": True, "reason": reason}

        # Execute trade
        result = self.groww.buy(symbol, quantity)
        if result.get("success"):
            trade = result.get("trade", {})
            self.risk.record_trade(trade)
            self.trade_log.append({
                "action": "BUY",
                "symbol": symbol,
                "quantity": quantity,
                "price": price,
                "rationale": rationale,
                "timestamp": datetime.now().isoformat(),
                "mode": self.mode,
            })
            console.print(
                f"[green]✅ BUY executed: {quantity} × {symbol} @ ₹{price:.2f}[/green]"
            )
        else:
            console.print(f"[red]❌ BUY failed: {result.get('error')}[/red]")
        return result

    def _tool_sell_stock(self, inp: dict) -> dict:
        symbol = inp["symbol"].upper()
        quantity = inp["quantity"]
        rationale = inp.get("rationale", "")
        console.print(f"[yellow]💰 Attempting SELL: {quantity} × {symbol}[/yellow]")
        console.print(f"   [dim]Rationale: {rationale}[/dim]")

        # Get portfolio for risk check
        portfolio = self.groww.get_portfolio()
        holdings_dict = {h["symbol"]: h for h in portfolio.get("holdings", [])}

        # Risk check
        allowed, reason = self.risk.check_sell(symbol, quantity, 0, holdings_dict)
        if not allowed:
            console.print(f"[red]❌ Risk check failed: {reason}[/red]")
            return {"success": False, "risk_rejected": True, "reason": reason}

        # Execute trade
        result = self.groww.sell(symbol, quantity)
        if result.get("success"):
            trade = result.get("trade", {})
            self.risk.record_trade(trade)
            pnl = result.get("pnl", 0)
            pnl_color = "green" if pnl >= 0 else "red"
            self.trade_log.append({
                "action": "SELL",
                "symbol": symbol,
                "quantity": quantity,
                "pnl": pnl,
                "rationale": rationale,
                "timestamp": datetime.now().isoformat(),
                "mode": self.mode,
            })
            console.print(
                f"[{pnl_color}]✅ SELL executed: {quantity} × {symbol} | "
                f"P&L: ₹{pnl:+.2f}[/{pnl_color}]"
            )
        else:
            console.print(f"[red]❌ SELL failed: {result.get('error')}[/red]")
        return result

    def _tool_get_risk_status(self, _inp: dict) -> dict:
        portfolio = self.groww.get_portfolio()
        portfolio_value = portfolio.get("total_portfolio_value", 0)
        status = self.risk.get_status(portfolio_value)
        console.print(
            f"[dim]⚡ Risk: {status['daily_trades']}/{status['max_daily_trades']} trades today | "
            f"Daily loss: ₹{status['daily_loss_inr']:.0f} ({status['daily_loss_pct']:.1f}%)[/dim]"
        )
        return status

    def _tool_check_stop_losses(self, _inp: dict) -> dict:
        portfolio = self.groww.get_portfolio()
        triggered = []
        for holding in portfolio.get("holdings", []):
            symbol = holding["symbol"]
            current_price = holding["current_price"]
            avg_price = holding["avg_price"]
            if self.risk.check_stop_loss(symbol, current_price, avg_price):
                triggered.append({
                    "symbol": symbol,
                    "qty": holding["qty"],
                    "avg_price": avg_price,
                    "current_price": current_price,
                    "loss_pct": holding["pnl_pct"],
                })
        if triggered:
            console.print(f"[red]🚨 Stop-loss triggered for: {[t['symbol'] for t in triggered]}[/red]")
        else:
            console.print("[green]✅ No stop-losses triggered[/green]")
        return {"triggered": triggered, "count": len(triggered)}

    # ── Agent Loop ────────────────────────────────────────────────────────────

    def run_analysis_cycle(self) -> str:
        """
        Run one full analysis cycle:
        1. Claude fetches news and market data via tools
        2. Analyzes sentiment and opportunities
        3. Makes buy/sell decisions
        4. Returns a summary of actions taken
        """
        console.print(
            Panel(
                f"[bold]Starting analysis cycle[/bold]\n"
                f"[dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}[/dim]",
                border_style="blue",
            )
        )

        messages = [
            {
                "role": "user",
                "content": (
                    "Run a complete trading analysis cycle:\n"
                    "1. Check market indices for overall sentiment\n"
                    "2. Check stop-losses on current holdings\n"
                    "3. Fetch latest financial news (last 12 hours)\n"
                    "4. Analyze news for trading opportunities\n"
                    "5. Check current portfolio and risk status\n"
                    "6. Execute appropriate BUY/SELL orders based on your analysis\n"
                    "7. Provide a concise summary of what you did and why\n\n"
                    "Be decisive but careful. Only act on strong, clear signals."
                ),
            }
        ]

        system = self.SYSTEM_PROMPT.format(mode=self.mode.upper())

        # Agentic tool-use loop
        while True:
            response = anthropic_client.messages.create(
                model="claude-opus-4-6",
                max_tokens=8096,
                thinking={"type": "adaptive"},
                system=system,
                tools=TOOLS,
                messages=messages,
            )

            # Append assistant response to conversation
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                # Extract final text response
                final_text = next(
                    (b.text for b in response.content if b.type == "text"), "Analysis complete."
                )
                console.print(Panel(final_text, title="📋 Agent Summary", border_style="green"))
                return final_text

            if response.stop_reason != "tool_use":
                logger.warning(f"Unexpected stop reason: {response.stop_reason}")
                break

            # Execute all tool calls
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_name = block.name
                tool_input = block.input
                console.print(f"[dim]🔧 Tool: [bold]{tool_name}[/bold][/dim]")

                try:
                    result = self._handle_tool(tool_name, tool_input)
                except Exception as e:
                    logger.error(f"Tool {tool_name} failed: {e}")
                    result = {"error": str(e)}

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=str),
                })

            messages.append({"role": "user", "content": tool_results})

        return "Analysis cycle complete"

    def print_trade_log(self):
        """Print all trades executed this session."""
        if not self.trade_log:
            console.print("[dim]No trades executed this session[/dim]")
            return
        table = Table(title="Trade Log (This Session)", box=box.SIMPLE)
        table.add_column("Time", style="dim")
        table.add_column("Action", style="bold")
        table.add_column("Symbol", style="cyan")
        table.add_column("Qty", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("Rationale")
        for t in self.trade_log:
            action_color = "green" if t["action"] == "BUY" else "yellow"
            pnl = t.get("pnl")
            pnl_str = f"₹{pnl:+.2f}" if pnl is not None else "—"
            table.add_row(
                t["timestamp"][11:19],
                f"[{action_color}]{t['action']}[/{action_color}]",
                t["symbol"],
                str(t["quantity"]),
                pnl_str,
                t.get("rationale", "")[:60],
            )
        console.print(table)


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Autonomous Groww Stock Trading Agent")
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default=os.getenv("TRADING_MODE", "paper"),
        help="Trading mode: paper (simulated) or live (real money). Default: paper",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Minutes between analysis cycles (default: 30)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single analysis cycle and exit",
    )
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        console.print("[bold red]Error: ANTHROPIC_API_KEY environment variable not set[/bold red]")
        sys.exit(1)

    if args.mode == "live" and not os.getenv("GROWW_AUTH_TOKEN"):
        console.print(
            "[bold red]Error: GROWW_AUTH_TOKEN required for live mode.[/bold red]\n"
            "Get your token from the Groww mobile app (network requests) "
            "or set TRADING_MODE=paper in .env"
        )
        sys.exit(1)

    agent = TradingAgent(mode=args.mode)

    if args.once:
        agent.run_analysis_cycle()
        agent.print_trade_log()
        return

    console.print(
        f"\n[bold]Starting continuous trading loop[/bold] — "
        f"analysis every [cyan]{args.interval}[/cyan] minutes\n"
        f"Press [bold]Ctrl+C[/bold] to stop\n"
    )

    cycle_count = 0
    try:
        while True:
            cycle_count += 1
            console.print(f"\n[bold blue]═══ Cycle #{cycle_count} ═══[/bold blue]")
            try:
                agent.run_analysis_cycle()
            except anthropic.RateLimitError:
                logger.warning("Rate limited by Anthropic API — waiting 60s")
                time.sleep(60)
            except anthropic.APIError as e:
                logger.error(f"Anthropic API error: {e}")
            except Exception as e:
                logger.error(f"Unexpected error in analysis cycle: {e}", exc_info=True)

            if not args.once:
                console.print(
                    f"\n[dim]Next cycle in {args.interval} minutes "
                    f"({datetime.now().strftime('%H:%M:%S')} → "
                    f"approx {args.interval}min)[/dim]"
                )
                time.sleep(args.interval * 60)

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Shutting down trading agent...[/bold yellow]")
        agent.print_trade_log()
        console.print("[bold green]Goodbye![/bold green]")


if __name__ == "__main__":
    main()
