# 🤖 Autonomous Stock Trading Agent — Groww + Claude AI

An autonomous agent that monitors Indian financial news and autonomously executes buy/sell orders on your Groww account using **Claude Opus 4.6** as its reasoning engine.

> ⚠️ **Disclaimer**: This project is for **educational purposes**. Stock trading involves substantial financial risk. Always test in **paper mode** first. The authors are not responsible for financial losses.

---

## Architecture

```
Financial News (RSS)
  ├─ Economic Times Markets
  ├─ Moneycontrol
  ├─ Business Standard
  ├─ Livemint
  └─ NDTV Profit
          │
          ▼
   Claude Opus 4.6   ←── Tools ──┐
   (Reasoning Engine)            │
          │                      ├─ get_latest_news
          │                      ├─ get_stock_news
          │                      ├─ get_market_summary
          │                      ├─ get_stock_quote
          │                      ├─ get_portfolio
          │                      ├─ buy_stock ──► Risk Manager ──► Groww
          │                      └─ sell_stock ─► Risk Manager ──► Groww
          │
          ▼
    Trade Summary
```

## Features

- **News-driven trading** — Reads 7+ Indian financial news RSS feeds
- **AI-powered analysis** — Claude Opus 4.6 with adaptive thinking analyzes news sentiment and makes decisions
- **Groww integration** — Paper trading (simulated) and Live trading (real Groww account)
- **Risk management** — Position limits, daily loss caps, stop-losses, trade frequency limits
- **Nifty 50 coverage** — Recognizes all 50 Nifty stocks in news text
- **Rich terminal UI** — Color-coded portfolio tables and trade logs

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set your ANTHROPIC_API_KEY
```

### 3. Run in paper trading mode (safe, no real money)

```bash
# Single analysis cycle
python trading_agent.py --once

# Continuous mode (every 30 minutes)
python trading_agent.py --interval 30

# Faster cycles for testing
python trading_agent.py --interval 5
```

### 4. Live trading (real money — test thoroughly in paper mode first!)

```bash
# Set in .env:
# TRADING_MODE=live
# GROWW_AUTH_TOKEN=<your-token>

python trading_agent.py --mode live --interval 30
```

## Getting Your Groww Auth Token

Groww does not provide an official API. To use live mode:

1. Open the Groww app on Android
2. Enable Developer Options → USB Debugging
3. Use a proxy tool (like Charles Proxy or mitmproxy) to capture network traffic
4. Log in to Groww — capture the `Authorization: Bearer <token>` header
5. Copy the JWT token to `GROWW_AUTH_TOKEN` in your `.env`

> **Security**: Never share or commit your auth token. It grants full access to your Groww account.

## Risk Management

The agent enforces these guardrails by default (configurable via `.env`):

| Rule | Default |
|------|---------|
| Max position size | 10% of portfolio |
| Daily loss limit | 3% of portfolio |
| Max open positions | 10 stocks |
| Max trades per day | 20 |
| Stop-loss | 5% below average buy price |
| Min trade value | ₹500 |
| Max trade value | ₹50,000 |

## File Structure

```
agentpro/
├── trading_agent.py    # Main agent — Claude + tool loop
├── groww_client.py     # Groww API client (paper + live)
├── news_fetcher.py     # Financial news RSS fetcher
├── risk_manager.py     # Risk management guardrails
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
└── README.md
```

## How the Agent Thinks

Each analysis cycle, Claude:

1. **Checks market indices** — Is NIFTY bullish/bearish today?
2. **Checks stop-losses** — Any holdings down >5%? Sell immediately.
3. **Reads latest news** — Last 12 hours from 7 sources
4. **Identifies signals** — Strong earnings? Contract wins? Regulatory trouble?
5. **Checks portfolio** — What do I hold? How much cash?
6. **Checks risk status** — Within daily trade/loss limits?
7. **Acts decisively** — Buy on positive catalysts, sell on negative ones
8. **Reports** — Summary of reasoning and actions taken

## Example Claude Decision

> "RELIANCE appears in 3 news articles this morning — all covering their Jio-BlackRock asset management JV receiving SEBI approval. This is a significant regulatory catalyst. Checking NIFTY: up 0.8% today, bullish backdrop. RELIANCE not currently held. Risk check: within limits. Buying 50 shares of RELIANCE."
