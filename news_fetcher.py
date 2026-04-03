"""
Financial News Fetcher
======================
Fetches stock market news from multiple free sources:
- Economic Times Markets RSS
- Moneycontrol RSS
- Business Standard RSS
- NSE announcements
- Alpha Vantage News Sentiment API (optional, needs API key)
"""

import os
import logging
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional
import requests
import feedparser
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Free RSS feeds for Indian financial news
RSS_FEEDS = {
    "economic_times_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "economic_times_stocks": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "moneycontrol_markets": "https://www.moneycontrol.com/rss/latestnews.xml",
    "business_standard": "https://www.business-standard.com/rss/markets-106.rss",
    "livemint_market": "https://www.livemint.com/rss/markets",
    "ndtv_profit": "https://feeds.feedburner.com/ndtvprofit-latest",
    "financial_express": "https://www.financialexpress.com/market/feed/",
}

# Nifty 50 stocks and their common aliases in news
NIFTY50_STOCKS = {
    "RELIANCE": ["Reliance", "RIL", "Reliance Industries"],
    "TCS": ["TCS", "Tata Consultancy", "Tata Consultancy Services"],
    "HDFCBANK": ["HDFC Bank", "HDFCBANK"],
    "INFY": ["Infosys", "Infy"],
    "ICICIBANK": ["ICICI Bank", "ICICIBANK"],
    "HINDUNILVR": ["Hindustan Unilever", "HUL", "FMCG"],
    "ITC": ["ITC"],
    "SBIN": ["SBI", "State Bank", "State Bank of India"],
    "BHARTIARTL": ["Bharti Airtel", "Airtel"],
    "KOTAKBANK": ["Kotak Mahindra", "Kotak Bank"],
    "LT": ["Larsen & Toubro", "L&T", "L and T"],
    "HCLTECH": ["HCL Technologies", "HCL Tech"],
    "AXISBANK": ["Axis Bank"],
    "ASIANPAINT": ["Asian Paints"],
    "MARUTI": ["Maruti Suzuki", "Maruti"],
    "SUNPHARMA": ["Sun Pharma", "Sun Pharmaceutical"],
    "TITAN": ["Titan Company", "Titan"],
    "ULTRACEMCO": ["UltraTech Cement", "UltraTech"],
    "WIPRO": ["Wipro"],
    "ONGC": ["ONGC", "Oil and Natural Gas"],
    "BAJFINANCE": ["Bajaj Finance"],
    "BAJAJFINSV": ["Bajaj Finserv"],
    "NTPC": ["NTPC"],
    "POWERGRID": ["Power Grid", "PowerGrid"],
    "M&M": ["Mahindra & Mahindra", "M&M", "Mahindra"],
    "TATASTEEL": ["Tata Steel"],
    "TECHM": ["Tech Mahindra", "TechM"],
    "ADANIENT": ["Adani Enterprises", "Adani"],
    "ADANIPORTS": ["Adani Ports"],
    "JSWSTEEL": ["JSW Steel"],
    "COALINDIA": ["Coal India"],
    "DRREDDY": ["Dr Reddy's", "Dr. Reddy"],
    "CIPLA": ["Cipla"],
    "BRITANNIA": ["Britannia"],
    "DIVISLAB": ["Divi's Laboratories", "Divi Labs"],
    "GRASIM": ["Grasim"],
    "HEROMOTOCO": ["Hero MotoCorp", "Hero Moto"],
    "HINDALCO": ["Hindalco"],
    "INDUSINDBK": ["IndusInd Bank"],
    "NESTLEIND": ["Nestle India"],
    "BPCL": ["BPCL", "Bharat Petroleum"],
    "EICHERMOT": ["Eicher Motors", "Royal Enfield"],
    "APOLLOHOSP": ["Apollo Hospitals"],
    "HDFCLIFE": ["HDFC Life"],
    "SBILIFE": ["SBI Life"],
    "BAJAJ-AUTO": ["Bajaj Auto"],
    "TATAMOTORS": ["Tata Motors"],
    "UPL": ["UPL"],
    "TATACONSUM": ["Tata Consumer"],
}


def _seen_articles_cache() -> set:
    """In-memory cache to deduplicate articles within a session."""
    if not hasattr(_seen_articles_cache, "_cache"):
        _seen_articles_cache._cache = set()
    return _seen_articles_cache._cache


def _article_hash(title: str, link: str) -> str:
    return hashlib.md5(f"{title}{link}".encode()).hexdigest()


def fetch_rss_news(feed_name: str, feed_url: str, max_hours: int = 24) -> list[dict]:
    """Fetch and parse a single RSS feed."""
    articles = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_hours)
    seen = _seen_articles_cache()

    try:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries:
            # Parse publish time
            pub_time = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                pub_time = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                pub_time = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)

            if pub_time and pub_time < cutoff:
                continue

            title = getattr(entry, "title", "")
            link = getattr(entry, "link", "")
            art_hash = _article_hash(title, link)
            if art_hash in seen:
                continue
            seen.add(art_hash)

            summary = getattr(entry, "summary", "")
            # Strip HTML tags from summary
            if summary:
                soup = BeautifulSoup(summary, "lxml")
                summary = soup.get_text(separator=" ", strip=True)[:500]

            article = {
                "source": feed_name,
                "title": title,
                "summary": summary,
                "link": link,
                "published": pub_time.isoformat() if pub_time else None,
                "related_stocks": _extract_stock_mentions(title + " " + summary),
            }
            articles.append(article)

    except Exception as e:
        logger.warning(f"Failed to fetch {feed_name}: {e}")

    return articles


def _extract_stock_mentions(text: str) -> list[str]:
    """Find NSE stock symbols mentioned in text."""
    mentioned = []
    text_lower = text.lower()
    for symbol, aliases in NIFTY50_STOCKS.items():
        for alias in aliases:
            if alias.lower() in text_lower:
                mentioned.append(symbol)
                break
    return mentioned


def fetch_all_news(max_hours: int = 12, max_articles: int = 50) -> list[dict]:
    """
    Fetch financial news from all configured RSS sources.

    Args:
        max_hours: Only include articles from the last N hours
        max_articles: Maximum total articles to return

    Returns:
        List of articles sorted by publish time (newest first)
    """
    all_articles = []
    for feed_name, feed_url in RSS_FEEDS.items():
        articles = fetch_rss_news(feed_name, feed_url, max_hours)
        all_articles.extend(articles)
        logger.debug(f"Fetched {len(articles)} articles from {feed_name}")

    # Sort newest first
    def sort_key(a):
        return a.get("published") or "1970-01-01T00:00:00+00:00"

    all_articles.sort(key=sort_key, reverse=True)
    result = all_articles[:max_articles]
    logger.info(f"Total news fetched: {len(result)} articles from {len(RSS_FEEDS)} sources")
    return result


def fetch_stock_news(symbol: str, max_hours: int = 24) -> list[dict]:
    """
    Fetch news specifically about a given NSE stock symbol.

    Args:
        symbol: NSE stock symbol (e.g., 'RELIANCE')
        max_hours: Look back window in hours

    Returns:
        List of news articles mentioning this stock
    """
    all_news = fetch_all_news(max_hours=max_hours, max_articles=200)
    return [a for a in all_news if symbol in a.get("related_stocks", [])]


def fetch_alpha_vantage_news(symbol: str, limit: int = 10) -> list[dict]:
    """
    Fetch news with sentiment from Alpha Vantage API.
    Requires ALPHA_VANTAGE_API_KEY environment variable.
    Free tier: 25 requests/day.
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        logger.debug("ALPHA_VANTAGE_API_KEY not set, skipping Alpha Vantage news")
        return []

    # Alpha Vantage uses US ticker symbols; map common Indian stocks
    us_ticker_map = {"INFY": "INFY", "WIT": "WIPRO", "HDB": "HDFCBANK", "IBN": "ICICIBANK"}
    av_symbol = us_ticker_map.get(symbol, symbol)

    try:
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=NEWS_SENTIMENT&tickers={av_symbol}&limit={limit}&apikey={api_key}"
        )
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        articles = []
        for item in data.get("feed", []):
            articles.append({
                "source": "alpha_vantage",
                "title": item.get("title", ""),
                "summary": item.get("summary", "")[:500],
                "link": item.get("url", ""),
                "published": item.get("time_published", ""),
                "overall_sentiment": item.get("overall_sentiment_label", ""),
                "sentiment_score": item.get("overall_sentiment_score", 0),
                "related_stocks": [symbol],
            })
        return articles
    except Exception as e:
        logger.warning(f"Alpha Vantage news fetch failed: {e}")
        return []


def get_market_summary() -> dict:
    """
    Get Indian market indices summary (NIFTY 50, SENSEX, NIFTY BANK).
    Uses Yahoo Finance API.
    """
    indices = {
        "NIFTY50": "^NSEI",
        "SENSEX": "^BSESN",
        "NIFTYBANK": "^NSEBANK",
        "NIFTYMIDCAP": "NIFTY_MID_SELECT.NS",
    }
    result = {}
    for name, ticker in indices.items():
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            data = resp.json()
            meta = data["chart"]["result"][0]["meta"]
            price = meta.get("regularMarketPrice", 0)
            prev = meta.get("chartPreviousClose", price)
            change = price - prev
            change_pct = (change / prev * 100) if prev else 0
            result[name] = {
                "price": price,
                "change": change,
                "change_pct": round(change_pct, 2),
                "trend": "UP" if change >= 0 else "DOWN",
            }
        except Exception as e:
            logger.warning(f"Failed to fetch index {name}: {e}")
            result[name] = {"error": str(e)}

    return result
