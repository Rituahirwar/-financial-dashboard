#!/usr/bin/env python3


import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
import pandas as pd
import numpy as np
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.responses import HTMLResponse
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
import yfinance as yf
import uvicorn
import os
from pathlib import Path
import nest_asyncio

# =============================================================================
# ENHANCED DATABASE MODELS
# =============================================================================

Base = declarative_base()

class Portfolio(Base):
    __tablename__ = "portfolios"
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    holdings = relationship("Holding", back_populates="portfolio", cascade="all, delete-orphan")
    transactions = relationship("Transaction", back_populates="portfolio", cascade="all, delete-orphan")

class Holding(Base):
    __tablename__ = "holdings"
    
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"))
    symbol = Column(String, nullable=False)
    company_name = Column(String)
    shares = Column(Float, nullable=False)
    avg_cost = Column(Float, nullable=False)
    sector = Column(String)
    created_at = Column(DateTime, default=datetime.now)
    
    portfolio = relationship("Portfolio", back_populates="holdings")

class Transaction(Base):
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"))
    symbol = Column(String, nullable=False)
    transaction_type = Column(String, nullable=False)
    shares = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    fees = Column(Float, default=0.0)
    date = Column(DateTime, default=datetime.now)
    notes = Column(Text)
    
    portfolio = relationship("Portfolio", back_populates="transactions")

class MarketData(Base):
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    price = Column(Float, nullable=False)
    change = Column(Float)
    change_percent = Column(Float)
    volume = Column(Float)
    market_cap = Column(Float)
    pe_ratio = Column(Float)
    timestamp = Column(DateTime, default=datetime.now)

# =============================================================================
# ENHANCED PYDANTIC MODELS
# =============================================================================

class HoldingResponse(BaseModel):
    id: int
    symbol: str
    company_name: str
    sector: str
    shares: float
    avg_cost: float
    current_price: float
    market_value: float
    gain_loss: float
    gain_loss_percent: float
    weight: float
    day_change: float
    day_change_percent: float

class PortfolioSummary(BaseModel):
    total_value: float
    total_cost: float
    total_gain_loss: float
    total_gain_loss_percent: float
    day_change: float
    day_change_percent: float
    cash_balance: float
    holdings_count: int
    holdings: List[HoldingResponse]

class TransactionRequest(BaseModel):
    symbol: str
    transaction_type: str  # buy/sell
    shares: float
    price: Optional[float] = None  # If None, use current market price
    fees: float = 0.0
    notes: Optional[str] = None

class MarketDataResponse(BaseModel):
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: float
    market_cap: Optional[float]
    pe_ratio: Optional[float]

class MonteCarloProjection(BaseModel):
    years: int
    simulations: int
    results: Dict[str, List[float]]

class HistoricalPerformance(BaseModel):
    dates: List[str]
    values: List[float]

class NewsArticle(BaseModel):
    title: str
    publisher: str
    link: str

# =============================================================================
# REAL MARKET DATA SERVICE
# =============================================================================

class RealMarketDataProvider:
    """Real market data using Yahoo Finance"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 60  # 1 minute cache
    
    async def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive stock information"""
        try:
            ticker = yf.Ticker(symbol)
            # yfinance calls are not natively async, run them in a thread pool
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(None, lambda: ticker.info) # Fetch info for company name, sector, etc.
            hist = await loop.run_in_executor(None, lambda: ticker.history(period="1y")) # Get enough history for stats
            
            if hist.empty or 'Close' not in hist.columns:
                return self._get_fallback_data(symbol)
            
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else info.get('previousClose', current_price)
            
            return {
                'symbol': symbol,
                'company_name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'current_price': current_price,
                'change': current_price - prev_close,
                'change_percent': ((current_price - prev_close) / prev_close * 100) if prev_close > 0 else 0,
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                # Calculate daily returns, handling potential NaNs
                'daily_return_std': hist['Close'].pct_change().dropna().std() if not hist['Close'].pct_change().dropna().empty else 0.015, # Default volatility
                'avg_daily_return': hist['Close'].pct_change().dropna().mean() if not hist['Close'].pct_change().dropna().empty else 0.0005, # Default average return
            }
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return self._get_fallback_data(symbol)
    
    def _get_fallback_data(self, symbol: str) -> Dict[str, Any]:
        """Fallback data if API fails"""
        base_prices = {
            'AAPL': 175.50, 'GOOGL': 2750.25, 'MSFT': 350.75, 'AMZN': 3200.80,
            'TSLA': 850.40, 'NVDA': 550.30, 'META': 425.60, 'NFLX': 450.20,
            'SPY': 420.15, 'QQQ': 380.45
        }
        
        base_price = base_prices.get(symbol, 100.0)
        change = np.random.uniform(-5, 5)
        
        return {
            'symbol': symbol,
            'company_name': f"{symbol} Inc.",
            'sector': 'Technology',
            'current_price': base_price + change,
            'change': change,
            'change_percent': (change / base_price) * 100,
            'volume': np.random.randint(1000000, 50000000),
            'market_cap': base_price * 1000000000,
            'pe_ratio': np.random.uniform(15, 35),
            'daily_return_std': 0.015, # Consistent default
            'avg_daily_return': 0.0005, # Consistent default
        }
    
    async def get_multiple_stocks(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get data for multiple stocks"""
        tasks = [self.get_stock_info(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        return {res['symbol']: res for res in results}
        
# =============================================================================
# ENHANCED PORTFOLIO ANALYZER
# =============================================================================

class AdvancedPortfolioAnalyzer:
    """Advanced portfolio analysis with real market data"""
    
    def __init__(self, market_provider: RealMarketDataProvider):
        self.market_provider = market_provider
    
    async def calculate_portfolio_summary(self, holdings: List[Holding], cash_balance: float = 0.0) -> PortfolioSummary:
        """Calculate comprehensive portfolio summary"""
        if not holdings:
            return PortfolioSummary(
                total_value=cash_balance, total_cost=0, total_gain_loss=0,
                total_gain_loss_percent=0, day_change=0, day_change_percent=0,
                cash_balance=cash_balance, holdings_count=0, holdings=[]
            )
        
        symbols = [h.symbol for h in holdings]
        market_data = await self.market_provider.get_multiple_stocks(symbols)
        
        holding_responses = []
        total_value = cash_balance
        total_cost = 0
        total_day_change = 0
        
        for holding in holdings:
            stock_data = market_data.get(holding.symbol, {})
            current_price = stock_data.get('current_price', holding.avg_cost)
            
            market_value = holding.shares * current_price
            cost_basis = holding.shares * holding.avg_cost
            gain_loss = market_value - cost_basis
            gain_loss_percent = (gain_loss / cost_basis * 100) if cost_basis > 0 else 0
            
            day_change_per_share = stock_data.get('change', 0)
            day_change = holding.shares * day_change_per_share
            day_change_percent = stock_data.get('change_percent', 0)
            
            holding_responses.append(HoldingResponse(
                id=holding.id,
                symbol=holding.symbol,
                company_name=stock_data.get('company_name', holding.company_name or holding.symbol),
                sector=stock_data.get('sector', holding.sector or 'Unknown'),
                shares=holding.shares,
                avg_cost=holding.avg_cost,
                current_price=current_price,
                market_value=market_value,
                gain_loss=gain_loss,
                gain_loss_percent=gain_loss_percent,
                weight=0,  # Will be calculated after total_value
                day_change=day_change,
                day_change_percent=day_change_percent
            ))
            
            total_value += market_value
            total_cost += cost_basis
            total_day_change += day_change
        
        # Calculate weights
        for holding_response in holding_responses:
            holding_response.weight = (holding_response.market_value / total_value * 100) if total_value > 0 else 0
        
        total_gain_loss = total_value - total_cost - cash_balance
        total_gain_loss_percent = (total_gain_loss / total_cost * 100) if total_cost > 0 else 0
        total_day_change_percent = (total_day_change / (total_value - total_day_change) * 100) if total_value > total_day_change else 0
        
        return PortfolioSummary(
            total_value=total_value,
            total_cost=total_cost + cash_balance,
            total_gain_loss=total_gain_loss,
            total_gain_loss_percent=total_gain_loss_percent,
            day_change=total_day_change,
            day_change_percent=total_day_change_percent,
            cash_balance=cash_balance,
            holdings_count=len(holdings),
            holdings=holding_responses
        )
        
    async def run_monte_carlo_simulation(self, holdings: List[Holding], years: int = 10, simulations: int = 500) -> MonteCarloProjection:
        """Run Monte Carlo simulation for portfolio projection"""
        if not holdings:
            raise ValueError("Cannot run simulation on an empty portfolio.")

        symbols = [h.symbol for h in holdings]
        market_data = await self.market_provider.get_multiple_stocks(symbols)

        # Calculate portfolio-level metrics
        valid_holdings_data = []
        for h in holdings:
            if h.symbol in market_data and 'current_price' in market_data[h.symbol]:
                valid_holdings_data.append({
                    'symbol': h.symbol, 
                    'value': h.shares * market_data[h.symbol]['current_price']
                })
        
        if not valid_holdings_data:
            raise ValueError("Could not retrieve market data for any holdings.")

        portfolio_df = pd.DataFrame(valid_holdings_data)
        portfolio_df['weight'] = portfolio_df['value'] / portfolio_df['value'].sum()

        # Create a DataFrame from the market data for easier merging
        stats_df = pd.DataFrame.from_dict(market_data, orient='index')[['avg_daily_return', 'daily_return_std']]
        
        # Merge weights and stats, ensuring alignment
        merged_df = portfolio_df.set_index('symbol').join(stats_df)

        # Calculate weighted average return and standard deviation
        avg_return = (merged_df['weight'] * merged_df['avg_daily_return']).sum()
        std_dev = (merged_df['weight'] * merged_df['daily_return_std']).sum()

        initial_value = portfolio_df['value'].sum()
        trading_days = 252 * years

        # Run simulation
        simulation_results = np.zeros((trading_days, simulations))
        if initial_value <= 0:
            # If initial value is zero or negative, simulation is not meaningful.
            raise ValueError("Initial portfolio value must be positive to run simulation.")
        for i in range(simulations):
            daily_returns = np.random.normal(avg_return, std_dev, trading_days)
            price_path = np.cumprod(1 + daily_returns) * initial_value
            simulation_results[:, i] = price_path

        results_df = pd.DataFrame(simulation_results)
        # Pydantic requires string keys for the dictionary, so convert integer columns to strings.
        results_df.columns = results_df.columns.map(str)
        return MonteCarloProjection(
            years=years,
            simulations=simulations,
            results=results_df.to_dict(orient='list')
        )
        
    async def get_historical_performance(self, holdings: List[Holding], period: str = "1y") -> HistoricalPerformance:
        """Calculate historical portfolio value over a given period."""
        if not holdings:
            return HistoricalPerformance(dates=[], values=[])

        symbols = [h.symbol for h in holdings]
        
        # Fetch historical data for all symbols at once
        try:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, lambda: yf.download(symbols, period=period, progress=False))
            if data.empty:
                return HistoricalPerformance(dates=[], values=[])
            
            # Handle both single and multi-level column structures from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                close_prices = data['Close']
            else:
                # If only one stock, columns are not multi-level. Reformat for consistency.
                close_prices = data[['Close']].rename(columns={'Close': symbols[0]})
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return HistoricalPerformance(dates=[], values=[])

        # Calculate the total value for each day
        portfolio_value = pd.Series(0.0, index=close_prices.index)
        for holding in holdings:
            if holding.symbol in close_prices.columns:
                portfolio_value += close_prices[holding.symbol].fillna(method='ffill') * holding.shares

        return HistoricalPerformance(dates=portfolio_value.index.strftime('%Y-%m-%d').tolist(), values=portfolio_value.tolist())

# =============================================================================
# DATABASE SETUP
# =============================================================================

# Use an environment variable for the database URL in production,
# with a fallback to a local file for development.
DATA_DIR = os.environ.get("RENDER_DATA_DIR", ".")
DATABASE_URL = f"sqlite:///{os.path.join(DATA_DIR, 'premium_financial_dashboard.db')}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =============================================================================
# ENHANCED FASTAPI BACKEND
# =============================================================================

app = FastAPI(
    title="Premium Financial Dashboard API",
    description="Professional-grade portfolio management system",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
market_provider = RealMarketDataProvider()
portfolio_analyzer = AdvancedPortfolioAnalyzer(market_provider)

def initialize_premium_data():
    """Initialize with real market data from major companies"""
    db = SessionLocal()
    
    try:
        existing_portfolio = db.query(Portfolio).first()
        if existing_portfolio:
            return
        
        # Create premium portfolio with real companies
        portfolio = Portfolio(
            name="Premium Growth Portfolio",
            description="Diversified technology and growth stock portfolio"
        )
        db.add(portfolio)
        db.commit()
        db.refresh(portfolio)
        
        # Real companies with actual data
        premium_holdings = [
            ("AAPL", "Apple Inc.", 25, 145.50, "Technology"),
            ("GOOGL", "Alphabet Inc.", 15, 2650.00, "Technology"),
            ("MSFT", "Microsoft Corporation", 30, 335.00, "Technology"),
            ("AMZN", "Amazon.com Inc.", 12, 3150.00, "Consumer Discretionary"),
            ("TSLA", "Tesla Inc.", 8, 800.00, "Consumer Discretionary"),
            ("NVDA", "NVIDIA Corporation", 20, 520.00, "Technology"),
            ("META", "Meta Platforms Inc.", 18, 420.00, "Technology"),
            ("NFLX", "Netflix Inc.", 10, 445.00, "Communication Services"),
        ]
        
        for symbol, company_name, shares, avg_cost, sector in premium_holdings:
            holding = Holding(
                portfolio_id=portfolio.id,
                symbol=symbol,
                company_name=company_name,
                shares=shares,
                avg_cost=avg_cost,
                sector=sector
            )
            db.add(holding)
            
            # Add sample transaction
            transaction = Transaction(
                portfolio_id=portfolio.id,
                symbol=symbol,
                transaction_type="buy",
                shares=shares,
                price=avg_cost,
                date=datetime.now() - timedelta(days=np.random.randint(1, 365)),
                notes=f"Initial purchase of {company_name}"
            )
            db.add(transaction)
        
        db.commit()
        print("✅ Premium portfolio data initialized with real companies")
        
    except Exception as e:
        print(f"Error initializing premium data: {e}")
        db.rollback()
    finally:
        db.close()

@app.get("/api/portfolio/summary", response_model=PortfolioSummary)
async def get_portfolio_summary(db: Session = Depends(get_db)):
    """Get comprehensive portfolio summary"""
    portfolio = db.query(Portfolio).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    holdings = db.query(Holding).filter(Holding.portfolio_id == portfolio.id).all()
    return await portfolio_analyzer.calculate_portfolio_summary(holdings, cash_balance=5000.0)

@app.get("/api/market-data/{symbol}", response_model=MarketDataResponse)
async def get_market_data(symbol: str):
    """Get real-time market data for a stock"""
    data = market_provider.get_stock_info(symbol)
    return MarketDataResponse(
        symbol=data['symbol'],
        price=data['current_price'],
        change=data['change'],
        change_percent=data['change_percent'],
        volume=data['volume'],
        market_cap=data['market_cap'],
        pe_ratio=data['pe_ratio']
    )

@app.post("/api/transactions")
async def add_transaction(transaction: TransactionRequest, db: Session = Depends(get_db)):
    """Add a new transaction with real market data"""
    portfolio = db.query(Portfolio).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    # Get current market price if not provided
    if transaction.price is None:
        market_data = await market_provider.get_stock_info(transaction.symbol)
        transaction.price = market_data['current_price']
    
    # Create transaction
    db_transaction = Transaction(
        portfolio_id=portfolio.id,
        symbol=transaction.symbol,
        transaction_type=transaction.transaction_type,
        shares=transaction.shares,
        price=transaction.price,
        fees=transaction.fees,
        notes=transaction.notes
    )
    db.add(db_transaction)
    
    # Update or create holding
    existing_holding = db.query(Holding).filter(
        Holding.portfolio_id == portfolio.id,
        Holding.symbol == transaction.symbol
    ).first()
    
    market_data = await market_provider.get_stock_info(transaction.symbol)
    
    if transaction.transaction_type == "buy":
        if existing_holding:
            total_cost = (existing_holding.shares * existing_holding.avg_cost) + \
                        (transaction.shares * transaction.price)
            total_shares = existing_holding.shares + transaction.shares
            existing_holding.avg_cost = total_cost / total_shares
            existing_holding.shares = total_shares
        else:
            new_holding = Holding(
                portfolio_id=portfolio.id,
                symbol=transaction.symbol,
                company_name=market_data['company_name'],
                shares=transaction.shares,
                avg_cost=transaction.price,
                sector=market_data['sector']
            )
            db.add(new_holding)
    
    elif transaction.transaction_type == "sell":
        if existing_holding and existing_holding.shares >= transaction.shares:
            existing_holding.shares -= transaction.shares
            if existing_holding.shares == 0:
                db.delete(existing_holding)
        else:
            raise HTTPException(status_code=400, detail="Insufficient shares to sell")
    
    db.commit()
    return {"message": "Transaction processed successfully", "current_price": transaction.price}

from fastapi import Query

@app.post("/api/portfolio/project", response_model=MonteCarloProjection)
async def project_portfolio_value(
    years: int = Query(10, ge=1, le=50),
    simulations: int = Query(500, ge=100, le=5000),
    db: Session = Depends(get_db)
):
    """Run Monte Carlo simulation to project future portfolio value"""
    portfolio = db.query(Portfolio).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    holdings = db.query(Holding).filter(Holding.portfolio_id == portfolio.id).all()
    return await portfolio_analyzer.run_monte_carlo_simulation(holdings, years=years, simulations=simulations)

@app.get("/api/portfolio/history", response_model=HistoricalPerformance)
async def get_portfolio_history(period: str = "1y", db: Session = Depends(get_db)):
    """Get historical portfolio value over a specified period."""
    portfolio = db.query(Portfolio).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    holdings = db.query(Holding).filter(Holding.portfolio_id == portfolio.id).all()
    return await portfolio_analyzer.get_historical_performance(holdings, period=period)

@app.get("/api/portfolio/news", response_model=List[NewsArticle])
async def get_portfolio_news(db: Session = Depends(get_db)):
    """Get recent news for stocks in the portfolio."""
    portfolio = db.query(Portfolio).first()
    if not portfolio:
        return []
    
    symbols = [h.symbol for h in db.query(Holding.symbol).filter(Holding.portfolio_id == portfolio.id).all()]
    if not symbols:
        return []

    # yfinance news can be slow; run it in an executor to avoid blocking
    loop = asyncio.get_event_loop()
    news = []
    async def fetch_news(symbol):
        ticker = yf.Ticker(symbol)
        return await loop.run_in_executor(None, getattr, ticker, 'news')

    news_results = await asyncio.gather(*(fetch_news(s) for s in symbols[:5]))
    for news_list in news_results:
        news.extend(news_list)
    
    # Process the raw news data to fit the NewsArticle model
    processed_news = []
    for article_data in news:
        # The actual data is deeply nested and keys can be missing.
        content = article_data.get('content', {})
        if not content:
            continue

        title = content.get('title')
        
        provider = content.get('provider', {})
        publisher = provider.get('displayName')

        # The link can be in one of two places
        link = content.get('link')

        # Use pubDate for sorting, which seems to be a standard ISO format string
        publish_time_str = content.get('pubDate')
        
        if all([title, publisher, link, publish_time_str]):
            try:
                # Convert to a comparable timestamp for sorting
                publish_time = datetime.fromisoformat(publish_time_str.replace('Z', '+00:00')).timestamp()
                processed_news.append({
                    'title': title,
                    'publisher': publisher,
                    'link': link,
                    'publish_time': publish_time
                })
            except (ValueError, TypeError):
                continue # Ignore articles with malformed dates

    # Sort by the numeric timestamp
    sorted_news = sorted(processed_news, key=lambda x: x['publish_time'], reverse=True)

    # Create the final list of Pydantic models for the response, which FastAPI will validate.
    return [
        NewsArticle(title=a['title'], publisher=a['publisher'], link=a['link'])
        for a in sorted_news[:10]
    ]

@app.get("/api/transactions")
async def get_transactions(limit: int = 50, db: Session = Depends(get_db)):
    """Get transaction history"""
    portfolio = db.query(Portfolio).first()
    if not portfolio:
        return []
    
    transactions = db.query(Transaction).filter(
        Transaction.portfolio_id == portfolio.id
    ).order_by(Transaction.date.desc()).limit(limit).all()
    
    return [
        {
            "id": t.id,
            "symbol": t.symbol,
            "type": t.transaction_type,
            "shares": t.shares,
            "price": t.price,
            "fees": t.fees,
            "total": t.shares * t.price + t.fees,
            "date": t.date.isoformat(),
            "notes": t.notes
        }
        for t in transactions
    ]

# =============================================================================
# PREMIUM DASH FRONTEND
# =============================================================================

def create_navbar():
    """Creates the navigation bar for the dashboard."""
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Overview", href="/")),
            dbc.NavItem(dbc.NavLink("Holdings", href="/holdings")),
            dbc.NavItem(dbc.NavLink("Transactions", href="/transactions")),
            dbc.NavItem(dbc.NavLink("Projections", href="/projections")),
            dbc.NavItem(dbc.NavLink("News", href="/news")),
            dbc.Col(
                html.Div([
                    dbc.Label("🌙", className="me-2"),
                    dbc.Switch(id="theme-switch", value=False, className="d-inline-block"),
                    dbc.Label("☀️", className="ms-2"),
                ], className="d-flex align-items-center"),
                width='auto',
            )
        ],
        brand="Premium Financial Dashboard",
        brand_href="/",
        color="primary",
        dark=True,
        className="mb-4",
    )

def layout_overview():
    """Returns the layout for the overview page."""
    return html.Div([
        html.Div(id='portfolio-cards', className="mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("Portfolio Allocation"),
                dbc.CardBody(dcc.Graph(id='allocation-chart', style={'height': '400px'}))
            ]), width=6),
            dbc.Col(dbc.Card([
                dbc.CardHeader("Holdings Performance"),
                dbc.CardBody(dcc.Graph(id='performance-chart', style={'height': '400px'}))
            ]), width=6),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("Historical Portfolio Performance (1Y)"),
                dbc.CardBody(dcc.Graph(id='historical-performance-chart', style={'height': '400px'}))
            ])),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("Sector Allocation"),
                dbc.CardBody(dcc.Graph(id='market-overview-chart', style={'height': '300px'}))
            ])),
        ], className="mb-4"),
    ])

def layout_holdings():
    """Returns the layout for the holdings page."""
    return dbc.Card([
        dbc.CardHeader(html.H5("Portfolio Holdings")),
        dbc.CardBody(html.Div(id='holdings-table'))
    ])

def layout_transactions():
    """Returns the layout for the transactions page."""
    return dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H5("Add Transaction")),
            dbc.CardBody(create_transaction_form())
        ]), width=6),
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H5("Recent Transactions")),
            dbc.CardBody(html.Div(id='transactions-list', style={'maxHeight': '400px', 'overflowY': 'auto'}))
        ]), width=6),
    ])

def layout_projections():
    """Returns the layout for the Monte Carlo projections page."""
    return dbc.Card([
        dbc.CardHeader(html.H5("Future Portfolio Projection (Monte Carlo)")),
        dbc.CardBody(dbc.Row([
            dbc.Col(dcc.Graph(id='monte-carlo-chart', style={'height': '400px'}), width=9),
            dbc.Col([
                dbc.Label("Projection Years"),
                dcc.Slider(id='mc-years-slider', min=1, max=30, step=1, value=10, marks={i: str(i) for i in [1, 5, 10, 20, 30]}),
                dbc.Label("Simulations", className="mt-3"),
                dcc.Slider(id='mc-sims-slider', min=100, max=1000, step=100, value=500, marks={i: str(i) for i in [100, 500, 1000]}),
                dbc.Button("Run Simulation", id="run-mc-button", color="info", className="w-100 mt-4"),
                dbc.Spinner(html.Div(id="mc-spinner"))
            ], width=3, className="d-flex flex-column justify-content-center")
        ]))
    ])

def layout_news():
    """Returns the layout for the news page."""
    return dbc.Card([
        dbc.CardHeader(html.H5("Relevant News")),
        dbc.CardBody(html.Div(id='news-feed-list', style={'maxHeight': '80vh', 'overflowY': 'auto'}))
    ])

def create_transaction_form():
    """Creates the transaction form component."""
    return dbc.Form([
        dbc.Row([
            dbc.Col([
                dbc.Label("Stock Symbol"),
                dbc.Input(id='symbol-input', placeholder='e.g., AAPL', type='text')
            ], width=6),
            dbc.Col([
                dbc.Label("Transaction Type"),
                dbc.Select(
                    id='type-input',
                    options=[
                        {'label': 'Buy', 'value': 'buy'},
                        {'label': 'Sell', 'value': 'sell'}
                    ],
                    value='buy'
                )
            ], width=6)
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Shares"),
                dbc.Input(id='shares-input', placeholder='Number of shares', type='number')
            ], width=6),
            dbc.Col([
                dbc.Label("Price (Optional)"),
                dbc.Input(id='price-input', placeholder='Current market price', type='number')
            ], width=6)
        ], className="mb-3"),
        dbc.Button(
            [html.I(className="fas fa-check me-2"), "Execute Transaction"],
            id='submit-transaction',
            color='primary',
            className="w-100"
        )
    ])

def create_premium_dash_app():
    """Create premium Dash application with modern UI"""
    app = dash.Dash(
        __name__,
        url_base_pathname='/',
        external_stylesheets=[dbc.themes.BOOTSTRAP, 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'],
        suppress_callback_exceptions=True
    )

    # Custom CSS styling
    app.layout = dbc.Container([
        # Header Section
        dcc.Location(id="url", refresh=False),
        create_navbar(),

        # Store for theme preference
        dcc.Store(id='theme-store', storage_type='local', data='light'),
        # Central data store for sharing data between callbacks
        dcc.Store(id='portfolio-data-store'),
        dcc.Store(id='transactions-data-store'),
        dcc.Store(id='news-data-store'),
        dcc.Store(id='historical-data-store'),
        # This link's href will be updated by the theme-switch callback
        html.Link(
            id="theme-link",
            rel="stylesheet",
            href=dbc.themes.BOOTSTRAP
        ),
        
        # Page content will be rendered here
        html.Div(id="page-content"),

        # Auto-refresh and alerts
        html.Div([
            dcc.Interval(id='refresh-interval', interval=30000, n_intervals=0),  # 30 seconds
            dbc.Toast(
                id="transaction-toast",
                header="Transaction Status",
                is_open=False,
                dismissable=True,
                icon="success",
                duration=4000,
                style={"position": "fixed", "top": 66, "right": 10, "width": 350, "zIndex": 9999}
            )
        ]),
        
        # Footer
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.P([
                    "Built with ❤️ by Ritu • ",
                    html.A("View API Docs", href="http://127.0.0.1:8000/docs", target="_blank", className="text-decoration-none")
                ], className="text-center text-muted small")
            ])
        ], className="mb-4")
        
    ], fluid=True, className="px-4")
    
    # Callback to render page content based on URL
    @app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
    def display_page(pathname):
        if pathname == '/holdings':
            return layout_holdings()
        elif pathname == '/transactions':
            return layout_transactions()
        elif pathname == '/projections':
            return layout_projections()
        elif pathname == '/news':
            return layout_news()
        else:
            return layout_overview()

    # MASTER DATA-FETCHING CALLBACK
    @app.callback(
        [Output('portfolio-data-store', 'data'),
         Output('transactions-data-store', 'data'),
         Output('news-data-store', 'data'),
         Output('historical-data-store', 'data')],
        Input('refresh-interval', 'n_intervals')
    )
    def update_master_data_store(n):
        """
        This single callback fetches all data on a timer and stores it.
        All other callbacks will read from these stores, making page navigation instant.
        """
        db = None
        try:
            db = SessionLocal()
            
            async def fetch_all_data():
                # Use asyncio.gather to run all async data fetching concurrently
                results = await asyncio.gather(
                    get_portfolio_summary(db),
                    get_transactions(db=db),
                    get_portfolio_news(db=db),
                    get_portfolio_history(db=db) # Keep history fetch for overview
                )
                return results

            # Run the async helper function once to get all data in parallel
            portfolio_data, transactions_data, news_data, history_data = asyncio.run(fetch_all_data())

            # Return data ready for dcc.Store (Pydantic models converted to dicts)
            return portfolio_data.model_dump(), transactions_data, news_data, history_data.model_dump()

        except Exception as e:
            print(f"Error fetching data: {e}")
            return None, None, None, None # Return None on error
        finally:
            if db:
                db.close()

    # Callback for the overview page
    @app.callback(
        [Output('portfolio-cards', 'children'),
         Output('allocation-chart', 'figure'),
         Output('performance-chart', 'figure'),
         Output('market-overview-chart', 'figure'),
         Output('historical-performance-chart', 'figure')],
        [Input('portfolio-data-store', 'data'),
         Input('historical-data-store', 'data'),
         Input('theme-store', 'data')]
    )
    def update_overview_page(portfolio_data, history_data, theme):
        if not portfolio_data or not history_data:
            return (dbc.Alert("Loading portfolio data...", color="info"), go.Figure(), go.Figure(), go.Figure(), go.Figure())
        
        # Determine Plotly template based on theme
        plotly_template = "plotly_dark" if theme == "dark" else "plotly_white"

        # Portfolio Summary Cards
        cards = create_portfolio_cards(portfolio_data)
        
        # Charts
        allocation_fig = create_allocation_chart(portfolio_data, plotly_template)
        performance_fig = create_performance_chart(portfolio_data, plotly_template)
        market_overview_fig = create_market_overview_chart(portfolio_data, plotly_template)
        historical_fig = create_historical_chart(history_data, plotly_template)
        
        return cards, allocation_fig, performance_fig, market_overview_fig, historical_fig

    # Callback for holdings page
    @app.callback(
        Output('holdings-table', 'children'),
        Input('portfolio-data-store', 'data')
    )
    def update_holdings_page(portfolio_data):
        if not portfolio_data:
            return dbc.Alert("Loading holdings data...", color="info")
        return create_holdings_table(portfolio_data.get('holdings', []))

    # Callback for transactions page
    @app.callback(
        Output('transactions-list', 'children'),
        Input('transactions-data-store', 'data')
    )
    def update_transactions_page(transactions_data):
        if not transactions_data:
            return dbc.Alert("Loading transaction data...", color="info")
        return create_transactions_list(transactions_data)

    # Callback for news page
    @app.callback(
        Output('news-feed-list', 'children'),
        Input('news-data-store', 'data')
    )
    def update_news_page(news_data):
        if news_data is None:
            return dbc.Alert("Loading news...", color="info")
        return create_news_feed(news_data)

    # Theme switcher callback
    @app.callback(
        [Output('theme-link', 'href'),
         Output('theme-store', 'data')],
        Input('theme-switch', 'value')
    )
    def switch_theme(is_light):
        theme = "light" if is_light else "dark"
        theme_url = dbc.themes.BOOTSTRAP if is_light else dbc.themes.DARKLY # Use DARKLY for dark mode
        return theme_url, theme
    
    # Transaction submission callback
    @app.callback(
        [Output('transaction-toast', 'is_open'),
         Output('transaction-toast', 'children'),
         Output('transaction-toast', 'icon'),
         Output('symbol-input', 'value'),
         Output('shares-input', 'value'),
         Output('price-input', 'value')],
        [Input('submit-transaction', 'n_clicks')],
        [State('symbol-input', 'value'),
         State('type-input', 'value'),
         State('shares-input', 'value'),
         State('price-input', 'value')],
        prevent_initial_call=True
    )
    def submit_transaction(n_clicks, symbol, trans_type, shares, price):
        if not n_clicks or not symbol or not shares:
            return False, "", "danger", symbol, shares, price
        db = SessionLocal()
        try:
            # Call the async function directly from the sync callback via asyncio.run()
            transaction_data = {
                "symbol": symbol.upper(),
                "transaction_type": trans_type,
                "shares": float(shares),
                "price": float(price) if price else None,
            }
            result = asyncio.run(add_transaction(TransactionRequest(**transaction_data), db))

            if result:
                price_info = f" at ${result['current_price']:.2f}" if result.get('current_price') else ""
                message = f"Successfully {trans_type} {shares} shares of {symbol.upper()}{price_info}"
                return True, message, "success", "", "", ""
            else:
                return True, "Error: Transaction failed", "danger", symbol, shares, price
                
        except Exception as e:
            return True, f"Error: {str(e)}", "danger", symbol, shares, price
        finally:
            db.close()
    
    # Monte Carlo simulation callback
    @app.callback(
        [Output('monte-carlo-chart', 'figure'),
         Output('mc-spinner', 'children')],
        [Input('run-mc-button', 'n_clicks')],
        [State('mc-years-slider', 'value'),
         State('mc-sims-slider', 'value'),
         State('theme-store', 'data')],
        prevent_initial_call=True
    )
    def run_monte_carlo(n_clicks, years, sims, theme):
        if not n_clicks or not years or not sims:
            return go.Figure(layout={'template': "plotly_dark" if theme == "dark" else "plotly_white"}), ""
        db = SessionLocal()
        try:
            # Call the async function directly from the sync callback via asyncio.run()
            projection_data = asyncio.run(project_portfolio_value(years, sims, db))
            if not projection_data:
                raise ValueError("Simulation returned no data.")
            
            plotly_template = "plotly_dark" if theme == "dark" else "plotly_white"
            # Create figure
            fig = go.Figure()
            results_df = pd.DataFrame(projection_data.model_dump()['results'])
            
            # Plot all simulation paths
            for col in results_df.columns:
                fig.add_trace(go.Scatter(y=results_df[col], mode='lines', line=dict(width=0.5, color='lightgray'), showlegend=False))
            
            # Plot median, 25th and 75th percentiles
            fig.add_trace(go.Scatter(y=results_df.quantile(0.5, axis=1), mode='lines', line=dict(width=2, color='blue'), name='Median'))
            fig.add_trace(go.Scatter(y=results_df.quantile(0.75, axis=1), mode='lines', line=dict(width=1, dash='dash', color='green'), name='75th Percentile'))
            fig.add_trace(go.Scatter(y=results_df.quantile(0.25, axis=1), mode='lines', line=dict(width=1, dash='dash', color='red'), name='25th Percentile'))

            fig.update_layout(
                title=f"{sims} Simulations over {years} Years", 
                xaxis_title="Trading Days", 
                yaxis_title="Portfolio Value ($)",
                template=plotly_template
            )
            return fig, ""
        except Exception as e:
            print(f"Error running Monte Carlo: {e}")
            return go.Figure(layout={'template': "plotly_dark" if theme == "dark" else "plotly_white"}).add_annotation(text=f"Error: {e}", showarrow=False), ""
        finally:
            db.close()

    return app

# Helper functions for creating UI components
def create_loading_components():
    """Create loading state components"""
    loading_card = dbc.Alert("Loading portfolio data...", color="info", className="text-center")
    empty_fig = go.Figure()
    empty_fig.add_annotation(text="Loading...", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
    return loading_card, empty_fig, empty_fig, "Loading...", "Loading...", empty_fig

def create_portfolio_cards(portfolio_data):
    """Create portfolio summary cards"""
    total_value = portfolio_data['total_value']
    total_gain_loss = portfolio_data['total_gain_loss']
    total_gain_loss_percent = portfolio_data['total_gain_loss_percent']
    day_change = portfolio_data['day_change']
    day_change_percent = portfolio_data['day_change_percent']
    holdings_count = portfolio_data['holdings_count']
    
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-wallet fa-2x text-primary mb-2"),
                        html.H3(f"${total_value:,.2f}", className="mb-1"),
                        html.P("Total Portfolio Value", className="text-muted mb-0"),
                        html.Small([
                            html.I(className="fas fa-arrow-up text-success me-1") if day_change >= 0 else html.I(className="fas fa-arrow-down text-danger me-1"),
                            f"${abs(day_change):,.2f} ({day_change_percent:+.2f}%) today"
                        ], className=f"{'text-success' if day_change >= 0 else 'text-danger'}")
                    ], className="text-center")
                ])
            ], className="h-100 shadow-sm border-0", style={'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 'color': 'white'})
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-chart-line fa-2x text-success mb-2"),
                        html.H3(f"${total_gain_loss:,.2f}", className="mb-1"),
                        html.P("Total Gain/Loss", className="text-muted mb-0"),
                        html.Small(f"{total_gain_loss_percent:+.2f}% return", 
                                 className=f"{'text-success' if total_gain_loss >= 0 else 'text-danger'}")
                    ], className="text-center")
                ])
            ], className="h-100 shadow-sm border-0", 
               style={'background': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)' if total_gain_loss >= 0 else 'linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)', 'color': 'white'})
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-coins fa-2x text-warning mb-2"),
                        html.H3(f"{holdings_count}", className="mb-1"),
                        html.P("Holdings", className="text-muted mb-0"),
                        html.Small(f"${portfolio_data['cash_balance']:,.2f} cash available", className="text-muted")
                    ], className="text-center")
                ])
            ], className="h-100 shadow-sm border-0", style={'background': 'linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)', 'color': '#2c3e50'})
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-percentage fa-2x text-info mb-2"),
                        html.H3(f"{(total_value / (portfolio_data['total_cost']) * 100 - 100):+.1f}%", className="mb-1"),
                        html.P("Total Return", className="text-muted mb-0"),
                        html.Small(f"Since inception", className="text-muted")
                    ], className="text-center")
                ])
            ], className="h-100 shadow-sm border-0", style={'background': 'linear-gradient(135deg, #fad0c4 0%, #fad0c4 1%, #ffd1ff 100%)', 'color': '#2c3e50'})
        ], width=3)
    ], className="mb-4")

def create_allocation_chart(portfolio_data, template):
    """Create modern portfolio allocation pie chart"""
    holdings = portfolio_data['holdings']
    if not holdings and portfolio_data['cash_balance'] == 0:
        return go.Figure()
    
    # Prepare data
    labels = [h['symbol'] for h in holdings]   # ✅ only ticker in legend
    hover_texts = [f"{h['symbol']} - {h['company_name']}" for h in holdings]
    values = [h['market_value'] for h in holdings]
    colors = px.colors.qualitative.Set3[:len(holdings)]
    
    # Add cash if significant
    if portfolio_data['cash_balance'] > 0:
        labels.append("Cash")
        hover_texts.append("Cash Balance")
        values.append(portfolio_data['cash_balance'])
        colors.append('#95a5a6')
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors, line=dict(color='white', width=2)),
        textinfo='percent',   # ✅ only % inside pie
        textfont_size=12,
        hovertext=hover_texts,
        hovertemplate='<b>%{hovertext}</b><br>Value: $%{value:,.2f}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(text="Portfolio Allocation", x=0.5, font=dict(size=16, color='#2c3e50')),
        font=dict(size=11),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,             # ✅ push legend outside
            font=dict(size=10)
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        template=template
    )
    
    return fig


def create_performance_chart(portfolio_data, template):
    """Create holdings performance bar chart"""
    holdings = portfolio_data['holdings']
    if not holdings: 
        return go.Figure()
    
    # Sort by gain/loss percentage
    holdings_sorted = sorted(holdings, key=lambda x: x['gain_loss_percent'])
    
    symbols = [h['symbol'] for h in holdings_sorted]
    returns = [h['gain_loss_percent'] for h in holdings_sorted]
    colors = ['#e74c3c' if r < 0 else '#27ae60' for r in returns]
    
    fig = go.Figure(data=[go.Bar(
        x=symbols,
        y=returns,
        marker=dict(color=colors, line=dict(color='white', width=1)),
        text=[f"{r:+.1f}%" for r in returns],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Return: %{y:+.2f}%<br><extra></extra>'
    )])
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=dict(text="Holdings Performance", x=0.5, font=dict(size=16, color='#2c3e50')),
        xaxis=dict(title="Stock Symbol", tickfont=dict(size=11)),
        yaxis=dict(title="Return (%)", tickfont=dict(size=11)),
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=40),
        font=dict(size=11),
        template=template
    )
    
    return fig

def create_market_overview_chart(portfolio_data, template):
    """Create market overview chart showing sector allocation"""
    holdings = portfolio_data['holdings']
    if not holdings:
        return go.Figure()
    
    # Group by sector
    sector_data = {}
    for holding in holdings:
        sector = holding['sector']
        if sector in sector_data:
            sector_data[sector] += holding['market_value']
        else:
            sector_data[sector] = holding['market_value']
    
    sectors = list(sector_data.keys())
    values = list(sector_data.values())
    
    fig = go.Figure(data=[go.Bar(
        x=sectors,
        y=values,
        marker=dict(color=px.colors.qualitative.Pastel[:len(sectors)]),
        text=[f"${v:,.0f}" for v in values],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Value: $%{y:,.2f}<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(text="Sector Allocation", x=0.5, font=dict(size=16, color='#2c3e50')),
        xaxis=dict(title="Sector", tickangle=45, tickfont=dict(size=10)),
        yaxis=dict(title="Value ($)", tickfont=dict(size=11)),
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=80),
        font=dict(size=11),
        template=template
    )
    
    return fig

def create_historical_chart(history_data, template):
    """Create historical portfolio performance line chart."""
    if not history_data or not history_data.get('dates'):
        fig = go.Figure()
        fig.update_layout(template=template)
        return fig

    df = pd.DataFrame(history_data)
    df['dates'] = pd.to_datetime(df['dates'])

    fig = go.Figure(data=[go.Scatter(
        x=df['dates'],
        y=df['values'],
        mode='lines',
        fill='tozeroy',
        line=dict(color='#007bff', width=2),
        hovertemplate='<b>%{x|%B %d, %Y}</b><br>Value: $%{y:,.2f}<extra></extra>'
    )])

    fig.update_layout(
        xaxis_title="Date", yaxis_title="Portfolio Value ($)", template=template
    )
    return fig

def create_holdings_table(holdings):
    """Create professional holdings table"""
    if not holdings:
        return dbc.Alert("No holdings found", color="info")
    
    table_data = []
    for holding in holdings:
        table_data.append({
            'Symbol': holding['symbol'],
            'Company': holding['company_name'][:30] + "..." if len(holding['company_name']) > 30 else holding['company_name'],
            'Sector': holding['sector'],
            'Shares': f"{holding['shares']:.2f}",
            'Avg Cost': f"${holding['avg_cost']:.2f}",
            'Current Price': f"${holding['current_price']:.2f}",
            'Market Value': f"${holding['market_value']:,.2f}",
            'Day Change': f"${holding['day_change']:+,.2f}",
            'Total Return': f"{holding['gain_loss_percent']:+.2f}%",
            'Weight': f"{holding['weight']:.1f}%"
        })
    
    return dash_table.DataTable(
        data=table_data,
        columns=[
            {'name': 'Symbol', 'id': 'Symbol', 'type': 'text'},
            {'name': 'Company', 'id': 'Company', 'type': 'text'},
            {'name': 'Sector', 'id': 'Sector', 'type': 'text'},
            {'name': 'Shares', 'id': 'Shares', 'type': 'numeric'},
            {'name': 'Avg Cost', 'id': 'Avg Cost', 'type': 'text'},
            {'name': 'Current Price', 'id': 'Current Price', 'type': 'text'},
            {'name': 'Market Value', 'id': 'Market Value', 'type': 'text'},
            {'name': 'Day Change', 'id': 'Day Change', 'type': 'text'},
            {'name': 'Total Return', 'id': 'Total Return', 'type': 'text'},
            {'name': 'Weight', 'id': 'Weight', 'type': 'text'}
        ],
        style_cell={
            'textAlign': 'left',
            'padding': '12px',
            'fontFamily': 'Arial, sans-serif',
            'fontSize': '14px',
            'border': '1px solid #dee2e6'
        },
        style_header={
            'backgroundColor': '#f8f9fa',
            'fontWeight': 'bold',
            'color': '#212529',
            'border': '1px solid #dee2e6'
        },
        style_data_conditional=[
            {
                'if': {'column_id': 'Total Return', 'filter_query': '{Total Return} contains "+"'},
                'color': '#27ae60',
                'fontWeight': 'bold'
            },
            {
                'if': {'column_id': 'Total Return', 'filter_query': '{Total Return} contains "-"'},
                'color': '#e74c3c',
                'fontWeight': 'bold'
            },
            {
                'if': {'column_id': 'Day Change', 'filter_query': '{Day Change} contains "+"'},
                'color': '#27ae60'
            },
            {
                'if': {'column_id': 'Day Change', 'filter_query': '{Day Change} contains "-"'},
                'color': '#e74c3c'
            },
        ],
        style_as_list_view=True, # Removes vertical lines for a cleaner look
        style_data={
            'borderBottom': '1px solid #dee2e6',
            'backgroundColor': 'transparent'
        },
        sort_action='native',
        page_size=10,
        style_table={'overflowX': 'auto'}
    )

def create_transactions_list(transactions_data):
    """Create recent transactions list"""
    if not transactions_data:
        return dbc.Alert("No recent transactions", color="info")
    
    transaction_items = []
    for transaction in transactions_data[:8]:  # Show last 8 transactions
        date = datetime.fromisoformat(transaction['date'].replace('Z', '+00:00')).strftime('%m/%d/%Y')
        
        icon_class = "fas fa-arrow-up text-success" if transaction['type'] == 'buy' else "fas fa-arrow-down text-danger"
        color_class = "border-start border-success border-3" if transaction['type'] == 'buy' else "border-start border-danger border-3"
        
        transaction_items.append(
            dbc.ListGroupItem([
                html.Div([
                    html.Div([
                        html.I(className=icon_class),
                        html.Strong(f" {transaction['type'].upper()} {transaction['symbol']}", className="ms-2"),
                        html.Span(f" • {date}", className="text-muted small ms-2")
                    ]),
                    html.Div([
                        html.Span(f"{transaction['shares']:.2f} shares @ ${transaction['price']:.2f}"),
                        html.Br(),
                        html.Small(f"Total: ${transaction['total']:,.2f}", className="text-muted")
                    ])
                ])
            ], className=f"py-2 {color_class}")
        )
    
    return dbc.ListGroup(transaction_items, flush=True)

def create_news_feed(news_data):
    """Create a list of recent news articles."""
    if not news_data:
        return dbc.Alert("No recent news found for your holdings.", color="info")
    
    news_items = []
    for article in news_data:
        news_items.append(
            dbc.ListGroupItem([
                html.A(article['title'], href=article['link'], target="_blank", className="fw-bold text-decoration-none"),
                html.P(article['publisher'], className="text-muted small mb-0")
            ], className="py-2")
        )
    
    return dbc.ListGroup(news_items, flush=True)


# =============================================================================
# MAIN APPLICATION RUNNER
# =============================================================================

# Apply the patch to allow nested asyncio event loops.
nest_asyncio.apply()

# Create the Dash app instance now that the creation function is defined.
dash_app = create_premium_dash_app()

# Mount the Dash app as a sub-application on the FastAPI server
app.mount("/", WSGIMiddleware(dash_app.server))

if __name__ == "__main__":
    # This block is for local development.
    # The deployment server will use the 'app' object directly.
    print("🚀 Initializing premium database for local development...")
    initialize_premium_data()
    print("✅ Starting development server...")
    print("🌟 Your dashboard will be available at http://127.0.0.1:8000")
    uvicorn.run("financial_dashboard:app", host="127.0.0.1", port=8000, reload=True)