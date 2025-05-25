from typing import Dict, List, Set
import re

# Options Domain Keywords
OPTIONS_KEYWORDS = {
    'basic_terms': {
        'call', 'put', 'option', 'strike', 'expiry', 'expiration', 'premium', 
        'exercise', 'assignment', 'intrinsic', 'extrinsic', 'time value'
    },
    'pricing_models': {
        'black-scholes', 'black scholes', 'binomial', 'trinomial', 'monte carlo',
        'american', 'european', 'asian', 'barrier', 'exotic'
    },
    'greeks': {
        'delta', 'gamma', 'theta', 'vega', 'rho', 'charm', 'color', 'vanna',
        'volga', 'speed', 'zomma'
    },
    'strategies': {
        'straddle', 'strangle', 'spread', 'butterfly', 'condor', 'collar',
        'covered call', 'protective put', 'iron condor', 'calendar spread'
    },
    'risk_terms': {
        'volatility', 'implied volatility', 'historical volatility', 'skew',
        'hedging', 'delta neutral', 'gamma hedging', 'arbitrage'
    }
}

# Validation Patterns
TICKER_PATTERN = re.compile(r'^[A-Z]{1,5}$')
STRIKE_PATTERN = re.compile(r'^\d+\.?\d*$')
DATE_PATTERNS = [
    re.compile(r'\d{4}-\d{2}-\d{2}'),  # YYYY-MM-DD
    re.compile(r'\d{2}/\d{2}/\d{4}'),  # MM/DD/YYYY
    re.compile(r'\d{1,2}\s+(days?|weeks?|months?|years?)'),  # Natural language
]

# Market Conventions
TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365
BUSINESS_HOURS = (9, 16)  # 9 AM to 4 PM EST

# Risk Boundaries
PRICING_BOUNDS = {
    'min_stock_price': 0.01,
    'max_stock_price': 10000,
    'min_strike': 0.01,
    'max_strike': 10000,
    'min_time_to_expiry': 1/365,  # 1 day
    'max_time_to_expiry': 5,  # 5 years
    'min_volatility': 0.001,  # 0.1%
    'max_volatility': 5.0,  # 500%
    'min_interest_rate': -0.1,  # -10%
    'max_interest_rate': 0.3,  # 30%
}

# Error Messages
ERROR_MESSAGES = {
    'invalid_ticker': "Invalid ticker symbol. Please use valid stock symbols (e.g., AAPL, GOOGL).",
    'invalid_strike': "Strike price must be a positive number.",
    'invalid_expiry': "Expiry date must be in the future and within reasonable bounds.",
    'out_of_domain': "This query is outside the options trading domain. Please ask about options pricing, strategies, or analysis.",
    'insufficient_data': "Insufficient market data available for this analysis.",
    'calculation_error': "Unable to perform calculation with given parameters.",
    'high_uncertainty': "Results have high uncertainty. Please verify assumptions.",
}