from typing import Dict, List, Any
import json

# Options Theory Knowledge Base
OPTIONS_THEORY = {
    "black_scholes": {
        "description": "The Black-Scholes model is used for pricing European options",
        "formula": "C = S*N(d1) - K*exp(-r*T)*N(d2)",
        "assumptions": [
            "Constant volatility",
            "Constant risk-free rate", 
            "No dividends",
            "European exercise only",
            "Lognormal stock price distribution"
        ],
        "parameters": {
            "S": "Current stock price",
            "K": "Strike price", 
            "T": "Time to expiration",
            "r": "Risk-free interest rate",
            "σ": "Volatility"
        },
        "limitations": [
            "Not suitable for American options",
            "Assumes constant volatility",
            "Ignores dividends in basic form"
        ]
    },
    
    "greeks": {
        "delta": {
            "definition": "Rate of change of option price with respect to underlying price",
            "range_call": "(0, 1)",
            "range_put": "(-1, 0)",
            "interpretation": "Hedge ratio for delta-neutral portfolio"
        },
        "gamma": {
            "definition": "Rate of change of delta with respect to underlying price", 
            "range": "(0, ∞)",
            "interpretation": "Convexity measure, highest for at-the-money options"
        },
        "theta": {
            "definition": "Rate of change of option price with respect to time",
            "range": "(-∞, 0)",
            "interpretation": "Time decay, always negative for long options"
        },
        "vega": {
            "definition": "Rate of change of option price with respect to volatility",
            "range": "(0, ∞)",
            "interpretation": "Volatility sensitivity, highest for at-the-money options"
        },
        "rho": {
            "definition": "Rate of change of option price with respect to interest rate",
            "interpretation": "Interest rate sensitivity, more important for longer-term options"
        }
    },
    
    "option_strategies": {
        "long_call": {
            "description": "Bullish strategy with unlimited upside potential",
            "max_profit": "Unlimited",
            "max_loss": "Premium paid",
            "breakeven": "Strike + Premium"
        },
        "long_put": {
            "description": "Bearish strategy with high profit potential",
            "max_profit": "Strike - Premium",
            "max_loss": "Premium paid", 
            "breakeven": "Strike - Premium"
        },
        "straddle": {
            "description": "Volatility strategy expecting large price movement",
            "construction": "Long call + Long put (same strike and expiry)",
            "max_profit": "Unlimited",
            "max_loss": "Total premium paid",
            "breakeven": ["Strike - Total Premium", "Strike + Total Premium"]
        },
        "strangle": {
            "description": "Volatility strategy with lower cost than straddle",
            "construction": "Long call (higher strike) + Long put (lower strike)",
            "max_profit": "Unlimited",
            "max_loss": "Total premium paid"
        }
    },
    
    "risk_warnings": [
        "Options trading involves substantial risk and is not suitable for all investors",
        "Past performance does not guarantee future results", 
        "All calculations are theoretical and may not reflect actual market conditions",
        "Consider transaction costs and bid-ask spreads in real trading",
        "Market volatility can significantly impact option values",
        "Options may expire worthless, resulting in total loss of premium"
    ]
}

def get_theory_context(topic: str) -> Dict[str, Any]:
    """Retrieve theory context for grounding"""
    return OPTIONS_THEORY.get(topic, {})

def get_all_risk_warnings() -> List[str]:
    """Get all risk warnings for inclusion in responses"""
    return OPTIONS_THEORY["risk_warnings"]