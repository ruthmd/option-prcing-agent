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
        "short_call": {
            "description": "Bearish-to-neutral income strategy that sells a call without owning the underlying; profits if the stock stays below the strike",
            "construction": "Sell 1 call",
            "max_profit": "Premium received",
            "max_loss": "Unlimited",
            "breakeven": "Strike + Premium"
        },
        "short_put": {
            "description": "Bullish-to-neutral income strategy that sells a put; profits if the stock stays above the strike, obligates the seller to buy shares if assigned",
            "construction": "Sell 1 put",
            "max_profit": "Premium received",
            "max_loss": "Strike - Premium (large, since the stock can fall toward zero)",
            "breakeven": "Strike - Premium"
        },
        "covered_call": {
            "description": "Income strategy that sells a call against stock already owned, capping upside in exchange for premium income",
            "construction": "Long 100 shares + Short 1 call",
            "max_profit": "(Strike - Stock Purchase Price) + Premium received",
            "max_loss": "Stock Purchase Price - Premium received (if stock falls to zero)",
            "breakeven": "Stock Purchase Price - Premium received"
        },
        "protective_put": {
            "description": "Hedging strategy that buys a put against owned stock to insure against downside risk",
            "construction": "Long 100 shares + Long 1 put",
            "max_profit": "Unlimited",
            "max_loss": "Stock Purchase Price - Strike + Premium paid",
            "breakeven": "Stock Purchase Price + Premium paid"
        },
        "bull_call_spread": {
            "description": "Moderately bullish, defined-risk strategy that buys a lower-strike call and sells a higher-strike call to reduce cost at the expense of capped upside",
            "construction": "Long call (lower strike) + Short call (higher strike), same expiry",
            "max_profit": "(Higher Strike - Lower Strike) - Net Premium paid",
            "max_loss": "Net Premium paid",
            "breakeven": "Lower Strike + Net Premium paid"
        },
        "bear_put_spread": {
            "description": "Moderately bearish, defined-risk strategy that buys a higher-strike put and sells a lower-strike put to reduce cost at the expense of capped downside profit",
            "construction": "Long put (higher strike) + Short put (lower strike), same expiry",
            "max_profit": "(Higher Strike - Lower Strike) - Net Premium paid",
            "max_loss": "Net Premium paid",
            "breakeven": "Higher Strike - Net Premium paid"
        },
        "long_straddle": {
            "description": "Volatility strategy expecting a large price movement in either direction",
            "construction": "Long call + Long put (same strike and expiry)",
            "max_profit": "Unlimited",
            "max_loss": "Total premium paid",
            "breakeven": ["Strike - Total Premium", "Strike + Total Premium"]
        },
        "short_straddle": {
            "description": "Volatility-selling strategy that profits when the stock stays near the strike; carries unlimited risk if the stock moves sharply in either direction",
            "construction": "Short call + Short put (same strike and expiry)",
            "max_profit": "Total premium received",
            "max_loss": "Unlimited",
            "breakeven": ["Strike - Total Premium", "Strike + Total Premium"]
        },
        "long_strangle": {
            "description": "Volatility strategy with lower cost than a straddle, requiring a larger price move to become profitable",
            "construction": "Long call (higher strike) + Long put (lower strike), same expiry",
            "max_profit": "Unlimited",
            "max_loss": "Total premium paid",
            "breakeven": ["Lower Strike - Total Premium", "Higher Strike + Total Premium"]
        },
        "short_strangle": {
            "description": "Volatility-selling strategy with a wider profitable range than a short straddle, but still carries unlimited risk on a large move",
            "construction": "Short call (higher strike) + Short put (lower strike), same expiry",
            "max_profit": "Total premium received",
            "max_loss": "Unlimited",
            "breakeven": ["Lower Strike - Total Premium", "Higher Strike + Total Premium"]
        },
        "iron_condor": {
            "description": "Defined-risk, range-bound strategy that combines a short strangle with a further out-of-the-money long strangle to cap risk; profits when the stock stays within the short strikes",
            "construction": "Short put (higher) + Long put (lower) + Short call (lower) + Long call (higher), same expiry",
            "max_profit": "Net premium received",
            "max_loss": "(Width of either spread) - Net premium received",
            "breakeven": ["Short Put Strike - Net Premium", "Short Call Strike + Net Premium"]
        },
        "iron_butterfly": {
            "description": "Defined-risk variant of the short straddle that adds protective long options further out-of-the-money; narrower profit range than an iron condor but typically higher premium collected",
            "construction": "Short call + Short put (same at-the-money strike) + Long call (higher) + Long put (lower)",
            "max_profit": "Net premium received",
            "max_loss": "(Width of either wing) - Net premium received",
            "breakeven": ["ATM Strike - Net Premium", "ATM Strike + Net Premium"]
        },
        "long_butterfly": {
            "description": "Defined-risk, low-cost strategy that bets the stock pins near the middle strike at expiry; low probability of max profit but a favorable reward-to-risk ratio",
            "construction": "Long call (lower strike) + 2x Short call (middle strike) + Long call (higher strike), equally spaced",
            "max_profit": "(Middle Strike - Lower Strike) - Net premium paid",
            "max_loss": "Net premium paid",
            "breakeven": ["Lower Strike + Net Premium", "Higher Strike - Net Premium"]
        },
        "short_butterfly": {
            "description": "Opposite of the long butterfly; profits when the stock moves significantly away from the middle strike in either direction",
            "construction": "Short call (lower strike) + 2x Long call (middle strike) + Short call (higher strike), equally spaced",
            "max_profit": "Net premium received",
            "max_loss": "(Middle Strike - Lower Strike) - Net premium received",
            "breakeven": ["Lower Strike + Net Premium", "Higher Strike - Net Premium"]
        },
        "calendar_spread": {
            "description": "Time-decay strategy that sells a near-term option and buys a longer-term option at the same strike, profiting from the faster time decay of the short-dated leg",
            "construction": "Short near-term option + Long longer-term option, same strike and type",
            "max_profit": "Limited, maximized if the stock is near the strike at near-term expiry",
            "max_loss": "Net premium paid",
            "breakeven": "Depends on implied volatility term structure; not a single fixed price"
        },
        "collar": {
            "description": "Hedging strategy for an existing stock position that buys a protective put and sells a covered call to finance it, capping both downside and upside",
            "construction": "Long 100 shares + Long put (lower strike) + Short call (higher strike)",
            "max_profit": "(Call Strike - Stock Purchase Price) +/- Net Premium",
            "max_loss": "(Stock Purchase Price - Put Strike) +/- Net Premium",
            "breakeven": "Stock Purchase Price +/- Net Premium"
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