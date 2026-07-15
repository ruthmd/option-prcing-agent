# Vanila options

python main.py --query 'Price a call option on AAPL with strike $150, expiry in 30 days'
python main.py --query 'Price a European call option on AAPL with strike $150, expiry in 30 days'
python main.py --query 'Price a put option on AAPL strike $150 considering early exercise, expiry in 30 days'
python main.py --query 'Price an American barrier up-and-out call option on TSLA strike $200, expiry in 60 days'
python main.py --query 'Price a put option on MSFT with strike $375, expiry in 90 days'


#Complex option stratagies

# Single-leg (bullish/bearish directional)
python main.py --query 'Analyze a long call strategy on AAPL strike $150 expiring in 30 days'
python main.py --query 'Analyze a covered call strategy on MSFT strike $400 expiring in 45 days'
python main.py --query 'Analyze a protective put strategy on TSLA strike $180 expiring in 30 days'
python main.py --query 'Analyze a short put strategy on SPY strike $420 expiring in 30 days'

# Vertical spreads (defined risk, directional)
python main.py --query 'Analyze a bull call spread on AAPL expiring in 45 days'
python main.py --query 'Analyze a bear put spread on NVDA expiring in 45 days'

# Volatility strategies — long (betting on a big move) vs. short (betting on calm)
python main.py --query 'Analyze a long straddle strategy on TSLA expiring in 30 days'
python main.py --query 'Analyze a short straddle strategy on SPY expiring in 30 days'
python main.py --query 'Analyze a long strangle strategy on GOOGL expiring in 30 days'
python main.py --query 'Analyze a short strangle strategy on QQQ expiring in 30 days'

# Range-bound, defined-risk (the newly-grounded ones — good test of the KB expansion)
python main.py --query 'Analyze an iron condor strategy on SPY expiring in 45 days'
python main.py --query 'Analyze an iron butterfly strategy on AAPL expiring in 30 days'
python main.py --query 'Analyze a long butterfly strategy on MSFT expiring in 30 days'
python main.py --query 'Analyze a short butterfly strategy on GOOGL expiring in 30 days'

# Time-decay / hedging (the ones I just fixed the routing gap for)
python main.py --query 'Analyze a calendar spread strategy on AAPL expiring in 60 days'
python main.py --query 'Analyze a collar strategy on TSLA strike $200 expiring in 45 days'
python main.py --query 'Analyze a short call strategy on NVDA expiring in 30 days'


# Batched queries (WARNING: this can be token hungry)
python main.py --batch batched_queries/batch_queries.json          # no charts
python main.py --batch batched_queries/batch_queries.json --viz    # with charts + consolidated dashboard