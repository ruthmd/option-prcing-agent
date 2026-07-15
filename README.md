# Options Pricing Agent

An agentic AI system for options trading analysis with built-in guardrails and grounding mechanisms.

## Features

- 🤖 **Multi-Agent Architecture**: Specialized agents for different analysis types
- 🛡️ **Built-in Guardrails**: Reduces hallucinations and ensures domain-specific responses
- 📊 **Real-time Market Data**: Integration with yfinance for live market data
- 🧠 **Knowledge Grounding**: FAISS-based knowledge retrieval from established options theory
- ✅ **Output Validation**: Comprehensive validation of all calculations and results
- 📈 **Comprehensive Analysis**: Option pricing, Greeks, volatility, strategies, and more


## Core Agents

- **Market Data Agent**
  - **Role:** Real-time data collector
  - **Goal:** Fetch accurate, current market data
  - **Tools:** yfinance integration, data validation

- **Options Pricing Agent**
  - **Role:** Option valuation specialist
  - **Goal:** Calculate theoretical option prices
  - **Tools:** Black-Scholes, Binomial trees, Monte Carlo

- **Greeks Calculator Agent**
  - **Role:** Risk metrics analyst
  - **Goal:** Compute all Greeks (Delta, Gamma, Theta, Vega, Rho)
  - **Tools:** Numerical differentiation, analytical formulas

- **Volatility Analysis Agent**
  - **Role:** Volatility specialist
  - **Goal:** Analyze implied/historical volatility
  - **Tools:** Volatility smile modeling (by moneyness bucket), IV calculations

- **Strategy Analysis Agent**
  - **Role:** Options strategy expert
  - **Goal:** Evaluate complex multi-leg strategies
  - **Tools:** P&L analysis, risk-reward calculations

- **Validation Agent**
  - **Role:** Quality assurance specialist
  - **Goal:** Validate all calculations and outputs
  - **Tools:** Cross-validation, sanity checks, bounds testing


## LLM Providers

The agent uses two independently configurable LLM roles, both set in `.env`:

| Role | Env Var | Purpose |
|------|---------|---------|
| **Core generation** | `LLM_PROVIDER` | Writes the natural-language explanation of each computed result |
| **Verification (LLM-as-judge)** | `LLM_JUDGE` | Independently fact-checks educational / low-confidence responses against the knowledge base |

### Switching the core provider (`LLM_PROVIDER`)

Options: `local` (Ollama) | `openai` (GPT-4o) | `claude` (Claude Sonnet)

```bash
# .env
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-6
```

All pricing, Greeks, and strategy math is computed deterministically and is unaffected by this choice — the LLM only writes the explanation of an already-computed result, never the numbers themselves.

### Configuring the verification judge (`LLM_JUDGE`)

Options: `local` | `openai` | `claude` | `none` (disables the check entirely)

```bash
# .env
LLM_JUDGE=openai
OPENAI_API_KEY=sk-...
```

Keep `LLM_JUDGE` set to a **different** provider than `LLM_PROVIDER` — the judge is only a genuine second opinion when it isn't the same model grading its own output. The judge doesn't run on every query: it only fires for educational queries or when grounding confidence is already low, to avoid extra cost/latency on routine pricing math. Its verdict, and which provider produced it, is visible in the logs:

```
LLM-as-judge: triggered (query_type=educational, confidence=0.95) — checking with provider='openai'
LLM-as-judge: provider='openai' verdict — grounded=True, confidence=0.90, concerns=0
```

## Usage

### Interactive mode

```bash
python main.py
```

Prompts for queries one at a time; charts are generated automatically and opened in the browser.

### Plain query

```bash
python main.py --query 'Price a call option on AAPL with strike $150, expiring in 30 days'
```

> ⚠️ **Use single quotes.** Queries containing a `$` amount (e.g. `$150`) will be silently mangled by shell variable expansion if wrapped in double quotes instead.

No charts are generated in this mode by default.

### Query with visualizations

Add `--viz` to generate charts (payoff diagrams, Greeks radar, volatility smile, etc.) and open a dashboard in the browser:

```bash
python main.py --query 'Analyze an iron condor strategy on SPY expiring in 45 days' --viz
```

### Batch queries

Create a JSON file containing an array of query strings — see `batched_queries/batch_queries.json` for a ready-to-use example covering pricing, Greeks, volatility, all supported strategies, risk management, and educational queries.

```bash
python main.py --batch batched_queries/batch_queries.json          # no charts
python main.py --batch batched_queries/batch_queries.json --viz    # with charts + consolidated dashboard
```

> ⚠️ **Token/cost warning.** Every query in the batch makes its own independent call to `LLM_PROVIDER` (and to `LLM_JUDGE` too, for any educational/low-confidence queries in the batch) — cost and token usage scale linearly with the number of queries in the file. A 30-query batch means ~30 core-generation calls plus however many judge calls get triggered. Start with a small batch to sanity-check before running a large one.

Output locations:
- Per-query results: `outputs/batch_results_<timestamp>.json`
- Consolidated dashboard (only with `--viz`): `visualizations/batch_results/batch_dashboard_<timestamp>.html`
