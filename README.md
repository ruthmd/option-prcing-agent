# Options Pricing Agent

An agentic AI system for options trading analysis with built-in guardrails and grounding mechanisms.

## Features

- 🤖 **Multi-Agent Architecture**: Specialized agents for different analysis types
- 🛡️ **Built-in Guardrails**: Prevents hallucinations and ensures domain-specific responses
- 📊 **Real-time Market Data**: Integration with yfinance for live market data
- 🧠 **Knowledge Grounding**: FAISS-based knowledge retrieval from established options theory
- ✅ **Output Validation**: Comprehensive validation of all calculations and results
- 📈 **Comprehensive Analysis**: Option pricing, Greeks, volatility, strategies, and more

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd options-ai-agent

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env

# Run the application
python main.py