from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger

class OptionsVisualization:
    """Comprehensive visualization components for options analysis"""
    
    def __init__(self):
        self.theme_colors = {
            "primary": "#1f77b4",
            "secondary": "#ff7f0e", 
            "success": "#2ca02c",
            "danger": "#d62728",
            "warning": "#ff9500",
            "info": "#17a2b8",
            "background": "#f8f9fa",
            "grid": "#e0e0e0"
        }
        
        # Set default template
        self.default_template = {
            "layout": {
                "font": {"family": "Arial, sans-serif", "size": 12},
                "plot_bgcolor": "white",
                "paper_bgcolor": "white",
                "colorway": list(self.theme_colors.values())
            }
        }
    
    def create_payoff_diagram(
        self, 
        viz_data: Dict[str, Any],
        title: str = "Option Payoff Diagram"
    ) -> go.Figure:
        """Create option payoff diagram"""
        try:
            if "payoff_diagram" not in viz_data:
                return self._create_empty_chart("No payoff data available")
            
            payoff_data = viz_data["payoff_diagram"]
            
            fig = go.Figure()
            
            # Add payoff line
            fig.add_trace(go.Scatter(
                x=payoff_data["x"],
                y=payoff_data["y"],
                mode='lines',
                name=f'{payoff_data["option_type"].title()} Option',
                line=dict(color=self.theme_colors["primary"], width=3),
                hovertemplate="Stock Price: $%{x:.2f}<br>Profit/Loss: $%{y:.2f}<extra></extra>"
            ))
            
            # Add zero line
            fig.add_hline(
                y=0, 
                line_dash="dash", 
                line_color="gray",
                annotation_text="Breakeven"
            )
            
            # Add current price line
            fig.add_vline(
                x=payoff_data["current_price"],
                line_dash="dot",
                line_color=self.theme_colors["info"],
                annotation_text=f"Current: ${payoff_data['current_price']:.2f}"
            )
            
            # Add strike price line
            fig.add_vline(
                x=payoff_data["strike_price"],
                line_dash="dot",
                line_color=self.theme_colors["warning"],
                annotation_text=f"Strike: ${payoff_data['strike_price']:.2f}"
            )
            
            # Add breakeven point
            if "breakeven" in payoff_data:
                fig.add_vline(
                    x=payoff_data["breakeven"],
                    line_dash="solid",
                    line_color=self.theme_colors["success"],
                    annotation_text=f"B/E: ${payoff_data['breakeven']:.2f}"
                )
            
            # Update layout
            fig.update_layout(
                title={
                    "text": title,
                    "x": 0.5,
                    "font": {"size": 16, "color": "black"}
                },
                xaxis_title="Stock Price at Expiration ($)",
                yaxis_title="Profit/Loss ($)",
                showlegend=True,
                template="plotly_white",
                height=500,
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            # Add grid
            fig.update_xaxes(showgrid=True, gridcolor=self.theme_colors["grid"])
            fig.update_yaxes(showgrid=True, gridcolor=self.theme_colors["grid"])
            
            return fig
            
        except Exception as e:
            logger.error(f"Payoff diagram creation failed: {e}")
            return self._create_empty_chart(f"Error creating payoff diagram: {str(e)}")
    
    def create_greeks_radar_chart(
        self, 
        viz_data: Dict[str, Any],
        title: str = "Option Greeks Analysis"
    ) -> go.Figure:
        """Create radar chart for option Greeks"""
        try:
            if "greeks_chart" not in viz_data:
                return self._create_empty_chart("No Greeks data available")
            
            greeks_data = viz_data["greeks_chart"]["data"]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=greeks_data["values"],
                theta=greeks_data["labels"],
                fill='toself',
                name='Greeks',
                line_color=self.theme_colors["primary"],
                fillcolor=f'rgba(31, 119, 180, 0.3)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(greeks_data["values"]) * 1.1]
                    )
                ),
                title={
                    "text": title,
                    "x": 0.5,
                    "font": {"size": 16, "color": "black"}
                },
                template="plotly_white",
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Greeks radar chart creation failed: {e}")
            return self._create_empty_chart(f"Error creating Greeks chart: {str(e)}")
    
    def create_volatility_comparison_chart(
        self, 
        viz_data: Dict[str, Any],
        title: str = "Volatility Analysis"
    ) -> go.Figure:
        """Create volatility comparison bar chart"""
        try:
            if "volatility_chart" not in viz_data:
                return self._create_empty_chart("No volatility data available")
            
            vol_data = viz_data["volatility_chart"]["data"]
            
            fig = go.Figure()
            
            # Create bar chart
            fig.add_trace(go.Bar(
                x=vol_data["categories"],
                y=vol_data["values"],
                marker_color=[
                    self.theme_colors["primary"],
                    self.theme_colors["secondary"],
                    self.theme_colors["success"]
                ][:len(vol_data["categories"])],
                text=[f"{v:.1f}%" for v in vol_data["values"]],
                textposition='auto',
                hovertemplate="<b>%{x}</b><br>Volatility: %{y:.1f}%<extra></extra>"
            ))
            
            fig.update_layout(
                title={
                    "text": title,
                    "x": 0.5,
                    "font": {"size": 16, "color": "black"}
                },
                xaxis_title="Volatility Type",
                yaxis_title="Volatility (%)",
                template="plotly_white",
                height=400,
                showlegend=False
            )
            
            # Add grid
            fig.update_yaxes(showgrid=True, gridcolor=self.theme_colors["grid"])
            
            return fig
            
        except Exception as e:
            logger.error(f"Volatility chart creation failed: {e}")
            return self._create_empty_chart(f"Error creating volatility chart: {str(e)}")
    
    def create_volatility_surface(
        self,
        viz_data: Dict[str, Any],
        title: str = "Implied Volatility Surface"
    ) -> go.Figure:
        """Create a true 3D implied volatility surface (moneyness x days-to-expiry x IV),
        interpolated from options data spanning multiple expirations."""
        try:
            surface_data = viz_data.get("volatility_surface")
            if not surface_data or not surface_data.get("available"):
                reason = (surface_data or {}).get("reason", "No volatility surface data available")
                return self._create_empty_chart(reason)

            fig = go.Figure(data=[go.Surface(
                x=surface_data["x"],
                y=surface_data["y"],
                z=surface_data["z"],
                colorscale="Viridis",
                colorbar=dict(title="IV (%)"),
                hovertemplate="Moneyness: %{x:.2f}<br>Days to Expiry: %{y:.0f}<br>IV: %{z:.1f}%<extra></extra>"
            )])

            fig.update_layout(
                title={
                    "text": title,
                    "x": 0.5,
                    "font": {"size": 16, "color": "black"}
                },
                scene=dict(
                    xaxis_title=surface_data.get("xlabel", "Moneyness"),
                    yaxis_title=surface_data.get("ylabel", "Days to Expiry"),
                    zaxis_title=surface_data.get("zlabel", "Implied Volatility (%)")
                ),
                template="plotly_white",
                height=550
            )

            return fig

        except Exception as e:
            logger.error(f"Volatility surface creation failed: {e}")
            return self._create_empty_chart(f"Error creating volatility surface: {str(e)}")

    def create_iv_term_structure_chart(
        self,
        viz_data: Dict[str, Any],
        title: str = "IV Term Structure"
    ) -> go.Figure:
        """Create an ATM implied volatility term structure chart (ATM IV by expiry)"""
        try:
            surface_data = viz_data.get("volatility_surface") or {}
            term_structure = surface_data.get("term_structure")
            if not term_structure:
                return self._create_empty_chart("No IV term structure data available")

            days = [t["days_to_expiry"] for t in term_structure]
            atm_iv = [t["atm_implied_vol"] * 100 for t in term_structure]

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=days,
                y=atm_iv,
                mode='lines+markers',
                name='ATM Implied Volatility',
                line=dict(color=self.theme_colors["primary"], width=3),
                marker=dict(size=8),
                hovertemplate="Days to Expiry: %{x}<br>ATM IV: %{y:.1f}%<extra></extra>"
            ))

            fig.update_layout(
                title={
                    "text": title,
                    "x": 0.5,
                    "font": {"size": 16, "color": "black"}
                },
                xaxis_title="Days to Expiry",
                yaxis_title="ATM Implied Volatility (%)",
                template="plotly_white",
                height=400
            )

            fig.update_xaxes(showgrid=True, gridcolor=self.theme_colors["grid"])
            fig.update_yaxes(showgrid=True, gridcolor=self.theme_colors["grid"])

            return fig

        except Exception as e:
            logger.error(f"IV term structure chart creation failed: {e}")
            return self._create_empty_chart(f"Error creating IV term structure chart: {str(e)}")


    def create_monte_carlo_convergence_chart(
        self, 
        convergence_data: Dict[str, Any],
        title: str = "Monte Carlo Convergence Analysis"
    ) -> go.Figure:
        """Create Monte Carlo convergence visualization"""
        try:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Option Price Convergence", "Standard Error"),
                vertical_spacing=0.1
            )
            
            # Price convergence
            fig.add_trace(
                go.Scatter(
                    x=convergence_data["simulation_counts"], 
                    y=convergence_data["option_prices"],
                    mode='lines',
                    name='Option Price',
                    line=dict(color=self.theme_colors["primary"], width=2)
                ),
                row=1, col=1
            )
            
            # Standard error
            fig.add_trace(
                go.Scatter(
                    x=convergence_data["simulation_counts"], 
                    y=convergence_data["standard_errors"],
                    mode='lines',
                    name='Standard Error',
                    line=dict(color=self.theme_colors["danger"], width=2)
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title={
                    "text": title,
                    "x": 0.5,
                    "font": {"size": 16, "color": "black"}
                },
                template="plotly_white",
                height=600,
                showlegend=False
            )
            
            fig.update_xaxes(title_text="Number of Simulations", row=2, col=1)
            fig.update_yaxes(title_text="Option Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Standard Error ($)", row=2, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Monte Carlo convergence chart creation failed: {e}")
            return self._create_empty_chart(f"Error creating convergence chart: {str(e)}")
    
    def create_options_chain_heatmap(
        self, 
        options_data: pd.DataFrame,
        title: str = "Options Chain Heatmap"
    ) -> go.Figure:
        """Create options chain heatmap"""
        try:
            if options_data.empty:
                return self._create_empty_chart("No options chain data available")
            
            # Prepare data for heatmap
            calls = options_data[options_data['type'] == 'call'] if 'type' in options_data.columns else pd.DataFrame()
            puts = options_data[options_data['type'] == 'put'] if 'type' in options_data.columns else pd.DataFrame()
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Calls", "Puts"),
                horizontal_spacing=0.1
            )
            
            if not calls.empty and 'strike' in calls.columns and 'volume' in calls.columns:
                fig.add_trace(
                    go.Bar(
                        x=calls['strike'],
                        y=calls['volume'],
                        name='Call Volume',
                        marker_color=self.theme_colors["success"]
                    ),
                    row=1, col=1
                )
            
            if not puts.empty and 'strike' in puts.columns and 'volume' in puts.columns:
                fig.add_trace(
                    go.Bar(
                        x=puts['strike'],
                        y=puts['volume'],
                        name='Put Volume',
                        marker_color=self.theme_colors["danger"]
                    ),
                    row=1, col=2
                )
            
            fig.update_layout(
                title={
                    "text": title,
                    "x": 0.5,
                    "font": {"size": 16, "color": "black"}
                },
                template="plotly_white",
                height=500,
                showlegend=False
            )
            
            fig.update_xaxes(title_text="Strike Price ($)")
            fig.update_yaxes(title_text="Volume")
            
            return fig
            
        except Exception as e:
            logger.error(f"Options chain heatmap creation failed: {e}")
            return self._create_empty_chart(f"Error creating options heatmap: {str(e)}")
    
    def create_price_history_chart(
        self, 
        price_data: pd.DataFrame,
        title: str = "Price History with Volatility"
    ) -> go.Figure:
        """Create price history with volatility bands"""
        try:
            if price_data.empty or 'Close' not in price_data.columns:
                return self._create_empty_chart("No price history data available")
            
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                subplot_titles=("Price Chart", "Rolling Volatility"),
                vertical_spacing=0.1
            )
            
            # Price chart with candlesticks if OHLC data available
            if all(col in price_data.columns for col in ['Open', 'High', 'Low', 'Close']):
                fig.add_trace(
                    go.Candlestick(
                        x=price_data.index,
                        open=price_data['Open'],
                        high=price_data['High'],
                        low=price_data['Low'],
                        close=price_data['Close'],
                        name='Price'
                    ),
                    row=1, col=1
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=price_data.index,
                        y=price_data['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color=self.theme_colors["primary"], width=2)
                    ),
                    row=1, col=1
                )
            
            # Add moving averages
            if len(price_data) >= 20:
                ma20 = price_data['Close'].rolling(window=20).mean()
                fig.add_trace(
                    go.Scatter(
                        x=price_data.index,
                        y=ma20,
                        mode='lines',
                        name='20-Day MA',
                        line=dict(color=self.theme_colors["secondary"], width=1, dash='dash')
                    ),
                    row=1, col=1
                )
            
            # Rolling volatility
            if len(price_data) >= 20:
                returns = price_data['Close'].pct_change()
                rolling_vol = returns.rolling(window=20).std() * np.sqrt(252) * 100
                
                fig.add_trace(
                    go.Scatter(
                        x=price_data.index,
                        y=rolling_vol,
                        mode='lines',
                        name='20-Day Volatility',
                        line=dict(color=self.theme_colors["warning"], width=2)
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(
                title={
                    "text": title,
                    "x": 0.5,
                    "font": {"size": 16, "color": "black"}
                },
                template="plotly_white",
                height=700,
                xaxis_rangeslider_visible=False
            )
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Price history chart creation failed: {e}")
            return self._create_empty_chart(f"Error creating price chart: {str(e)}")
    
    def create_strategy_comparison_chart(
       self, 
       strategies_data: List[Dict[str, Any]],
       title: str = "Strategy Payoff Comparison"
   ) -> go.Figure:
       """Create comparison chart for multiple option strategies"""
       try:
           if not strategies_data:
               return self._create_empty_chart("No strategy data available")
           
           fig = go.Figure()
           
           colors = [
               self.theme_colors["primary"],
               self.theme_colors["secondary"], 
               self.theme_colors["success"],
               self.theme_colors["danger"],
               self.theme_colors["warning"],
               self.theme_colors["info"]
           ]
           
           for i, strategy in enumerate(strategies_data):
               if "payoff_data" not in strategy:
                   continue
                   
               payoff_data = strategy["payoff_data"]
               color = colors[i % len(colors)]
               
               fig.add_trace(go.Scatter(
                   x=payoff_data["x"],
                   y=payoff_data["y"],
                   mode='lines',
                   name=strategy.get("name", f"Strategy {i+1}"),
                   line=dict(color=color, width=2),
                   hovertemplate=f"<b>{strategy.get('name', 'Strategy')}</b><br>" +
                                "Stock Price: $%{x:.2f}<br>P&L: $%{y:.2f}<extra></extra>"
               ))
           
           # Add zero line
           fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
           
           fig.update_layout(
               title={
                   "text": title,
                   "x": 0.5,
                   "font": {"size": 16, "color": "black"}
               },
               xaxis_title="Stock Price at Expiration ($)",
               yaxis_title="Profit/Loss ($)",
               template="plotly_white",
               height=500,
               legend=dict(
                   orientation="h",
                   yanchor="bottom",
                   y=1.02,
                   xanchor="right",
                   x=1
               )
           )
           
           # Add grid
           fig.update_xaxes(showgrid=True, gridcolor=self.theme_colors["grid"])
           fig.update_yaxes(showgrid=True, gridcolor=self.theme_colors["grid"])
           
           return fig
           
       except Exception as e:
           logger.error(f"Strategy comparison chart creation failed: {e}")
           return self._create_empty_chart(f"Error creating strategy comparison: {str(e)}")
   
    def create_risk_metrics_dashboard(
        self, 
        risk_data: Dict[str, Any],
        title: str = "Risk Metrics Dashboard"
    ) -> go.Figure:
        """Create comprehensive risk metrics dashboard"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Greeks Distribution",
                    "Risk Concentration", 
                    "Time Decay Profile",
                    "Volatility Sensitivity"
                ),
                specs=[
                    [{"type": "bar"}, {"type": "pie"}],
                    [{"type": "scatter"}, {"type": "scatter"}]
                ]
            )
            
            # Greeks distribution
            if "greeks" in risk_data:
                greeks = risk_data["greeks"]
                greek_names = list(greeks.keys())
                greek_values = [abs(v) for v in greeks.values()]
                
                fig.add_trace(
                    go.Bar(
                        x=greek_names,
                        y=greek_values,
                        name="Greeks",
                        marker_color=self.theme_colors["primary"]
                    ),
                    row=1, col=1
                )
            
            # Risk concentration pie chart
            if "risk_breakdown" in risk_data:
                risk_breakdown = risk_data["risk_breakdown"]
                fig.add_trace(
                    go.Pie(
                        labels=list(risk_breakdown.keys()),
                        values=list(risk_breakdown.values()),
                        name="Risk Types"
                    ),
                    row=1, col=2
                )
            
            # Time decay profile
            if "time_decay_profile" in risk_data:
                time_profile = risk_data["time_decay_profile"]
                fig.add_trace(
                    go.Scatter(
                        x=time_profile["days"],
                        y=time_profile["values"],
                        mode='lines+markers',
                        name="Time Decay",
                        line=dict(color=self.theme_colors["danger"])
                    ),
                    row=2, col=1
                )
            
            # Volatility sensitivity
            if "vol_sensitivity" in risk_data:
                vol_sens = risk_data["vol_sensitivity"]
                fig.add_trace(
                    go.Scatter(
                        x=vol_sens["volatility_levels"],
                        y=vol_sens["option_values"],
                        mode='lines+markers',
                        name="Vol Sensitivity",
                        line=dict(color=self.theme_colors["warning"])
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                title={
                    "text": title,
                    "x": 0.5,
                    "font": {"size": 16, "color": "black"}
                },
                template="plotly_white",
                height=800,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Risk dashboard creation failed: {e}")
            return self._create_empty_chart(f"Error creating risk dashboard: {str(e)}")
    
    def create_arbitrage_analysis_chart(
        self, 
        arbitrage_data: Dict[str, Any],
        title: str = "Arbitrage Opportunity Analysis"
    ) -> go.Figure:
        """Create arbitrage analysis visualization"""
        try:
            if not arbitrage_data or "opportunities" not in arbitrage_data:
                return self._create_empty_chart("No arbitrage data available")
            
            opportunities = arbitrage_data["opportunities"]
            
            fig = go.Figure()
            
            # Create scatter plot of opportunities
            x_vals = []
            y_vals = []
            colors = []
            sizes = []
            hover_texts = []
            
            for opp in opportunities:
                x_vals.append(opp.get("strike", 0))
                y_vals.append(opp.get("profit_potential", 0))
                colors.append(opp.get("confidence", 0.5))
                sizes.append(min(max(abs(opp.get("profit_potential", 0)) * 5, 10), 50))
                hover_texts.append(
                    f"Strike: ${opp.get('strike', 0):.2f}<br>" +
                    f"Profit: ${opp.get('profit_potential', 0):.4f}<br>" +
                    f"Type: {opp.get('type', 'Unknown')}<br>" +
                    f"Confidence: {opp.get('confidence', 0):.1%}"
                )
            
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=colors,
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Confidence"),
                    line=dict(width=1, color="DarkSlateGrey")
                ),
                text=hover_texts,
                hovertemplate="%{text}<extra></extra>",
                name="Arbitrage Opportunities"
            ))
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            
            fig.update_layout(
                title={
                    "text": title,
                    "x": 0.5,
                    "font": {"size": 16, "color": "black"}
                },
                xaxis_title="Strike Price ($)",
                yaxis_title="Profit Potential ($)",
                template="plotly_white",
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Arbitrage analysis chart creation failed: {e}")
            return self._create_empty_chart(f"Error creating arbitrage chart: {str(e)}")
        

# Add these methods to the existing OptionsVisualization class

    def create_strategy_payoff_chart(
        self, 
        viz_data: Dict[str, Any],
        title: str = "Strategy Payoff Diagram"
    ) -> go.Figure:
        """Create strategy payoff diagram with multiple components"""
        try:
            if "strategy_payoff" not in viz_data:
                return self._create_empty_chart("No strategy payoff data available")
            
            payoff_data = viz_data["strategy_payoff"]["data"]
            
            fig = go.Figure()
            
            # Main payoff line
            fig.add_trace(go.Scatter(
                x=payoff_data["x"],
                y=payoff_data["y"],
                mode='lines',
                name=payoff_data["strategy_name"],
                line=dict(color=self.theme_colors["primary"], width=4),
                hovertemplate="Stock Price: $%{x:.2f}<br>P&L: $%{y:.2f}<extra></extra>"
            ))
            
            # Add profit/loss zones
            profit_mask = np.array(payoff_data["y"]) > 0
            loss_mask = np.array(payoff_data["y"]) < 0
            
            if np.any(profit_mask):
                fig.add_trace(go.Scatter(
                    x=np.array(payoff_data["x"])[profit_mask],
                    y=np.array(payoff_data["y"])[profit_mask],
                    fill='tozeroy',
                    fillcolor='rgba(40, 167, 69, 0.3)',
                    line=dict(color='rgba(40, 167, 69, 0)'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            if np.any(loss_mask):
                fig.add_trace(go.Scatter(
                    x=np.array(payoff_data["x"])[loss_mask],
                    y=np.array(payoff_data["y"])[loss_mask],
                    fill='tozeroy',
                    fillcolor='rgba(220, 53, 69, 0.3)',
                    line=dict(color='rgba(220, 53, 69, 0)'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
            
            # Add breakeven points
            for bp in payoff_data.get("breakeven_points", []):
                fig.add_vline(
                    x=bp,
                    line_dash="dot",
                    line_color=self.theme_colors["success"],
                    annotation_text=f"B/E: ${bp:.2f}",
                    annotation_position="top"
                )
            
            # Add max profit/loss annotations
            max_profit = payoff_data.get("max_profit")
            max_loss = payoff_data.get("max_loss")
            
            if isinstance(max_profit, (int, float)):
                fig.add_annotation(
                    x=0.02, y=0.98,
                    xref="paper", yref="paper",
                    text=f"Max Profit: ${max_profit:.2f}",
                    showarrow=False,
                    bgcolor="rgba(40, 167, 69, 0.8)",
                    bordercolor="white",
                    font=dict(color="white")
                )
            else:
                fig.add_annotation(
                    x=0.02, y=0.98,
                    xref="paper", yref="paper",
                    text=f"Max Profit: {max_profit}",
                    showarrow=False,
                    bgcolor="rgba(40, 167, 69, 0.8)",
                    bordercolor="white",
                    font=dict(color="white")
                )
            
            if isinstance(max_loss, (int, float)):
                fig.add_annotation(
                    x=0.02, y=0.02,
                    xref="paper", yref="paper",
                    text=f"Max Loss: ${abs(max_loss):.2f}",
                    showarrow=False,
                    bgcolor="rgba(220, 53, 69, 0.8)",
                    bordercolor="white",
                    font=dict(color="white")
                )
            else:
                fig.add_annotation(
                    x=0.02, y=0.02,
                    xref="paper", yref="paper",
                    text=f"Max Loss: {max_loss}",
                    showarrow=False,
                    bgcolor="rgba(220, 53, 69, 0.8)",
                    bordercolor="white",
                    font=dict(color="white")
                )
            
            fig.update_layout(
                title={
                    "text": title,
                    "x": 0.5,
                    "font": {"size": 16, "color": "black"}
                },
                xaxis_title="Stock Price at Expiration ($)",
                yaxis_title="Profit/Loss ($)",
                template="plotly_white",
                height=500,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Strategy payoff chart creation failed: {e}")
            return self._create_empty_chart(f"Error creating strategy chart: {str(e)}")
    
    def create_risk_dashboard(
        self, 
        viz_data: Dict[str, Any],
        title: str = "Portfolio Risk Dashboard"
    ) -> go.Figure:
        """Create comprehensive risk dashboard"""
        try:
            if "risk_dashboard" not in viz_data:
                return self._create_empty_chart("No risk dashboard data available")
            
            risk_data = viz_data["risk_dashboard"]["data"]
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Value at Risk",
                    "Portfolio Greeks",
                    "Position Concentration",
                    "Risk Metrics Summary"
                ),
                specs=[
                    [{"type": "bar"}, {"type": "bar"}],
                    [{"type": "pie"}, {"type": "indicator"}]
                ]
            )
            
            # VaR Chart
            var_data = risk_data.get("var_data", {})
            if var_data:
                fig.add_trace(
                    go.Bar(
                        x=var_data["labels"],
                        y=var_data["values"],
                        name="VaR",
                        marker_color=[self.theme_colors["danger"], self.theme_colors["warning"]]
                    ),
                    row=1, col=1
                )
            
            # Greeks Chart
            greeks_data = risk_data.get("greeks_data", {})
            if greeks_data:
                colors = [
                    self.theme_colors["primary"],
                    self.theme_colors["secondary"],
                    self.theme_colors["danger"],
                    self.theme_colors["success"]
                ]
                
                fig.add_trace(
                    go.Bar(
                        x=greeks_data["labels"],
                        y=greeks_data["values"],
                        name="Greeks",
                        marker_color=colors
                    ),
                    row=1, col=2
                )
            
            # Concentration Pie Chart
            concentration = risk_data.get("concentration", {})
            if concentration:
                # Show top 5 concentrations
                sorted_conc = sorted(concentration.items(), key=lambda x: x[1], reverse=True)[:5]
                labels, values = zip(*sorted_conc) if sorted_conc else ([], [])
                
                fig.add_trace(
                    go.Pie(
                        labels=labels,
                        values=values,
                        name="Concentration"
                    ),
                    row=2, col=1
                )
            
            # Risk Summary Indicator
            total_var = sum(var_data.get("values", [0])) if var_data else 0
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=total_var,
                    title={"text": "Total VaR ($)"},
                    gauge={
                        "axis": {"range": [None, total_var * 2]},
                        "bar": {"color": self.theme_colors["primary"]},
                        "steps": [
                            {"range": [0, total_var * 0.5], "color": "lightgray"},
                            {"range": [total_var * 0.5, total_var * 1.5], "color": "gray"}
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": total_var * 1.2
                        }
                    }
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title={
                    "text": title,
                    "x": 0.5,
                    "font": {"size": 16, "color": "black"}
                },
                template="plotly_white",
                height=700,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Risk dashboard creation failed: {e}")
            return self._create_empty_chart(f"Error creating risk dashboard: {str(e)}")
    
    def create_stress_test_chart(
        self, 
        viz_data: Dict[str, Any],
        title: str = "Stress Test Results"
    ) -> go.Figure:
        """Create stress test results visualization"""
        try:
            if "stress_tests" not in viz_data:
                return self._create_empty_chart("No stress test data available")
            
            stress_data = viz_data["stress_tests"]["data"]
            scenarios = stress_data["scenarios"]
            pnl_impact = stress_data["pnl_impact"]
            
            # Clean up scenario names
            clean_scenarios = [scenario.replace("_", " ").title() for scenario in scenarios]
            
            # Color code based on positive/negative impact
            colors = [self.theme_colors["success"] if pnl >= 0 else self.theme_colors["danger"] for pnl in pnl_impact]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=clean_scenarios,
                y=pnl_impact,
                marker_color=colors,
                text=[f"${pnl:,.0f}" for pnl in pnl_impact],
                textposition='auto',
                hovertemplate="<b>%{x}</b><br>P&L Impact: $%{y:,.0f}<extra></extra>"
            ))
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            
            fig.update_layout(
                title={
                    "text": title,
                    "x": 0.5,
                    "font": {"size": 16, "color": "black"}
                },
                xaxis_title="Stress Scenario",
                yaxis_title="P&L Impact ($)",
                template="plotly_white",
                height=500,
                showlegend=False
            )
            
            # Rotate x-axis labels if needed
            fig.update_xaxes(tickangle=45)
            
            return fig
            
        except Exception as e:
            logger.error(f"Stress test chart creation failed: {e}")
            return self._create_empty_chart(f"Error creating stress test chart: {str(e)}")
    
    def create_comprehensive_strategy_dashboard(
        self, 
        strategy_data: Dict[str, Any]
    ) -> Dict[str, go.Figure]:
        """Create comprehensive strategy analysis dashboard"""
        try:
            dashboard = {}
            
            # Strategy payoff diagram
            if "strategy_payoff" in strategy_data:
                dashboard["payoff"] = self.create_strategy_payoff_chart(
                    strategy_data,
                    "Strategy Payoff Analysis"
                )
            
            # Greeks radar chart
            if "strategy_greeks" in strategy_data:
                dashboard["greeks"] = self.create_greeks_radar_chart(
                    strategy_data,
                    "Strategy Risk Profile"
                )
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Strategy dashboard creation failed: {e}")
            return {"error": self._create_empty_chart(f"Dashboard error: {str(e)}")}
    
    def create_comprehensive_risk_dashboard(
        self, 
        risk_data: Dict[str, Any]
    ) -> Dict[str, go.Figure]:
        """Create comprehensive risk management dashboard"""
        try:
            dashboard = {}
            
            # Risk metrics dashboard
            if "risk_dashboard" in risk_data:
                dashboard["risk_overview"] = self.create_risk_dashboard(
                    risk_data,
                    "Portfolio Risk Overview"
                )
            
            # Stress test results
            if "stress_tests" in risk_data:
                dashboard["stress_tests"] = self.create_stress_test_chart(
                    risk_data,
                    "Stress Test Analysis"
                )
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Risk dashboard creation failed: {e}")
            return {"error": self._create_empty_chart(f"Dashboard error: {str(e)}")}


    def create_comprehensive_analysis_dashboard(
        self, 
        analysis_data: Dict[str, Any]
    ) -> Dict[str, go.Figure]:
        """Create comprehensive dashboard with multiple charts"""
        try:
            dashboard = {}
            
            # Payoff diagram
            if "payoff_diagram" in analysis_data:
                dashboard["payoff"] = self.create_payoff_diagram(
                    analysis_data, 
                    "Option Strategy Payoff Analysis"
                )
            
            # Greeks radar chart
            if "greeks_chart" in analysis_data:
                dashboard["greeks"] = self.create_greeks_radar_chart(
                    analysis_data,
                    "Risk Profile (Greeks)"
                )
            
            # Volatility analysis
            if "volatility_chart" in analysis_data:
                dashboard["volatility"] = self.create_volatility_comparison_chart(
                    analysis_data,
                    "Volatility Environment"
                )
            
            if "payoff_diagram" in analysis_data:
                dashboard["payoff"] = self.create_payoff_diagram(
                    analysis_data, 
                    "Option Strategy Payoff Analysis"
                )
            
            if "greeks_chart" in analysis_data:
                dashboard["greeks"] = self.create_greeks_radar_chart(
                    analysis_data,
                    "Risk Profile (Greeks)"
                )
            
            if "volatility_chart" in analysis_data:
                dashboard["volatility"] = self.create_volatility_comparison_chart(
                    analysis_data,
                    "Volatility Environment"
                )
            
            # New strategy charts
            if "strategy_payoff" in analysis_data:
                dashboard["strategy_payoff"] = self.create_strategy_payoff_chart(
                    analysis_data,
                    "Strategy Payoff Analysis"
                )
            
            # New risk management charts
            if "risk_dashboard" in analysis_data:
                dashboard["risk_dashboard"] = self.create_risk_dashboard(
                    analysis_data,
                    "Portfolio Risk Analysis"
                )
            
            if "stress_tests" in analysis_data:
                dashboard["stress_tests"] = self.create_stress_test_chart(
                    analysis_data,
                    "Stress Test Results"
                )

            # Monte Carlo convergence
            if "convergence_data" in analysis_data:
                dashboard["convergence"] = self.create_monte_carlo_convergence_chart(
                    analysis_data["convergence_data"],
                    "Simulation Convergence"
                )
            
            # Price history
            if "price_data" in analysis_data:
                dashboard["price_history"] = self.create_price_history_chart(
                    analysis_data["price_data"],
                    "Historical Price & Volatility"
                )
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Dashboard creation failed: {e}")
            return {"error": self._create_empty_chart(f"Dashboard error: {str(e)}")}
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create empty chart with message"""
        fig = go.Figure()
        
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        
        fig.update_layout(
            template="plotly_white",
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        
        return fig
    
    def export_chart_to_html(self, fig: go.Figure, filename: str) -> str:
        """Export chart to HTML file"""
        try:
            html_string = fig.to_html(
                include_plotlyjs=True,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
                }
            )
            
            with open(filename, 'w') as f:
                f.write(html_string)
            
            logger.info(f"Chart exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Chart export failed: {e}")
            return ""
    
    def save_dashboard_as_html(
        self, 
        dashboard: Dict[str, go.Figure], 
        filename: str = "options_dashboard.html"
    ) -> str:
        """Save entire dashboard as HTML file"""
        try:
            html_parts = [
                """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Options Analysis Dashboard</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 0; padding: 20px 0; }
                        .dashboard-title { text-align: center; color: #333; margin-bottom: 30px; }
                        .chart-grid { display: flex; flex-direction: column; gap: 30px; max-width: 1400px; margin: 0 auto; }
                        .chart-container { width: 100%; }
                    </style>
                </head>
                <body>
                    <h1 class="dashboard-title">Options Trading Analysis Dashboard</h1>
                """
            ]
            
            # Add timestamp
            html_parts.append(f'<p style="text-align: center; color: #666;">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>')
            
            # Add charts
            html_parts.append('<div class="chart-grid">')
            
            for i, (chart_name, fig) in enumerate(dashboard.items()):
                # full_html=False is required here: without it, each chart
                # renders as a complete standalone <html><head><body> document
                # (Plotly's default), and concatenating several of those into
                # one page produces invalid, nested <html>/<body> tags.
                #
                # include_plotlyjs='cdn' on the first chart only (then False
                # for the rest) makes Plotly emit a version-pinned CDN URL
                # matching the installed plotly package, instead of a
                # hardcoded "plotly-latest.min.js" alias. That alias can lag
                # behind the Plotly.js version some traces are serialized
                # for — e.g. numpy-array trace data (used by the payoff
                # chart's profit/loss fill regions) is encoded as a compact
                # binary payload that only newer Plotly.js versions can
                # decode, so a stale CDN build silently drops just those
                # traces while plain-list traces still render.
                chart_html = fig.to_html(
                    include_plotlyjs='cdn' if i == 0 else False,
                    full_html=False,
                    div_id=f"chart_{chart_name}",
                    config={'displayModeBar': True, 'displaylogo': False}
                )
                
                html_parts.append(f'<div class="chart-container">')
                html_parts.append(chart_html)
                html_parts.append('</div>')
            
            html_parts.append('</div>')
            html_parts.append('</body></html>')
            
            # Write to file
            with open(filename, 'w') as f:
                f.write('\n'.join(html_parts))
            
            logger.info(f"Dashboard saved to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Dashboard export failed: {e}")
            return ""

    # Usage example and test
if __name__ == "__main__":
    # Test visualization components
    viz = OptionsVisualization()
    
    # Sample data for testing
    test_viz_data = {
        "payoff_diagram": {
            "x": list(range(80, 121)),
            "y": [max(0, x - 100) - 5 for x in range(80, 121)],  # Call option payoff
            "current_price": 100,
            "strike_price": 100,
            "option_type": "call",
            "breakeven": 105
        },
        "greeks_chart": {
            "data": {
                "labels": ["Delta", "Gamma", "Theta", "Vega", "Rho"],
                "values": [0.6, 0.5, 0.3, 0.8, 0.4]
            }
        },
        "volatility_chart": {
            "data": {
                "categories": ["30-Day Historical", "90-Day Historical", "Implied Vol"],
                "values": [25.5, 28.2, 31.8]
            }
        }
    }
    
    print("Testing Options Visualization Components")
    print("="*50)
    
    # Test payoff diagram
    payoff_fig = viz.create_payoff_diagram(test_viz_data)
    print("✅ Payoff diagram created")
    
    # Test Greeks radar chart
    greeks_fig = viz.create_greeks_radar_chart(test_viz_data)
    print("✅ Greeks radar chart created")
    
    # Test volatility chart
    vol_fig = viz.create_volatility_comparison_chart(test_viz_data)
    print("✅ Volatility comparison chart created")
    
    # Create comprehensive dashboard
    dashboard = viz.create_comprehensive_analysis_dashboard(test_viz_data)
    print(f"✅ Dashboard created with {len(dashboard)} charts")
    
    # Export dashboard
    filename = viz.save_dashboard_as_html(dashboard, "test_dashboard.html")
    if filename:
        print(f"✅ Dashboard exported to {filename}")
    else:
        print("❌ Dashboard export failed")
    
    print("\nVisualization components test completed!")