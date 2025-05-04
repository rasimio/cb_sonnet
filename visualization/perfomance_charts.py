"""
Performance Visualization Module for TensorTrade

This module provides visualization tools for backtest and live trading results.
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import logging

from matplotlib import pyplot as plt

# Import plotly if available
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not installed. Some visualization features will be disabled.")

# Configure logging
logger = logging.getLogger(__name__)


def plot_backtest_results(results: Dict[str, Any], output_path: Optional[str] = None,
                          show_plot: bool = True) -> None:
    """
    Plot backtest results using Plotly

    Args:
        results: Dictionary with backtest results
        output_path: Path to save the output HTML file
        show_plot: Whether to show the plot in the notebook/browser
    """
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly is not available. Install with: pip install plotly")
        return None

    # Extract data from results
    equity_curve = results['equity_curve']
    position_history = results['position_history']
    trades = results['trades']
    metrics = results['metrics']
    price_data = results['data']

    # Create figure with subplots - using 4 rows instead of 3 to separate drawdown
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        subplot_titles=("Price and Trades", "Equity Curve", "Drawdown", "Position Size")
    )

    # Plot price
    fig.add_trace(
        go.Candlestick(
            x=price_data.index,
            open=price_data['open'],
            high=price_data['high'],
            low=price_data['low'],
            close=price_data['close'],
            name="Price"
        ),
        row=1, col=1
    )

    # Plot trades with clear flags
    for trade in trades:
        # Determine marker color and shape based on trade type and profit/loss
        if trade['type'] == 'long':
            marker_color = 'green' if trade['pnl'] > 0 else 'red'
            # Use flag shapes instead of triangles
            entry_marker = 'triangle-up'
            exit_marker = 'triangle-down'
        else:  # short
            marker_color = 'red' if trade['pnl'] > 0 else 'green'
            entry_marker = 'triangle-down'
            exit_marker = 'triangle-up'

        # Plot entry point with a flag
        fig.add_trace(
            go.Scatter(
                x=[trade['entry_date']],
                y=[trade['entry_price'] * 1.005],  # Slight offset to make flags more visible
                mode='markers+text',
                marker=dict(
                    symbol=entry_marker,
                    size=12,
                    color=marker_color,
                    line=dict(width=2, color='black')
                ),
                text=['BUY' if trade['type'] == 'long' else 'SELL'],
                textposition='top center',
                name=f"{trade['type'].capitalize()} Entry",
                showlegend=False
            ),
            row=1, col=1
        )

        # Plot exit point with a flag
        fig.add_trace(
            go.Scatter(
                x=[trade['exit_date']],
                y=[trade['exit_price'] * 0.995],  # Slight offset downward for visibility
                mode='markers+text',
                marker=dict(
                    symbol=exit_marker,
                    size=12,
                    color=marker_color,
                    line=dict(width=2, color='black')
                ),
                text=['EXIT'],
                textposition='bottom center',
                name=f"{trade['type'].capitalize()} Exit",
                showlegend=False
            ),
            row=1, col=1
        )

        # Connect entry to exit with a line
        fig.add_trace(
            go.Scatter(
                x=[trade['entry_date'], trade['exit_date']],
                y=[trade['entry_price'], trade['exit_price']],
                mode='lines',
                line=dict(
                    color=marker_color,
                    width=1,
                    dash='dot'
                ),
                showlegend=False
            ),
            row=1, col=1
        )

    # Plot equity curve (now separate from drawdown)
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode='lines',
            name="Equity",
            line=dict(color='blue', width=2),
            fill='tozeroy'
        ),
        row=2, col=1
    )

    # Plot drawdown in its own subplot
    equity_series = pd.Series(equity_curve.values, index=equity_curve.index)
    peak = equity_series.cummax()
    drawdown = -((equity_series - peak) / peak) * 100

    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            name="Drawdown %",
            line=dict(color='red', width=2),
            fill='tozeroy'
        ),
        row=3, col=1
    )

    # Plot position history
    position_colors = position_history.map({1: 'green', -1: 'red', 0: 'gray'})

    fig.add_trace(
        go.Scatter(
            x=position_history.index,
            y=position_history.values,
            mode='lines',
            name="Position",
            line=dict(color='purple', width=2)
        ),
        row=4, col=1
    )

    # Add performance metrics as annotations
    metrics_text = (
        f"Total Return: {metrics['total_return_pct']:.2f}%<br>"
        f"Annualized Return: {metrics['annualized_return_pct']:.2f}%<br>"
        f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}<br>"
        f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%<br>"
        f"Win Rate: {metrics['win_rate_pct']:.2f}%<br>"
        f"Profit Factor: {metrics['profit_factor']:.2f}<br>"
        f"Total Trades: {metrics['total_trades']}"
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.99,
        text=metrics_text,
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        align="left"
    )

    # Update layout
    fig.update_layout(
        title="Trading Backtest Results",
        xaxis_rangeslider_visible=False,
        height=1000,  # Increased height to accommodate 4 subplots
        width=1200,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Update y-axis titles
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Equity", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=3, col=1)
    fig.update_yaxes(title_text="Position", row=4, col=1)

    # Save to file if requested
    if output_path:
        fig.write_html(output_path)
        logger.info(f"Plot saved to {output_path}")

    # Show plot if requested
    if show_plot:
        fig.show()

    return fig


def create_performance_dashboard(results: Dict[str, Any], output_dir: str) -> str:
    """
    Create a comprehensive performance dashboard with multiple charts

    Args:
        results: Dictionary with backtest results
        output_dir: Directory to save the dashboard files

    Returns:
        Path to the main dashboard HTML file
    """
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly is not available. Install with: pip install plotly")
        return None

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate individual charts
    plot_backtest_results(results, os.path.join(output_dir, "backtest_results.html"), False)
    plot_trade_distribution(results, os.path.join(output_dir, "trade_distribution.html"), False)
    plot_monthly_returns(results, os.path.join(output_dir, "monthly_returns.html"), False)

    # Get the template file path
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    os.makedirs(template_dir, exist_ok=True)
    template_path = os.path.join(template_dir, 'dashboard_template.html')

    # If template file doesn't exist, create it
    if not os.path.exists(template_path):
        create_dashboard_template(template_path)

    try:
        # Read the template file
        with open(template_path, 'r') as f:
            dashboard_html = f.read()

        # Replace template variables with actual values
        dashboard_html = dashboard_html.replace('{{START_DATE}}', results['equity_curve'].index[0].strftime('%Y-%m-%d'))
        dashboard_html = dashboard_html.replace('{{END_DATE}}', results['equity_curve'].index[-1].strftime('%Y-%m-%d'))
        dashboard_html = dashboard_html.replace('{{TOTAL_RETURN}}', f"{results['metrics']['total_return_pct']:.2f}")
        dashboard_html = dashboard_html.replace('{{ANNUAL_RETURN}}',
                                                f"{results['metrics']['annualized_return_pct']:.2f}")
        dashboard_html = dashboard_html.replace('{{SHARPE_RATIO}}', f"{results['metrics']['sharpe_ratio']:.2f}")
        dashboard_html = dashboard_html.replace('{{MAX_DRAWDOWN}}', f"{results['metrics']['max_drawdown_pct']:.2f}")
        dashboard_html = dashboard_html.replace('{{WIN_RATE}}', f"{results['metrics']['win_rate_pct']:.2f}")
        dashboard_html = dashboard_html.replace('{{PROFIT_FACTOR}}', f"{results['metrics']['profit_factor']:.2f}")
        dashboard_html = dashboard_html.replace('{{TOTAL_TRADES}}', f"{results['metrics']['total_trades']}")
        dashboard_html = dashboard_html.replace('{{FINAL_EQUITY}}', f"{results['metrics']['final_equity']:.2f}")

        # Write dashboard HTML to file
        dashboard_path = os.path.join(output_dir, "dashboard.html")
        with open(dashboard_path, 'w') as f:
            f.write(dashboard_html)

        logger.info(f"Performance dashboard created at {dashboard_path}")
        return dashboard_path

    except Exception as e:
        logger.error(f"Error creating dashboard: {str(e)}")
        return None


def create_dashboard_template(template_path: str) -> None:
    """
    Create the dashboard HTML template file

    Args:
        template_path: Path to save the template file
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(template_path), exist_ok=True)

    # Load template from a separate file if it exists, otherwise use a default template
    template_content = """<!DOCTYPE html>
<html>
<head>
    <title>Trading Performance Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
            margin-bottom: 20px;
        }
        .metrics-panel {
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
        }
        .metric-box {
            width: 23%;
            padding: 15px;
            text-align: center;
            background-color: #f9f9f9;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-label {
            font-size: 14px;
            color: #7f8c8d;
        }
        .chart-container {
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
        }
        .chart-title {
            font-size: 18px;
            margin-bottom: 10px;
        }
        iframe {
            width: 100%;
            height: 600px;
            border: none;
        }
        .tabs {
            display: flex;
            margin-bottom: 20px;
        }
        .tab {
            padding: 10px 20px;
            background-color: #ddd;
            border-radius: 5px 5px 0 0;
            cursor: pointer;
            margin-right: 5px;
        }
        .tab.active {
            background-color: white;
            box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Trading Performance Dashboard</h1>
        <p>Backtest from {{START_DATE}} to {{END_DATE}}</p>
    </div>

    <div class="container">
        <div class="metrics-panel">
            <div class="metric-box">
                <div class="metric-label">Total Return</div>
                <div class="metric-value">{{TOTAL_RETURN}}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Annualized Return</div>
                <div class="metric-value">{{ANNUAL_RETURN}}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">{{SHARPE_RATIO}}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value">{{MAX_DRAWDOWN}}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{{WIN_RATE}}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Profit Factor</div>
                <div class="metric-value">{{PROFIT_FACTOR}}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value">{{TOTAL_TRADES}}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Final Equity</div>
                <div class="metric-value">${{FINAL_EQUITY}}</div>
            </div>
        </div>

        <div class="tabs">
            <div class="tab active" onclick="showTab('overview')">Overview</div>
            <div class="tab" onclick="showTab('trades')">Trade Analysis</div>
            <div class="tab" onclick="showTab('monthly')">Monthly Returns</div>
        </div>

        <div id="overview" class="tab-content active">
            <div class="chart-container">
                <div class="chart-title">Backtest Results</div>
                <iframe src="backtest_results.html"></iframe>
            </div>
        </div>

        <div id="trades" class="tab-content">
            <div class="chart-container">
                <div class="chart-title">Trade Distribution Analysis</div>
                <iframe src="trade_distribution.html"></iframe>
            </div>
        </div>

        <div id="monthly" class="tab-content">
            <div class="chart-container">
                <div class="chart-title">Monthly Returns</div>
                <iframe src="monthly_returns.html"></iframe>
            </div>
        </div>
    </div>

    <script>
        function showTab(tabId) {
            // Hide all tab contents
            var tabContents = document.getElementsByClassName('tab-content');
            for (var i = 0; i < tabContents.length; i++) {
                tabContents[i].classList.remove('active');
            }

            // Deactivate all tabs
            var tabs = document.getElementsByClassName('tab');
            for (var i = 0; i < tabs.length; i++) {
                tabs[i].classList.remove('active');
            }

            // Activate the selected tab and its content
            document.getElementById(tabId).classList.add('active');
            var selectedTab = document.querySelector('.tab[onclick="showTab(\\''+tabId+'\\')"]');
            selectedTab.classList.add('active');
        }
    </script>
</body>
</html>
"""

    # Write template to file
    with open(template_path, 'w') as f:
        f.write(template_content)

    logger.info(f"Dashboard template created at {template_path}")


# Keep the rest of the functions from the original file
def plot_trade_distribution(results: Dict[str, Any], output_path: Optional[str] = None,
                            show_plot: bool = True) -> None:
    """
    Plot trade distribution statistics

    Args:
        results: Dictionary with backtest results
        output_path: Path to save the output HTML file
        show_plot: Whether to show the plot
    """
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly is not available. Install with: pip install plotly")
        return None

    trades = results['trades']
    metrics = results['metrics']

    # Convert trades to DataFrame
    trades_df = pd.DataFrame(trades)

    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Trade PnL Distribution",
            "Cumulative PnL",
            "Trade Duration Distribution",
            "Win/Loss by Trade Type"
        )
    )

    # Plot 1: Trade PnL Distribution
    fig.add_trace(
        go.Histogram(
            x=trades_df['pnl'],
            name="PnL Distribution",
            marker_color='blue',
            opacity=0.7,
            nbinsx=20
        ),
        row=1, col=1
    )

    # Plot 2: Cumulative PnL
    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()

    fig.add_trace(
        go.Scatter(
            x=trades_df.index,
            y=trades_df['cumulative_pnl'],
            mode='lines+markers',
            name="Cumulative PnL",
            line=dict(color='green', width=2)
        ),
        row=1, col=2
    )

    # Plot 3: Trade Duration
    trades_df['duration'] = (trades_df['exit_date'] - trades_df['entry_date']).dt.total_seconds() / 3600  # hours

    fig.add_trace(
        go.Histogram(
            x=trades_df['duration'],
            name="Duration (hours)",
            marker_color='orange',
            opacity=0.7,
            nbinsx=20
        ),
        row=2, col=1
    )

    # Plot 4: Win/Loss by Trade Type
    # Calculate win counts by trade type
    trade_results = pd.crosstab(
        trades_df['type'],
        trades_df['pnl'] > 0,
        rownames=['Trade Type'],
        colnames=['Result'],
        values=trades_df['pnl'],
        aggfunc='count'
    ).fillna(0)

    if True in trade_results.columns and False in trade_results.columns:
        win_counts = trade_results[True].values
        loss_counts = trade_results[False].values
    elif True in trade_results.columns:
        win_counts = trade_results[True].values
        loss_counts = [0] * len(trade_results.index)
    elif False in trade_results.columns:
        win_counts = [0] * len(trade_results.index)
        loss_counts = trade_results[False].values
    else:
        win_counts = []
        loss_counts = []

    trade_types = list(trade_results.index)

    fig.add_trace(
        go.Bar(
            x=trade_types,
            y=win_counts,
            name="Wins",
            marker_color='green',
            opacity=0.7
        ),
        row=2, col=2
    )

    fig.add_trace(
        go.Bar(
            x=trade_types,
            y=loss_counts,
            name="Losses",
            marker_color='red',
            opacity=0.7
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        title="Trade Analysis",
        height=800,
        width=1200,
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Save to file if requested
    if output_path:
        fig.write_html(output_path)
        logger.info(f"Trade distribution plot saved to {output_path}")

    # Show plot if requested
    if show_plot:
        fig.show()

    return fig


def plot_monthly_returns(results: Dict[str, Any], output_path: Optional[str] = None,
                         show_plot: bool = True) -> None:
    """
    Plot monthly returns heatmap

    Args:
        results: Dictionary with backtest results
        output_path: Path to save the output HTML file
        show_plot: Whether to show the plot
    """
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly is not available. Install with: pip install plotly")
        return None

    # Extract equity curve
    equity_curve = results['equity_curve']

    # Calculate daily returns
    daily_returns = equity_curve.pct_change().dropna()

    # Group by month and year and calculate cumulative return for each month
    monthly_returns = daily_returns.groupby([
        daily_returns.index.year,
        daily_returns.index.month
    ]).apply(lambda x: (1 + x).prod() - 1) * 100

    # Create a DataFrame with years as rows and months as columns
    monthly_returns_matrix = monthly_returns.unstack().T

    # Get month and year labels
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    years = sorted(monthly_returns.index.get_level_values(0).unique())

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=monthly_returns_matrix.values,
        x=years,
        y=months,
        colorscale=[
            [0, 'rgb(255,0,0)'],  # Red for negative returns
            [0.5, 'rgb(255,255,255)'],  # White for zero
            [1, 'rgb(0,128,0)']  # Green for positive returns
        ],
        zmid=0,  # Center the color scale at zero
        colorbar=dict(
            title='Return %',
            titleside='right'
        ),
        text=np.round(monthly_returns_matrix.values, 2),
        hovertemplate='Year: %{x}<br>Month: %{y}<br>Return: %{z:.2f}%<extra></extra>'
    ))

    # Update layout
    fig.update_layout(
        title="Monthly Returns Heatmap",
        xaxis=dict(title='Year'),
        yaxis=dict(title='Month', categoryorder='array', categoryarray=months),
        height=600,
        width=1000
    )

    # Save to file if requested
    if output_path:
        fig.write_html(output_path)
        logger.info(f"Monthly returns plot saved to {output_path}")

    # Show plot if requested
    if show_plot:
        fig.show()

    return fig


def plot_comparative_analysis(results_dict: Dict[str, Dict[str, Any]],
                              metrics_to_plot: List[str] = None,
                              output_path: Optional[str] = None,
                              show_plot: bool = True) -> None:
    """
    Plot comparative analysis of multiple backtest results

    Args:
        results_dict: Dictionary of backtest results with strategy names as keys
        metrics_to_plot: List of metrics to include in the comparison
        output_path: Path to save the output HTML file
        show_plot: Whether to show the plot
    """
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly is not available. Install with: pip install plotly")
        return None

    if metrics_to_plot is None:
        metrics_to_plot = [
            'total_return_pct', 'annualized_return_pct', 'sharpe_ratio',
            'max_drawdown_pct', 'win_rate_pct', 'profit_factor'
        ]

    # Extract metrics from results
    strategies = list(results_dict.keys())
    metrics_data = {}

    for metric in metrics_to_plot:
        metrics_data[metric] = [results_dict[strategy]['metrics'].get(metric, 0) for strategy in strategies]

    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Performance Metrics Comparison", "Equity Curves Comparison"),
        vertical_spacing=0.2,
        row_heights=[0.5, 0.5]
    )

    # Plot 1: Performance metrics comparison
    for i, metric in enumerate(metrics_data.keys()):
        # Format metric name for display
        display_metric = ' '.join(word.capitalize() for word in metric.split('_'))

        fig.add_trace(
            go.Bar(
                x=strategies,
                y=metrics_data[metric],
                name=display_metric,
                text=[f"{val:.2f}" for val in metrics_data[metric]],
                textposition='auto'
            ),
            row=1, col=1
        )

    # Plot 2: Equity curves comparison
    for strategy in strategies:
        equity_curve = results_dict[strategy]['equity_curve']
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                mode='lines',
                name=strategy
            ),
            row=2, col=1
        )

    # Update layout
    fig.update_layout(
        title="Strategy Comparison",
        barmode='group',
        height=1000,
        width=1200,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Save to file if requested
    if output_path:
        fig.write_html(output_path)
        logger.info(f"Comparative analysis plot saved to {output_path}")

    # Show plot if requested
    if show_plot:
        fig.show()

    return fig


def plot_live_trading_dashboard(data: pd.DataFrame, trades: List[Dict], metrics: Dict[str, Any],
                                output_path: Optional[str] = None):
    """
    Create a live trading dashboard with real-time updates

    Args:
        data: DataFrame with price data
        trades: List of trade dictionaries
        metrics: Dictionary with performance metrics
        output_path: Path to save the HTML file
    """
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly is not available. Install with: pip install plotly")
        return None

    # Create figure with subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        subplot_titles=("Price Chart and Trades", "Equity Curve", "Drawdown", "Positions")
    )

    # Plot price candlesticks
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name="Price"
        ),
        row=1, col=1
    )

    # Plot trades
    if trades:
        for trade in trades:
            # Determine marker color and shape based on trade type and profit/loss
            if trade['type'] == 'long':
                marker_color = 'green' if trade.get('pnl', 0) > 0 else 'red'
                entry_marker = 'triangle-up'
                exit_marker = 'triangle-down'
            else:  # short
                marker_color = 'green' if trade.get('pnl', 0) > 0 else 'red'
                entry_marker = 'triangle-down'
                exit_marker = 'triangle-up'

            # Plot entry point
            fig.add_trace(
                go.Scatter(
                    x=[trade['entry_date']],
                    y=[trade['entry_price'] * 1.005],
                    mode='markers+text',
                    marker=dict(
                        symbol=entry_marker,
                        size=12,
                        color=marker_color,
                        line=dict(width=2, color='black')
                    ),
                    text=['BUY' if trade['type'] == 'long' else 'SELL'],
                    textposition='top center',
                    name=f"{trade['type'].capitalize()} Entry",
                    showlegend=False
                ),
                row=1, col=1
            )

            # Plot exit point if trade is closed
            if 'exit_date' in trade and 'exit_price' in trade:
                fig.add_trace(
                    go.Scatter(
                        x=[trade['exit_date']],
                        y=[trade['exit_price'] * 0.995],
                        mode='markers+text',
                        marker=dict(
                            symbol=exit_marker,
                            size=12,
                            color=marker_color,
                            line=dict(width=2, color='black')
                        ),
                        text=['EXIT'],
                        textposition='bottom center',
                        name=f"{trade['type'].capitalize()} Exit",
                        showlegend=False
                    ),
                    row=1, col=1
                )

                # Connect entry to exit with a line
                fig.add_trace(
                    go.Scatter(
                        x=[trade['entry_date'], trade['exit_date']],
                        y=[trade['entry_price'], trade['exit_price']],
                        mode='lines',
                        line=dict(
                            color=marker_color,
                            width=1,
                            dash='dot'
                        ),
                        showlegend=False
                    ),
                    row=1, col=1
                )

    # Plot equity curve
    if 'equity_curve' in metrics:
        fig.add_trace(
            go.Scatter(
                x=metrics['equity_curve'].index,
                y=metrics['equity_curve'].values,
                mode='lines',
                name="Equity",
                line=dict(color='blue', width=2),
                fill='tozeroy'
            ),
            row=2, col=1
        )

    # Plot drawdown
    if 'equity_curve' in metrics:
        equity_series = metrics['equity_curve']
        peak = equity_series.cummax()
        drawdown = -((equity_series - peak) / peak) * 100

        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode='lines',
                name="Drawdown %",
                line=dict(color='red', width=2),
                fill='tozeroy'
            ),
            row=3, col=1
        )

    # Plot position sizes
    if 'position_sizes' in metrics:
        fig.add_trace(
            go.Scatter(
                x=metrics['position_sizes'].index,
                y=metrics['position_sizes'].values,
                mode='lines',
                name="Position Size",
                line=dict(color='purple', width=2)
            ),
            row=4, col=1
        )

    # Add performance metrics as annotations
    metrics_text = (
        f"Total Return: {metrics.get('total_return_pct', 0):.2f}%<br>"
        f"Profit: ${metrics.get('profit', 0):.2f}<br>"
        f"Win Rate: {metrics.get('win_rate_pct', 0):.2f}%<br>"
        f"Total Trades: {metrics.get('total_trades', 0)}<br>"
        f"Open Positions: {metrics.get('open_positions', 0)}"
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.99,
        text=metrics_text,
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        align="left"
    )

    # Update layout
    fig.update_layout(
        title="Live Trading Dashboard",
        xaxis_rangeslider_visible=False,
        height=1000,
        width=1200
    )

    # Update y-axis titles
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Equity", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=3, col=1)
    fig.update_yaxes(title_text="Position Size", row=4, col=1)

    # Save to file if requested
    if output_path:
        fig.write_html(output_path)
        logger.info(f"Live trading dashboard saved to {output_path}")

    return fig


# Matplotlib-based fallback functions for environments without Plotly
def plot_backtest_results_mpl(results: Dict[str, Any], output_path: Optional[str] = None,
                              show_plot: bool = True) -> None:
    """
    Plot backtest results using matplotlib (fallback)

    Args:
        results: Dictionary with backtest results
        output_path: Path to save the output file
        show_plot: Whether to show the plot
    """
    # Extract data from results
    equity_curve = results['equity_curve']
    trades = results['trades']
    metrics = results['metrics']

    # Create figure with subplots
    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True, gridspec_kw={'height_ratios': [3, 2, 1, 1]})

    # Plot 1: Price and trades
    axs[0].plot(results['data'].index, results['data']['close'], label='Close Price')

    # Plot trades
    for trade in trades:
        if trade['type'] == 'long':
            color = 'g' if trade['pnl'] > 0 else 'r'
            marker_entry = '^'
            marker_exit = 'v'
        else:  # short
            color = 'g' if trade['pnl'] > 0 else 'r'
            marker_entry = 'v'
            marker_exit = '^'

        # Plot entry and exit points
        axs[0].plot(trade['entry_date'], trade['entry_price'], marker=marker_entry, color=color, markersize=10)
        axs[0].plot(trade['exit_date'], trade['exit_price'], marker=marker_exit, color=color, markersize=10)

        # Connect with line
        axs[0].plot([trade['entry_date'], trade['exit_date']],
                    [trade['entry_price'], trade['exit_price']],
                    '--', color=color, alpha=0.5)

        # Add text labels
        axs[0].annotate('BUY' if trade['type'] == 'long' else 'SELL',
                     xy=(trade['entry_date'], trade['entry_price']),
                     xytext=(0, 10), textcoords='offset points',
                     ha='center', va='bottom')

        axs[0].annotate('EXIT',
                     xy=(trade['exit_date'], trade['exit_price']),
                     xytext=(0, -10), textcoords='offset points',
                     ha='center', va='top')

    axs[0].set_title('Price Chart and Trades')
    axs[0].legend()
    axs[0].grid(True)

    # Plot 2: Equity curve
    axs[1].plot(equity_curve.index, equity_curve.values, label='Equity', color='b')
    axs[1].set_title('Equity Curve')
    axs[1].grid(True)
    axs[1].fill_between(equity_curve.index, 0, equity_curve.values, alpha=0.3, color='b')

    # Plot 3: Drawdown in a separate subplot
    equity_series = pd.Series(equity_curve.values, index=equity_curve.index)
    peak = equity_series.cummax()
    drawdown = -((equity_series - peak) / peak) * 100

    axs[2].plot(drawdown.index, drawdown.values, color='r', label='Drawdown %')
    axs[2].fill_between(drawdown.index, 0, drawdown.values, alpha=0.3, color='r')
    axs[2].set_title('Drawdown %')
    axs[2].grid(True)

    # Plot 4: Position history
    position_history = results['position_history']
    axs[3].plot(position_history.index, position_history.values, label='Position', color='purple')
    axs[3].set_title('Position Size')
    axs[3].grid(True)

    # Add metrics as text
    metrics_text = (
        f"Total Return: {metrics['total_return_pct']:.2f}%\n"
        f"Annualized Return: {metrics['annualized_return_pct']:.2f}%\n"
        f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
        f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%\n"
        f"Win Rate: {metrics['win_rate_pct']:.2f}%\n"
        f"Profit Factor: {metrics['profit_factor']:.2f}\n"
        f"Total Trades: {metrics['total_trades']}"
    )

    # Add text to the first subplot
    axs[0].text(0.02, 0.98, metrics_text, transform=axs[0].transAxes,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

    # Adjust layout
    plt.tight_layout()

    # Save to file if requested
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Plot saved to {output_path}")

    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)