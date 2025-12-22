# main_pipeline.py
import torch
import numpy as np
import pandas as pd
import yaml
import logging
import random
import os
from torch.utils.data import DataLoader
from pathlib import Path

# Import our custom modules
from src.data.data_collector_v5 import DataCollector
from src.data.dataset_v5 import DataModule, DataConfig, StockDataset
from src.model.krnn_v5 import KRNNRegressor, ModelConfig
from src.utils.trainer_v5 import Trainer
from src.risk.evt import EVTEngine
from src.risk.moment_bounds import DiscreteConditionalMomentSolver
from src.portfolio.optimizer import MeanCVaROptimizer
from src.utils.generate_report import ReportGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed=2304571):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Global Seed set to {seed}")


def initialize_data_pipeline(config_path: str = "config_v5.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    data_dir = Path(config['paths']['data'])

    # Ensure data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)

    required_files = ['train.parquet', 'validation.parquet', 'test.parquet']
    if all((data_dir / f).exists() for f in required_files):
        logger.info("Data files found. Skipping download.")
        return config

    logger.info("Data files missing. Starting Data Collection...")
    collector = DataCollector(config)
    collector.collect_data()
    return config


def run_project_pipeline():
    # --- Step 0: Robustness & Init ---
    set_seed(42)
    print("\n[Phase 0] Initializing Pipeline & Checking Data...")
    config_dict = initialize_data_pipeline("config_v5.yaml")

    # Ensure report directory exists
    Path("./reports").mkdir(parents=True, exist_ok=True)
    reporter = ReportGenerator(output_dir="./reports")

    # --- Step 1: Data Loading ---
    print("\n[Phase 1] Loading Data Module (RAM Cache)...")

    # Use GPU if available for data loading optimization
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_config = DataConfig(
        sequence_length=config_dict['data']['window_size'],
        batch_size=config_dict['data']['batch_size'],
        device=device  # Pass device to load data directly to GPU if supported
    )
    data_module = DataModule(data_config)
    data_module.setup(Path(config_dict['paths']['data']))
    dataloaders = data_module.get_dataloaders()

    # --- Step 2: Model Training ---
    print("\n[Phase 2] Training K-Parallel KRNN (Heteroscedastic - Overfit Mode)...")
    feature_dim = len(data_module.train_dataset.feature_cols)

    # Updated ModelConfig to match new KRNNRegressor signature
    model_config = ModelConfig(
        input_dim=feature_dim,
        hidden_dim=config_dict['model']['hidden_dim'],
        num_layers=config_dict['model']['num_layers'],  # Added missing param
        k_dups=3,
        output_dim=2,  # Mu + Sigma
        dropout=config_dict['model']['dropout'],
        device=device
    )

    model = KRNNRegressor(model_config)

    # Initialize Trainer with new GaussianNLL logic
    trainer = Trainer(model, config=config_dict)

    # Train
    training_metrics = trainer.train(train_loader=dataloaders['train'], val_loader=dataloaders['val'])
    print(f"Training Complete. Final Val GNLL: {training_metrics.get('loss', 'N/A')}")

    # --- Step 3: Systematic Portfolio Selection ---
    print("\n[Phase 3] Running Systematic Portfolio Selection & Robust Risk Analysis...")

    val_df_path = Path(config_dict['paths']['data']) / 'validation.parquet'
    val_df = pd.read_parquet(val_df_path)
    all_tickers = val_df['Ticker'].unique()

    candidates = []
    top_n = config_dict.get('portfolio', {}).get('top_n_assets', 10)
    risk_conf = config_dict.get('portfolio', {}).get('risk_confidence', 0.95)
    n_sims = config_dict.get('portfolio', {}).get('n_simulations', 1000)

    model.eval()

    # --- 3a. Inference & Alpha Scan ---
    print("   Scanning assets for alpha...")
    for ticker in all_tickers:
        ticker_df = val_df[val_df['Ticker'] == ticker].copy()
        if len(ticker_df) < data_config.sequence_length + 20:
            continue

        # Create temporary dataset for inference
        ticker_ds = StockDataset(
            ticker_df,
            data_config.sequence_length,
            device=device
        )
        ticker_loader = DataLoader(ticker_ds, batch_size=2048, shuffle=False)  # Large batch for speed

        mus, sigmas, targets = [], [], []
        with torch.no_grad():
            for feat, targ in ticker_loader:
                # feat/targ already on device due to StockDataset logic
                mu, sigma = model(feat)
                mus.append(mu.cpu().numpy())
                sigmas.append(sigma.cpu().numpy())
                targets.append(targ.cpu().numpy())

        mus = np.concatenate(mus)
        sigmas = np.concatenate(sigmas)
        targets = np.concatenate(targets)

        avg_pred_return = np.mean(mus)
        avg_pred_volatility = np.mean(sigmas)

        # Calculate residuals for Risk Analysis
        residuals = (targets.flatten() - mus.flatten()) / (sigmas.flatten() + 1e-6)

        candidates.append({
            'Ticker': ticker,
            'Pred_Return': avg_pred_return,
            'Pred_Vol': avg_pred_volatility,
            'Residuals': residuals
        })

    # Sort by Absolute Return potential (Long/Short)
    candidates.sort(key=lambda x: abs(x['Pred_Return']), reverse=True)
    top_candidates = candidates[:top_n]
    print(f"   Selected Top {len(top_candidates)} Candidates.")

    # --- 3b. Robust Risk Analysis ---
    print(f"\n3. Risk Filter: Analyzing Heavy Tails (EVT) & Bounds (Naumova DCMP)...")

    scenarios_list = []
    final_tickers = []
    expected_returns_vec = []

    evt_engine = EVTEngine(tail_fraction=0.10)
    # Naumova Solver for Distributional Robustness
    dcmp_solver = DiscreteConditionalMomentSolver(n_points=500, support_range=(-10.0, 10.0))

    tail_data = {}
    candidates_report_data = []
    bounds_report_data = []

    for cand in top_candidates:
        ticker = cand['Ticker']
        residuals = cand['Residuals']

        # 1. EVT Analysis (Hill Estimator)
        evt_params = evt_engine.analyze_tails(residuals)
        gamma = evt_params['gamma']
        evt_metrics = evt_engine.calculate_risk_metrics(evt_params)

        # 2. Naumova Moment Bound (Worst-Case CVaR)
        dcmp_result = dcmp_solver.solve_dcmp(residuals, alpha=0.05, use_conditional=True)

        # Calculate Risk Ambiguity Gap
        risk_gap = dcmp_result.wc_cvar - evt_metrics['ES_0.99']

        tail_data[ticker] = {'gamma': gamma, 'residuals': residuals}

        # 3. Generate Forward-Looking Scenarios
        scenarios = evt_engine.generate_scenarios(
            n_simulations=n_sims,
            gamma=gamma,
            volatility=cand['Pred_Vol'],
            expected_return=cand['Pred_Return']
        )
        scenarios_list.append(scenarios)
        final_tickers.append(ticker)
        expected_returns_vec.append(cand['Pred_Return'])

        candidates_report_data.append({
            'Ticker': ticker,
            'Mu': cand['Pred_Return'],
            'Sigma': cand['Pred_Vol'],
            'Gamma': gamma,
            'ES': evt_metrics['ES_0.99'],
            'WC_ES': dcmp_result.wc_cvar,
            'Gap': risk_gap
        })

        bounds_report_data.append({
            'Ticker': ticker,
            'EVT_CVaR': evt_metrics['ES_0.99'],
            'DMP_CVaR': dcmp_result.wc_cvar,
            'Gamma': gamma
        })

        print(f"   - {ticker}: Gamma={gamma:.2f} | Gap={risk_gap:.2f}")

    if hasattr(reporter, 'plot_risk_bounds_comparison'):
        reporter.plot_risk_bounds_comparison(bounds_report_data)

    print("\n4. Optimization: Mean-CVaR (Long/Short)...")
    portfolio_report_data = []
    opt_metrics = {}

    if len(scenarios_list) > 1:
        scenarios_matrix = np.column_stack(scenarios_list)
        expected_returns_vec = np.array(expected_returns_vec)
        optimizer = MeanCVaROptimizer(confidence_level=risk_conf)

        # Set target return based on candidate quality
        abs_returns = np.abs(expected_returns_vec)
        target_ret = np.percentile(abs_returns, 75) * 0.5

        # We construct probabilities. For EVT simulation, they are uniform 1/N.
        # Ideally, we would use Scenario Reduction here to get weighted probabilities.
        # For this step, we assume uniform.
        uniform_probs = np.ones(n_sims) / n_sims

        result = optimizer.optimize(
            scenarios=scenarios_matrix,
            probabilities=uniform_probs,
            target_return=target_ret
        )

        if result:
            final_weights = result['weights']
            opt_metrics['Target_Ret'] = target_ret
            opt_metrics['CVaR'] = result['CVaR_Optimal']

            allocations = sorted(zip(final_tickers, final_weights), key=lambda x: abs(x[1]), reverse=True)
            for ticker, weight in allocations:
                if abs(weight) > 0.001:
                    portfolio_report_data.append({
                        'Ticker': ticker, 'Weight': weight, 'Gamma': tail_data[ticker]['gamma']
                    })

            gammas = [tail_data[t]['gamma'] for t in final_tickers]
            reporter.plot_allocation_vs_risk(final_tickers, final_weights, gammas)
        else:
            print("Optimizer failed.")

    # --- Phase 4: Evaluation ---
    print("\n[Phase 4] Out-of-Sample Evaluation...")
    test_loader = dataloaders['test']
    model.eval()
    all_mus, all_targets = [], []
    with torch.no_grad():
        for feat, targ in test_loader:
            mu, _ = model(feat)
            all_mus.append(mu.cpu().numpy())
            all_targets.append(targ.cpu().numpy().flatten())

    all_mus = np.concatenate(all_mus)
    all_targets = np.concatenate(all_targets)
    r2 = reporter.generate_diagnostics(all_mus, all_targets)

    # Strategy Simulation (Long/Short)
    strategy_returns = np.sign(all_mus) * all_targets
    cum_market = np.cumsum(all_targets)
    cum_strategy = np.cumsum(strategy_returns)

    test_metrics = {'R2': r2, 'Market_Cum': cum_market[-1], 'Strategy_Cum': cum_strategy[-1]}

    # --- Phase 5: Portfolio Backtest (The Pivot) ---
    print("\n[Phase 5] Running Out-of-Sample Portfolio Backtest...")

    test_file = Path(config_dict['paths']['data']) / 'test.parquet'
    test_df = pd.read_parquet(test_file)

    # Check if 'Log_Return' exists, if not use pct_change of Close or Target
    # Our synthetic data likely has 'Target' which is return.
    target_col = 'Target' if 'Target' in test_df.columns else 'Log_Return'
    if target_col not in test_df.columns:
        # Fallback for synthetic data structure
        # Assuming the last column is the target or creating it
        # For safety, let's look for known target columns
        for col in ['Target', 'Log_Return', 'Return', 'Close']:
            if col in test_df.columns:
                target_col = col
                break

    # Pivot to get Returns Matrix (Date x Ticker)
    try:
        returns_matrix = test_df.pivot(index='Date', columns='Ticker', values=target_col)
    except KeyError:
        print("Warning: Could not pivot test data for backtest. Columns found:", test_df.columns)
        returns_matrix = pd.DataFrame()  # Empty fallback

    if not returns_matrix.empty:
        # Rescale Z-Scores to Percentage Returns (approximate)
        returns_matrix = returns_matrix * 0.02

        if 'final_weights' in locals() and len(final_weights) > 0:
            weight_map = pd.Series(final_weights, index=final_tickers)
            portfolio_assets = [t for t in final_tickers if t in returns_matrix.columns]

            if portfolio_assets:
                aligned_returns = returns_matrix[portfolio_assets].fillna(0.0)
                aligned_weights = weight_map[portfolio_assets]
                aligned_weights = aligned_weights / (aligned_weights.abs().sum() + 1e-6)  # Normalize leverage

                portfolio_daily_rets = aligned_returns.dot(aligned_weights)

                # Benchmark (Equal Weight)
                equal_weights = np.ones(len(portfolio_assets)) / len(portfolio_assets)
                market_daily_rets = aligned_returns.dot(equal_weights)

                # Metrics
                total_port_ret = portfolio_daily_rets.sum()
                total_mkt_ret = market_daily_rets.sum()

                port_vol = portfolio_daily_rets.std() * np.sqrt(252)
                mkt_vol = market_daily_rets.std() * np.sqrt(252)

                port_sharpe = (portfolio_daily_rets.mean() * 252) / (port_vol + 1e-6)
                mkt_sharpe = (market_daily_rets.mean() * 252) / (mkt_vol + 1e-6)

                print("\n=== OUT-OF-SAMPLE RESULTS (2024+) ===")
                print(f"{'Metric':<15} {'Strategy':<12} {'Benchmark':<12}")
                print("-" * 45)
                print(f"{'Total Return':<15} {total_port_ret * 100:<10.2f}%  {total_mkt_ret * 100:<10.2f}%")
                print(f"{'Volatility':<15} {port_vol * 100:<10.2f}%  {mkt_vol * 100:<10.2f}%")
                print(f"{'Sharpe Ratio':<15} {port_sharpe:<10.4f}  {mkt_sharpe:<10.4f}")

                test_metrics.update({
                    'Strategy_Return': total_port_ret,
                    'Strategy_Sharpe': port_sharpe,
                })
            else:
                print("Warning: No portfolio assets found in test data.")
    else:
        print("Skipping detailed backtest due to data format issues.")

    # Save Final Report
    reporter.save_comprehensive_report(
        candidates=candidates_report_data,
        portfolio=portfolio_report_data,
        opt_metrics=opt_metrics,
        test_metrics=test_metrics,
        bounds_data=bounds_report_data
    )
    print(f"\nPipeline Complete. Reports generated in {reporter.output_dir}")


if __name__ == "__main__":
    run_project_pipeline()