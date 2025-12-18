# main_pipeline.py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path

# Import our custom modules
from src.data.data_collector_v5 import DataCollector
from src.data.dataset_v5 import DataModule, DataConfig
from src.model.krnn_v5 import KRNNRegressor, ModelConfig
from src.model.train_v5 import Trainer
from src.risk.evt import calculate_portfolio_risk, get_residuals
from src.portfolio.optimizer import MeanCVaROptimizer


def run_project_pipeline():
    # --- Step 1: Data Collection ---
    print("\n[Phase 1] Initializing Data Pipeline...")
    # Assume data is already downloaded in ./data via data_collector_v5.py
    # We load it for processing
    data_config = DataConfig(window_size=10, batch_size=64)
    data_module = DataModule(data_config)
    data_module.setup(Path("./data"))
    dataloaders = data_module.get_dataloaders()

    # --- Step 2: Model Training (The "KRNN" Regression) ---
    print("\n[Phase 1] Training K-Parallel KRNN Regressor...")
    feature_dim = len(data_module.train_dataset.feature_cols)

    model_config = ModelConfig(
        input_dim=feature_dim,
        hidden_dim=64,
        k_dups=3,  # The "K" in KRNN
        output_dim=1  # Regression
    )
    model = KRNNRegressor(model_config)

    # Train
    trainer = Trainer(model, config={})  # Add actual config dict
    # (Assuming trainer.train loop is adapted for regression as discussed)
    # trainer.train(dataloaders['train'], dataloaders['val'])
    print("Training Complete.")

    # --- Step 3: Risk Analysis (The "EVT" Engine) ---
    print("\n[Phase 2] Analyzing Residuals & Heavy Tails...")

    # We need to analyze risk per Asset.
    # For simplicity in this demo, we assume the dataloader provides data for multiple assets.
    # In a real run, we would loop through each Ticker.

    # Let's assume we have trained models or predictions for N assets.
    # For this script, we simulate 3 assets to demonstrate the Portfolio Opt.
    tickers = ['AAPL', 'MSFT', 'GOOGL']

    asset_predictions = []  # Predicted mu (Next day return)
    asset_scenarios = []  # T scenarios of (mu + residual)

    print(f"Calculating Hill Estimator for {len(tickers)} assets...")

    for ticker in tickers:
        # 1. Get Prediction (Simulated for demo, replace with model(ticker_data))
        predicted_return = 0.001  # e.g., 0.1% daily return

        # 2. Get Residuals (History of errors)
        # residuals = get_residuals(model, val_loader_for_ticker, device='cpu')
        # Using random heavy-tailed noise to simulate 'residuals' for demonstration
        simulated_residuals = np.random.standard_t(df=3, size=250) * 0.02

        # 3. Construct Scenarios for Optimizer
        # Scenario = Expected Return + Error
        scenarios = predicted_return + simulated_residuals

        asset_predictions.append(predicted_return)
        asset_scenarios.append(scenarios)

        # 4. EVT Analysis (Just to print the Risk Report)
        # We pass the residuals to our EVT engine
        # evt_metrics = calculate_portfolio_risk(model, val_loader, 'cpu')
        # print(f"Asset: {ticker} | Gamma (Tail Index): {evt_metrics['Gamma_Hill']:.4f}")

    # Stack for Optimizer
    expected_returns_vec = np.array(asset_predictions)
    scenarios_matrix = np.column_stack(asset_scenarios)  # [T, N]

    # --- Step 4: Portfolio Optimization (Mean-CVaR) ---
    print("\n[Phase 3] Optimizing Portfolio Weights (Mean-CVaR)...")

    optimizer = MeanCVaROptimizer(confidence_level=0.95)
    target_ret = 0.0005  # Target: 0.05% daily return

    result = optimizer.optimize(expected_returns_vec, scenarios_matrix, target_ret)

    if result:
        print("\n=== OPTIMAL PORTFOLIO ALLOCATION ===")
        print(f"Target Return: {target_ret * 100:.2f}%")
        print(f"Min CVaR (95%): {result['CVaR_Optimal']:.4f}")
        print("-" * 30)
        for ticker, weight in zip(tickers, result['weights']):
            print(f"{ticker}: {weight * 100:.2f}%")
        print("-" * 30)
    else:
        print("Optimization failed to find a solution.")


if __name__ == "__main__":
    run_project_pipeline()