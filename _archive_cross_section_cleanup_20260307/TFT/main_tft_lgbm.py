"""
Main Execution Script for TFT-LGBM Hybrid System
Complete end-to-end pipeline execution

Usage:
    python main_tft_lgbm.py --mode all          # Run complete pipeline
    python main_tft_lgbm.py --mode data         # Only data processing
    python main_tft_lgbm.py --mode tft          # Only TFT training
    python main_tft_lgbm.py --mode lgbm         # Only LGBM integration
    python main_tft_lgbm.py --mode strategy     # Only strategy execution
    python main_tft_lgbm.py --mode backtest     # Only backtesting
"""
import argparse
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config_tft import OUTPUT_DIR, validate_config
from factor_research import FactorResearchPipeline
from tft_data_processor import TFTDataProcessor
from train_tft import TFTTrainer
from train_tft_lgbm_walkforward import WalkForwardTrainer
from strategy_engine import StrategyEngine
from backtest_tft_lgbm import BacktestEngine

try:
    from train_lgbm_with_tft import TFTLGBMIntegrator
except ImportError:
    TFTLGBMIntegrator = None


class TFTLGBMPipeline:
    """Complete TFT-LGBM pipeline"""
    
    def __init__(self):
        self.factor_research = None
        self.data_processor = None
        self.tft_trainer = None
        self.walkforward_trainer = None
        self.lgbm_integrator = None
        self.strategy_engine = None
        self.backtest_engine = None

    def run_factor_research(self):
        print("\n" + "=" * 80)
        print("MODULE 0: FACTOR RESEARCH")
        print("=" * 80)

        self.factor_research = FactorResearchPipeline()
        metrics, selection_payload = self.factor_research.run_pipeline()

        print(f"\n✓ Module 0 completed!")
        print(f"  Selected factors: {selection_payload['selected_factors']}")
        print(f"  Diagnostics rows: {len(metrics)}")

        return metrics, selection_payload
        
    def run_data_processing(self):
        """Module 1: Data Processing"""
        print("\n" + "=" * 80)
        print("MODULE 1: TFT DATA PROCESSING")
        print("=" * 80)
        
        self.data_processor = TFTDataProcessor()
        final_df = self.data_processor.run_pipeline()
        
        print(f"\n✓ Module 1 completed!")
        print(f"  Output: {final_df.shape[0]:,} rows, {final_df.shape[1]} columns")
        
        return final_df
    
    def run_tft_base_training(self):
        """Module 2a: TFT Base Model Training (2021-2023)"""
        print("\n" + "=" * 80)
        print("MODULE 2a: TFT BASE MODEL TRAINING")
        print("=" * 80)
        
        self.tft_trainer = TFTTrainer()
        model = self.tft_trainer.run_pipeline()
        
        print(f"\n✓ Module 2a completed!")
        print(f"  Base model trained and saved")
        
        return model
    
    def run_walkforward_training(self):
        """Module 2b: Walk-Forward Training (无未来函数)"""
        print("\n" + "=" * 80)
        print("MODULE 2b: WALK-FORWARD TRAINING")
        print("=" * 80)
        
        self.walkforward_trainer = WalkForwardTrainer()
        signals = self.walkforward_trainer.run_walkforward()
        
        print(f"\n✓ Module 2b completed!")
        print(f"  TFT signals: {len(signals):,} rows")
        
        return signals
    
    def run_lgbm_integration(self):
        """Module 3: LGBM Integration"""
        print("\n" + "=" * 80)
        print("MODULE 3: LGBM INTEGRATION")
        print("=" * 80)

        if TFTLGBMIntegrator is None:
            raise FileNotFoundError('train_lgbm_with_tft.py not found in workspace')
        
        self.lgbm_integrator = TFTLGBMIntegrator()
        predictions = self.lgbm_integrator.run_pipeline()
        
        print(f"\n✓ Module 3 completed!")
        print(f"  LGBM predictions: {len(predictions):,} rows")
        
        return predictions
    
    def run_strategy_execution(self):
        """Module 4: Strategy Execution"""
        print("\n" + "=" * 80)
        print("MODULE 4: STRATEGY EXECUTION")
        print("=" * 80)
        
        self.strategy_engine = StrategyEngine()
        regimes, positions, weights = self.strategy_engine.run_pipeline()
        
        print(f"\n✓ Module 4 completed!")
        print(f"  Regimes: {len(regimes)} timestamps")
        print(f"  Positions: {len(positions):,} entries")
        print(f"  Weights: {len(weights):,} entries")
        
        return regimes, positions, weights
    
    def run_backtesting(self):
        """Module 5: Backtesting"""
        print("\n" + "=" * 80)
        print("MODULE 5: BACKTESTING")
        print("=" * 80)
        
        self.backtest_engine = BacktestEngine()
        equity_curve, metrics = self.backtest_engine.run_pipeline()
        
        print(f"\n✓ Module 5 completed!")
        print(f"  Equity curve: {len(equity_curve):,} points")
        print(f"  Total return: {metrics['total_return_pct']:.2f}%")
        print(f"  Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        
        return equity_curve, metrics
    
    def run_complete_pipeline(self):
        """Run all modules in sequence (无未来函数版本)"""
        print("\n" + "=" * 80)
        print("TFT-LGBM COMPLETE PIPELINE (无未来函数)")
        print("=" * 80)
        print("\nThis will run all modules:")
        print("  0. Factor Research")
        print("  1. Data Processing")
        print("  2a. TFT Base Model Training (2021-2023)")
        print("  2b. Walk-Forward Training (2024-2026)")
        print("  3. LGBM Integration")
        print("  4. Strategy Execution")
        print("  5. Backtesting")
        print("\n" + "=" * 80)
        
        # Validate configuration
        print("\nValidating configuration...")
        try:
            validate_config()
            print("✓ Configuration valid!")
        except ValueError as e:
            print(f"✗ Configuration error: {e}")
            return False
        
        try:
            self.run_factor_research()
        except Exception as e:
            print(f"\n✗ Module 0 failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Module 1: Data Processing
        try:
            self.run_data_processing()
        except Exception as e:
            print(f"\n✗ Module 1 failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Module 2a: TFT Base Model Training
        try:
            self.run_tft_base_training()
        except Exception as e:
            print(f"\n✗ Module 2a failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Module 2b: Walk-Forward Training
        try:
            self.run_walkforward_training()
        except Exception as e:
            print(f"\n✗ Module 2b failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Module 3: LGBM Integration
        try:
            self.run_lgbm_integration()
        except Exception as e:
            print(f"\n✗ Module 3 failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Module 4: Strategy Execution
        try:
            self.run_strategy_execution()
        except Exception as e:
            print(f"\n✗ Module 4 failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Module 5: Backtesting
        try:
            equity_curve, metrics = self.run_backtesting()
        except Exception as e:
            print(f"\n✗ Module 5 failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Final summary
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nFinal Results:")
        print(f"  Total Return: {metrics['total_return_pct']:.2f}%")
        print(f"  Annualized Return: {metrics['annualized_return_pct']:.2f}%")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        print(f"  Calmar Ratio: {metrics['calmar_ratio']:.2f}")
        print(f"\nAll outputs saved to: {OUTPUT_DIR}")
        
        return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='TFT-LGBM Hybrid Quantitative Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_tft_lgbm.py --mode all          # Run complete pipeline
  python main_tft_lgbm.py --mode research     # Only factor research
  python main_tft_lgbm.py --mode data         # Only data processing
  python main_tft_lgbm.py --mode tft          # Only TFT training
  python main_tft_lgbm.py --mode lgbm         # Only LGBM integration
  python main_tft_lgbm.py --mode strategy     # Only strategy execution
  python main_tft_lgbm.py --mode backtest     # Only backtesting
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['all', 'research', 'data', 'tft', 'lgbm', 'strategy', 'backtest'],
        default='all',
        help='Execution mode (default: all)'
    )
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = TFTLGBMPipeline()
    
    # Execute based on mode
    try:
        if args.mode == 'all':
            success = pipeline.run_complete_pipeline()
            sys.exit(0 if success else 1)

        elif args.mode == 'research':
            pipeline.run_factor_research()
        
        elif args.mode == 'data':
            pipeline.run_factor_research()
            pipeline.run_data_processing()
        
        elif args.mode == 'tft':
            pipeline.run_tft_base_training()
            pipeline.run_walkforward_training()
        
        elif args.mode == 'lgbm':
            pipeline.run_lgbm_integration()
        
        elif args.mode == 'strategy':
            pipeline.run_strategy_execution()
        
        elif args.mode == 'backtest':
            pipeline.run_backtesting()
        
        print("\n✓ Execution completed successfully!")
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n\n✗ Execution interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n✗ Execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
