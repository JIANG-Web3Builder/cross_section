"""
Module 2: TFT Training Pipeline
Base Training (2021-2023) + Rolling Fine-Tune (2024-2026)

Architecture:
- PyTorch Forecasting TFT implementation
- Quantile Loss for uncertainty estimation
- Walk-Forward fine-tuning for adaptation
- Signal extraction (trend, uncertainty, risk)
"""
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import CrossEntropy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from config_tft import (
    FINAL_DIR, MODEL_DIR, OUTPUT_DIR,
    START_DATE, BASE_TRAIN_END,
    TFT_CONFIG, TFT_VARIABLES, BASE_TRAIN_CONFIG,
    TARGET_NAME, refresh_tft_variables
)


class CustomCrossEntropy(CrossEntropy):
    """Custom CrossEntropy that casts target to Long (int64) before calculation"""
    def loss(self, y_pred, target):
        target = target.long()
        return super().loss(y_pred, target)


class TFTTrainer:
    """TFT Base Model Training Engine (2021-2023)"""
    
    def __init__(self, data_path: Path = None):
        """
        Initialize TFT Trainer
        
        Args:
            data_path: Path to TFT training dataset parquet
        """
        refresh_tft_variables()
        self.data_path = data_path or (FINAL_DIR / 'tft_training_dataset.parquet')
        self.data = None
        self.base_model = None
        self.training_dataset = None
        self.validation_dataset = None
        
        # Check CUDA availability
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare TFT training dataset"""
        print("\n" + "=" * 60)
        print("LOADING TFT TRAINING DATASET")
        print("=" * 60)
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")
        
        print(f"  Loading from: {self.data_path}")
        df = pd.read_parquet(self.data_path)
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Ensure target is int64 for CrossEntropy (critical for classification)
        if TARGET_NAME in df.columns:
            df[TARGET_NAME] = df[TARGET_NAME].astype('int64')
            print(f"  Target dtype: {df[TARGET_NAME].dtype}")
        
        # Sort by time_idx
        df = df.sort_values(['symbol', 'time_idx'])
        
        print(f"  Shape: {df.shape}")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Symbols: {df['symbol'].unique()}")
        
        self.data = df
        return df
    
    def create_timeseries_dataset(
        self, 
        data: pd.DataFrame,
        max_encoder_length: int = None,
        max_prediction_length: int = None,
        training: bool = True
    ) -> TimeSeriesDataSet:
        """
        Create PyTorch Forecasting TimeSeriesDataSet
        
        Args:
            data: DataFrame with time series data
            max_encoder_length: Encoder length (lookback window)
            max_prediction_length: Prediction length (forecast horizon)
            training: If True, create training dataset; else validation
            
        Returns:
            TimeSeriesDataSet object
        """
        max_encoder_length = max_encoder_length or TFT_CONFIG['max_encoder_length']
        max_prediction_length = max_prediction_length or TFT_CONFIG['max_prediction_length']
        
        # Prepare time-varying known reals (must exist in data)
        time_varying_known_reals = [
            col for col in TFT_VARIABLES['time_varying_known_reals']
            if col in data.columns
        ]
        
        # Prepare time-varying unknown reals
        time_varying_unknown_reals = [
            col for col in TFT_VARIABLES['time_varying_unknown_reals']
            if col in data.columns
        ]
        
        print(f"\n  Creating TimeSeriesDataSet:")
        print(f"    Encoder length: {max_encoder_length}")
        print(f"    Prediction length: {max_prediction_length}")
        print(f"    Time-varying known: {len(time_varying_known_reals)} features")
        print(f"    Time-varying unknown: {len(time_varying_unknown_reals)} features")
        
        # Create dataset
        dataset = TimeSeriesDataSet(
            data,
            time_idx='time_idx',
            target=TARGET_NAME,
            group_ids=['symbol'],
            min_encoder_length=max_encoder_length // 2,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=TFT_VARIABLES['static_categoricals'],
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            # Classification: No target normalizer needed (labels are 0, 1, 2)
            target_normalizer=None,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )
        
        return dataset
    
    def train_base_model(self) -> TemporalFusionTransformer:
        """
        Train base TFT model on 2021-2023 data
        
        Returns:
            Trained TFT model
        """
        print("\n" + "=" * 80)
        print("BASE MODEL TRAINING (2021-2023)")
        print("=" * 80)
        
        # Filter base training period
        train_data = self.data[
            (self.data['timestamp'] >= START_DATE) & 
            (self.data['timestamp'] < BASE_TRAIN_END)
        ].copy()
        
        print(f"\n  Training period: {START_DATE} to {BASE_TRAIN_END}")
        print(f"  Training samples: {len(train_data):,}")
        
        # Split into train/validation (80/20)
        max_time_idx = train_data['time_idx'].max()
        train_cutoff = int(max_time_idx * 0.8)
        
        train_subset = train_data[train_data['time_idx'] <= train_cutoff]
        val_subset = train_data[train_data['time_idx'] > train_cutoff]
        
        print(f"  Train samples: {len(train_subset):,}")
        print(f"  Validation samples: {len(val_subset):,}")
        
        # Create datasets
        print("\n  Creating training dataset...")
        training = self.create_timeseries_dataset(train_subset, training=True)
        
        print("  Creating validation dataset...")
        validation = self.create_timeseries_dataset(val_subset, training=False)
        
        # Create dataloaders
        train_dataloader = training.to_dataloader(
            train=True,
            batch_size=BASE_TRAIN_CONFIG['batch_size'],
            num_workers=0
        )
        
        val_dataloader = validation.to_dataloader(
            train=False,
            batch_size=BASE_TRAIN_CONFIG['batch_size'] * 2,
            num_workers=0
        )
        
        # Configure TFT model for Classification
        print("\n  Configuring TFT model (Classification Mode)...")
        print("  Note: Class imbalance will be handled through data sampling or custom loss")
        
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=BASE_TRAIN_CONFIG['learning_rate'],
            hidden_size=TFT_CONFIG['hidden_size'],
            attention_head_size=TFT_CONFIG['attention_head_size'],
            dropout=TFT_CONFIG['dropout'],
            hidden_continuous_size=TFT_CONFIG['hidden_continuous_size'],
            # Classification: Use CrossEntropy loss
            # Note: pytorch-forecasting's CrossEntropy doesn't support weight parameter
            # Class imbalance is addressed by Triple Barrier width adjustment (0.5 instead of 2.0)
            loss=CustomCrossEntropy(),
            # Classification: Output size = 3 classes (0=Hold, 1=Long, 2=Short)
            output_size=3,
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
        
        print(f"    Model parameters: {sum(p.numel() for p in tft.parameters()):,}")
        
        # Setup trainer
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=1e-4,
            patience=TFT_CONFIG['patience'],
            verbose=True,
            mode='min'
        )
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=MODEL_DIR,
            filename='tft_base_best',
            monitor='val_loss',
            mode='min',
            save_top_k=1
        )
        
        logger = TensorBoardLogger(
            save_dir=OUTPUT_DIR,
            name='tft_logs'
        )
        
        trainer = pl.Trainer(
            max_epochs=BASE_TRAIN_CONFIG['max_epochs'],
            accelerator='auto',
            devices=1,
            gradient_clip_val=TFT_CONFIG['gradient_clip_val'],
            callbacks=[early_stop_callback, checkpoint_callback],
            logger=logger,
            enable_progress_bar=True,
        )
        
        # Train model
        print("\n  Starting training...")
        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )
        
        # Load best model
        best_model_path = checkpoint_callback.best_model_path
        print(f"\n  Loading best model from: {best_model_path}")
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        
        # Evaluate on validation set with detailed metrics
        print("\n" + "=" * 60)
        print("VALIDATION METRICS (Classification)")
        print("=" * 60)
        
        best_tft.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                # Get predictions
                output = best_tft(batch)
                if isinstance(output, dict) and 'prediction' in output:
                    pred_logits = output['prediction']
                else:
                    pred_logits = output
                
                # Convert to class predictions
                pred_classes = torch.argmax(pred_logits, dim=-1)
                
                # Get targets
                targets = batch[1][0]  # (y, weight) tuple
                
                all_preds.extend(pred_classes.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Remove NaN targets if any
        valid_mask = ~np.isnan(all_targets)
        all_preds = all_preds[valid_mask]
        all_targets = all_targets[valid_mask].astype(int)
        
        # Confusion Matrix
        cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2])
        print("\n  Confusion Matrix:")
        print("  Predicted ->  [Hold]  [Long]  [Short]")
        for i, label in enumerate(['Hold', 'Long', 'Short']):
            print(f"  Actual {label:6s}: {cm[i]}")
        
        # Per-class metrics
        precision = precision_score(all_targets, all_preds, average=None, labels=[0, 1, 2], zero_division=0)
        recall = recall_score(all_targets, all_preds, average=None, labels=[0, 1, 2], zero_division=0)
        f1 = f1_score(all_targets, all_preds, average=None, labels=[0, 1, 2], zero_division=0)
        
        print("\n  Per-Class Metrics:")
        print(f"  {'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print(f"  {'-'*46}")
        for i, label in enumerate(['Hold', 'Long', 'Short']):
            print(f"  {label:<10} {precision[i]:<12.2%} {recall[i]:<12.2%} {f1[i]:<12.2%}")
        
        # Overall metrics
        macro_precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        macro_recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        
        print(f"\n  Macro Average:")
        print(f"    Precision: {macro_precision:.2%}")
        print(f"    Recall:    {macro_recall:.2%}")
        print(f"    F1-Score:  {macro_f1:.2%}")
        
        # Critical check for trading viability
        print("\n  ⚠️ Trading Viability Check:")
        long_precision = precision[1]
        short_precision = precision[2]
        
        if long_precision < 0.55 and short_precision < 0.55:
            print(f"  ❌ FAIL: Long Precision={long_precision:.2%}, Short Precision={short_precision:.2%}")
            print(f"     Model precision < 55% for both Long and Short")
            print(f"     ⚠️ This model is NOT suitable for trading!")
        elif long_precision < 0.55 or short_precision < 0.55:
            print(f"  ⚠️ WARNING: Long Precision={long_precision:.2%}, Short Precision={short_precision:.2%}")
            print(f"     One direction has low precision, use with caution")
        else:
            print(f"  ✓ PASS: Long Precision={long_precision:.2%}, Short Precision={short_precision:.2%}")
            print(f"     Model meets minimum precision threshold (>55%)")
        
        print("=" * 60)
        
        # Save base model
        base_model_path = MODEL_DIR / 'tft_base.pth'
        torch.save(best_tft.state_dict(), base_model_path)
        print(f"  Base model saved to: {base_model_path}")
        
        self.base_model = best_tft
        self.training_dataset = training
        
        return best_tft
    
    def run_pipeline(self) -> TemporalFusionTransformer:
        """
        Run TFT base model training pipeline
        
        注意: 此方法只训练基座模型 (2021-2023)
        滚动微调请使用 train_tft_lgbm_walkforward.py (无未来函数)
        
        Returns:
            Trained TFT base model
        """
        print("\n" + "=" * 80)
        print("TFT BASE MODEL TRAINING PIPELINE")
        print("=" * 80)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Train base model
        model = self.train_base_model()
        
        print("\n" + "=" * 80)
        print("TFT BASE MODEL TRAINING COMPLETED!")
        print("=" * 80)
        print("\n下一步: 运行 python train_tft_lgbm_walkforward.py 进行滚动训练")
        
        return model


def main():
    """Main execution - 只训练基座模型"""
    trainer = TFTTrainer()
    model = trainer.run_pipeline()
    
    print("\n✓ TFT基座模型训练完成!")
    print("✓ 下一步: 运行 python train_tft_lgbm_walkforward.py")


if __name__ == "__main__":
    main()
