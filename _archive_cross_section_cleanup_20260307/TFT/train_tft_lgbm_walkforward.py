"""
TFT Walk-Forward 信号生成

流程:
1. 使用已有的TFT基座模型 (2021-2023)
2. 从2024年开始，每个月:
   - TFT微调 (用过去6个月数据)
   - 生成当月TFT预测信号
   - 保存信号供LGBM使用

注意: 本脚本只负责生成TFT宏观信号
      LGBM训练请使用 train_lgbm_with_tft.py
"""
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

from scipy.stats import norm
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List
from collections import deque

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import CrossEntropy
import pytorch_lightning as pl
from train_tft import CustomCrossEntropy

from config_tft import (
    FINAL_DIR, OUTPUT_DIR, MODEL_DIR,
    ROLLING_START, END_DATE,
    TFT_CONFIG, TFT_VARIABLES, FINETUNE_CONFIG,
    TARGET_NAME, refresh_tft_variables
)


# ================= Triple Barrier Classification Signal Generation =================
# 不再使用复杂的 TFTRollingStrategy，改用简单的概率驱动择时


class TFTSignalGenerator:
    """TFT Walk-Forward 信号生成器"""
    
    def __init__(self):
        refresh_tft_variables()
        self.data_path = FINAL_DIR / 'tft_training_dataset.parquet'
        self.base_model_path = MODEL_DIR / 'tft_base.pth'
        self.data = None
        
        # 检查基座模型是否存在
        if not self.base_model_path.exists():
            raise FileNotFoundError(
                f"基座模型不存在: {self.base_model_path}\n"
                "请先运行 train_tft.py 训练基座模型"
            )
        
        print(f"✓ 找到基座模型: {self.base_model_path}")
    
    def load_data(self):
        """加载数据"""
        print("\n" + "=" * 80)
        print("加载训练数据")
        print("=" * 80)
        
        self.data = pd.read_parquet(self.data_path)
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        self.data = self.data.sort_values(['symbol', 'time_idx'])
        
        print(f"  数据形状: {self.data.shape}")
        print(f"  日期范围: {self.data['timestamp'].min()} 到 {self.data['timestamp'].max()}")
        
        return self.data
    
    def create_timeseries_dataset(self, data: pd.DataFrame) -> TimeSeriesDataSet:
        """创建时间序列数据集"""
        time_varying_known_reals = [
            col for col in TFT_VARIABLES['time_varying_known_reals']
            if col in data.columns
        ]
        
        time_varying_unknown_reals = [
            col for col in TFT_VARIABLES['time_varying_unknown_reals']
            if col in data.columns
        ]
        
        dataset = TimeSeriesDataSet(
            data,
            time_idx='time_idx',
            target=TARGET_NAME,
            group_ids=['symbol'],
            min_encoder_length=TFT_CONFIG['max_encoder_length'] // 2,
            max_encoder_length=TFT_CONFIG['max_encoder_length'],
            min_prediction_length=1,
            max_prediction_length=TFT_CONFIG['max_prediction_length'],
            static_categoricals=TFT_VARIABLES['static_categoricals'],
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            # Classification: No target normalizer
            target_normalizer=None,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )
        
        return dataset
    
    def finetune_tft(self, train_data: pd.DataFrame) -> TemporalFusionTransformer:
        """微调TFT模型"""
        # 创建数据集
        dataset = self.create_timeseries_dataset(train_data)
        dataloader = dataset.to_dataloader(
            train=True,
            batch_size=FINETUNE_CONFIG['batch_size'],
            num_workers=0
        )
        
        # 创建模型 (Classification)
        # Note: Class imbalance handled by Triple Barrier width=0.5
        model = TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=FINETUNE_CONFIG['learning_rate'],
            hidden_size=TFT_CONFIG['hidden_size'],
            attention_head_size=TFT_CONFIG['attention_head_size'],
            dropout=TFT_CONFIG['dropout'],
            hidden_continuous_size=TFT_CONFIG['hidden_continuous_size'],
            loss=CustomCrossEntropy(),
            output_size=3,  # 3 classes
        )
        
        # 加载基座模型权重
        model.load_state_dict(torch.load(self.base_model_path))
        
        # 微调（静默模式）
        trainer = pl.Trainer(
            max_epochs=FINETUNE_CONFIG['max_epochs'],
            accelerator='auto',
            devices=1,
            gradient_clip_val=TFT_CONFIG['gradient_clip_val'],
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
        
        trainer.fit(model, train_dataloaders=dataloader)
        
        return model, dataset
    
    def predict_tft(self, model, reference_dataset, pred_data: pd.DataFrame) -> pd.DataFrame:
        """生成TFT分类预浌信号 (Classification Probabilities)"""
        try:
            pred_dataset = TimeSeriesDataSet.from_dataset(
                reference_dataset,
                pred_data,
                predict=True,
                stop_randomization=True
            )
            
            pred_dataloader = pred_dataset.to_dataloader(
                train=False,
                batch_size=128,
                num_workers=0
            )
            
            raw_predictions = model.predict(
                pred_dataloader,
                mode='prediction',  # Classification mode
                return_index=True,
                return_x=False
            )
            
            # 处理预浌结果 - 分类模型输出logits或probabilities
            if isinstance(raw_predictions, list):
                preds = raw_predictions[0]
            else:
                preds = raw_predictions.output
            
            # 转换为概率
            if isinstance(preds, torch.Tensor):
                # Apply softmax to get probabilities (Batch, Time, 3)
                probs = torch.softmax(preds, dim=-1).cpu().numpy()
            else:
                probs = preds
            
            # 我们只关心预浌窗口的第一个点（从现在开始的预测）
            # probs shape: (Batch, Horizon, 3) -> 取 Horizon 的第一步
            if probs.ndim == 3:
                final_probs = probs[:, 0, :]  # Shape: (Batch, 3)
            else:
                final_probs = probs
            
            # 构造 DataFrame
            pred_df = pd.DataFrame(
                final_probs, 
                columns=['prob_neutral', 'prob_long', 'prob_short']
            )
            
            # 添加时间戳
            pred_df['timestamp'] = pred_data['timestamp'].unique()[:len(pred_df)]
            
            # 提取TFT信号 (用于LGBM兼容性)
            # 将概率转换为信号
            pred_df['tft_trend'] = pred_df['prob_long'] - pred_df['prob_short']  # -1 to 1
            pred_df['tft_uncertainty'] = pred_df['prob_neutral']  # 0 to 1
            pred_df['tft_upside'] = pred_df['prob_long']  # 0 to 1
            pred_df['tft_downside'] = pred_df['prob_short']  # 0 to 1
            
            return pred_df
            
        except Exception as e:
            print(f"    预测错误: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def output_signals(self, fold_num: int, period: str, tft_signals: pd.DataFrame):
        """输出当前周期的TFT信号统计和持仓策略（Score-based 相对强弱）"""
        print(f"\n{'=' * 70}")
        print(f"  📊 顶级量化择时报告 (Score-Based) - 第 {fold_num} 周")
        print(f"{'=' * 70}")
        
        # 计算平均概率
        avg_prob_long = tft_signals['prob_long'].mean()
        avg_prob_short = tft_signals['prob_short'].mean()
        avg_prob_neutral = tft_signals['prob_neutral'].mean()
        
        print(f"\n  🎲 市场概率预测:")
        print(f"    P(Long/Up):    {avg_prob_long:.2%}")
        print(f"    P(Short/Down): {avg_prob_short:.2%}")
        print(f"    P(Hold/Chop):  {avg_prob_neutral:.2%}")
        
        # === 核心择时逻辑: Score-Based (相对强弱而非绝对阈值) ===
        # Score = Prob_Long / (Prob_Long + Prob_Short)
        # 这个指标在 0-1 之间，0.5 表示中性，>0.6 表示看多，<0.4 表示看空
        directional_sum = avg_prob_long + avg_prob_short
        if directional_sum > 1e-6:  # 防止除零
            score = avg_prob_long / directional_sum
        else:
            score = 0.5  # 默认中性
        
        print(f"\n  🎯 方向性分数 (Score):")
        print(f"    Score = {score:.3f}  (0=空, 0.5=中性, 1=多)")
        print(f"    Neutral Prob = {avg_prob_neutral:.2%}")
        
        regime = "WAIT ☕"
        leverage = 0.0
        
        # 判断逻辑: Score > 0.6 且 Neutral < 0.5 -> 做多
        #           Score < 0.4 且 Neutral < 0.5 -> 做空
        #           其他 -> 观望
        
        if score > 0.6 and avg_prob_neutral < 0.5:
            regime = "STRONG_BULL 🚀"
            # 根据 Score 偏离度计算杠杆
            leverage = (score - 0.5) * 3.0  # 0.6->0.3x, 0.7->0.6x, 0.8->0.9x, 0.9->1.2x
            leverage = min(1.5, leverage)  # 上限 1.5x
            
        elif score < 0.4 and avg_prob_neutral < 0.5:
            regime = "STRONG_BEAR 🩸"
            # 根据 Score 偏离度计算杠杆
            leverage = (score - 0.5) * 3.0  # 0.4->-0.3x, 0.3->-0.6x, 0.2->-0.9x
            leverage = max(-1.0, leverage)  # 做空限制 1x
            
        elif avg_prob_neutral > 0.6:
            regime = "DEAD_CALM 😴"
            leverage = 0.0
            
        else:
            regime = "NOISY 🦆"  # 信号不明确
            leverage = 0.0
        
        print(f"\n  🎯 市场状态: {regime}")
        print(f"  📊 建议仓位: {leverage:+.2f}x")
        
        if abs(leverage) > 0.3:
            print("  💼 操作: 方向性明确，建议入场")
        else:
            print("  💼 操作: 信号不明确，空仓等待")
            
        print(f"\n  📝 逻辑说明:")
        print(f"    - Score-Based: 使用相对强弱而非绝对阈值")
        print(f"    - 做多条件: Score > 0.6 且 Neutral < 0.5")
        print(f"    - 做空条件: Score < 0.4 且 Neutral < 0.5")
        print(f"    - 杠杆计算: 根据 Score 偏离 0.5 的程度动态调整")
            
        print(f"{'=' * 70}")
    
    def run_walkforward(self):
        """运行TFT Walk-Forward信号生成（每日预测）"""
        print("\n" + "=" * 80)
        print("TFT WALK-FORWARD 每日信号生成")
        print("=" * 80)
        
        # 加载数据
        self.load_data()
        
        # 获取所有时间戳
        all_timestamps = self.data['timestamp'].unique()
        all_timestamps = pd.to_datetime(all_timestamps).sort_values()
        
        # 滚动窗口参数（每日预测）
        train_window = FINETUNE_CONFIG['train_window_hours']  # 720小时 = 1个月
        test_window = FINETUNE_CONFIG['test_window_hours']    # 24小时 = 1天
        embargo = FINETUNE_CONFIG['embargo_hours']            # 0小时（无隔离期）
        
        print(f"\n滚动窗口配置:")
        print(f"  训练窗口: {train_window} 小时 (~{train_window//24} 天) - 使用过去3个月数据")
        print(f"  预测窗口: {test_window} 小时 (~{test_window//24} 天) - 预测未来1周")
        print(f"  滚动步长: {test_window} 小时 - 每周滚动一次")
        
        all_tft_signals = []
        
        # 从2024年开始（有足够历史数据）
        start_date = pd.Timestamp(ROLLING_START)  # 2024-01-01
        start_idx = all_timestamps.searchsorted(start_date)
        
        # 确保有足够的训练数据
        current_idx = max(start_idx, train_window)
        fold_num = 0
        
        print(f"\n开始日期: {all_timestamps[current_idx]}")
        print(f"预计生成信号数: ~{(len(all_timestamps) - current_idx) // test_window} 天\n")
        
        while current_idx < len(all_timestamps):
            fold_num += 1
            
            # 定义时间切片（每日滚动）
            train_start_idx = current_idx - train_window
            train_end_idx = current_idx
            test_start_idx = current_idx
            test_end_idx = min(current_idx + test_window, len(all_timestamps))
            
            train_start = all_timestamps[train_start_idx]
            train_end = all_timestamps[train_end_idx - 1]
            test_start = all_timestamps[test_start_idx]
            test_end = all_timestamps[test_end_idx - 1]
            
            # 获取训练和测试时间段
            train_times = all_timestamps[train_start_idx:train_end_idx]
            test_times = all_timestamps[test_start_idx:test_end_idx]
            
            period_str = test_start.strftime('%Y-%m-%d')
            
            # 每次都输出进度（因为是每周预测）
            print(f"\n{'*' * 70}")
            print(f"  第 {fold_num} 周 - {period_str}")
            print(f"  训练: {train_start.strftime('%Y-%m-%d')} -> {train_end.strftime('%Y-%m-%d')} (过去{train_window//24}天)")
            print(f"  预测: {test_start.strftime('%Y-%m-%d')} -> {test_end.strftime('%Y-%m-%d')} (未来7天)")
            print(f"{'*' * 70}")
            
            # ========== Step 1: 准备训练数据 ==========
            train_data = self.data[
                self.data['timestamp'].isin(train_times)
            ].copy()
            
            if len(train_data) < 100:
                print(f"  ⚠️ 训练数据不足，跳过")
                current_idx += test_window
                continue
            
            # ========== Step 2: TFT微调 ==========
            print(f"  [1/3] 微调TFT模型 (训练样本: {len(train_data):,})...")
            
            tft_model, tft_dataset = self.finetune_tft(train_data)
            
            # ========== Step 3: 准备预测数据 ==========
            # TFT预测需要encoder历史数据 + 预测窗口
            encoder_length = TFT_CONFIG['max_encoder_length']  # 168小时
            
            # 预测数据需要包含: [当前时间-encoder_length, 当前时间+test_window]
            pred_start_idx = max(0, test_start_idx - encoder_length)
            pred_end_idx = test_end_idx
            pred_times_extended = all_timestamps[pred_start_idx:pred_end_idx]
            
            pred_data = self.data[
                self.data['timestamp'].isin(pred_times_extended)
            ].copy()
            
            if len(pred_data) < encoder_length:
                print(f"  ⚠️ 预测数据不足（需要{encoder_length}小时历史），跳过")
                current_idx += test_window
                continue
            
            print(f"  [2/3] 生成TFT预测信号 (预测样本: {len(pred_data):,})...")
            
            # ========== Step 4: 生成TFT预测信号 ==========
            tft_signals = self.predict_tft(tft_model, tft_dataset, pred_data)
            
            if tft_signals is None:
                print(f"  ⚠️ TFT预测失败，跳过")
                current_idx += test_window
                continue
            
            tft_signals['fold'] = fold_num
            tft_signals['date'] = test_start
            all_tft_signals.append(tft_signals)
            
            print(f"  [3/3] ✓ 生成信号成功 ({len(tft_signals)} 条)")
            
            # 输出预测结果和持仓策略
            self.output_signals(fold_num, period_str, tft_signals)
            
            # 滚动到下一个窗口
            current_idx += test_window
        
        # ========== 汇总并保存TFT信号 ==========
        if not all_tft_signals:
            raise ValueError("未生成任何TFT信号！")
        
        all_tft_df = pd.concat(all_tft_signals, ignore_index=True)
        
        # 保存TFT信号（供LGBM使用）
        tft_path = OUTPUT_DIR / 'tft_signals.parquet'
        all_tft_df.to_parquet(tft_path, index=False)
        
        print(f"\n{'=' * 80}")
        print(f"✓ TFT Walk-Forward信号生成完成！")
        print(f"✓ TFT信号: {len(all_tft_df):,} 条")
        print(f"✓ 保存路径: {tft_path}")
        print(f"\n下一步: 运行 train_lgbm_with_tft.py 进行LGBM训练")
        print(f"{'=' * 80}")
        
        return all_tft_df


WalkForwardTrainer = TFTSignalGenerator


def main():
    """主函数"""
    generator = TFTSignalGenerator()
    signals = generator.run_walkforward()
    
    print("\n✓ TFT信号生成完成！")
    print(f"✓ 信号已保存，可供LGBM使用")
    print(f"✓ 下一步运行: python train_lgbm_with_tft.py")


if __name__ == "__main__":
    main()
