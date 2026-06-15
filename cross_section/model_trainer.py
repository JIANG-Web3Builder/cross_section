"""
Phase 3: Model Training 模型训练模块 (V3 REGENESIS)
- 标签构建 (Label Generation)
- 滚动窗口训练 (Walk-Forward Analysis)
- 双模融合: LightGBM + CatBoost Ensemble
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from config import (
    FORWARD_HOURS, LGBM_PARAMS, CATBOOST_PARAMS, ENSEMBLE_WEIGHTS, OUTPUT_DIR,
    ROLLING_TRAIN_SIZE, ROLLING_TEST_SIZE, ROLLING_EMBARGO
)


class RollingPredictor:
    """滚动预测器 - 持有滚动训练产生的预测结果"""
    
    def __init__(self, predictions: pd.Series, feature_names: List[str]):
        """
        Args:
            predictions: 滚动训练产生的预测值 (MultiIndex: timestamp, symbol)
            feature_names: 特征名列表
        """
        self.predictions = predictions
        self.feature_names = feature_names
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        返回已计算的预测值
        注意: 滚动训练中预测已经完成，这里只是返回对应索引的预测值
        """
        # 获取X的索引
        common_idx = X.index.intersection(self.predictions.index)
        
        if len(common_idx) == 0:
            raise ValueError("No matching predictions found for the given data")
        
        return self.predictions.loc[common_idx]


class LossRecorder:
    """记录训练过程中的Loss"""
    
    def __init__(self):
        self.fold_losses = {}  # {fold_num: {'lgbm': [...], 'catboost': [...]}}
    
    def add_fold_loss(self, fold_num: int, model_type: str, losses: List[float]):
        if fold_num not in self.fold_losses:
            self.fold_losses[fold_num] = {}
        self.fold_losses[fold_num][model_type] = losses
    
    def plot_loss_curves(self, save_path=None):
        """绘制平均Loss曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # LGBM Loss
        ax1 = axes[0]
        lgbm_losses = [v.get('lgbm', []) for v in self.fold_losses.values() if 'lgbm' in v]
        if lgbm_losses:
            max_len = max(len(l) for l in lgbm_losses)
            for i, losses in enumerate(lgbm_losses):
                ax1.plot(losses, alpha=0.3, label=f'Fold {i+1}' if i < 3 else None)
            # 计算平均曲线
            padded = [l + [l[-1]]*(max_len-len(l)) if len(l) < max_len else l for l in lgbm_losses]
            avg_loss = np.mean(padded, axis=0)
            ax1.plot(avg_loss, 'r-', linewidth=2, label='Average')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('RMSE')
        ax1.set_title('LightGBM Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # CatBoost Loss
        ax2 = axes[1]
        cat_losses = [v.get('catboost', []) for v in self.fold_losses.values() if 'catboost' in v]
        if cat_losses:
            max_len = max(len(l) for l in cat_losses)
            for i, losses in enumerate(cat_losses):
                ax2.plot(losses, alpha=0.3, label=f'Fold {i+1}' if i < 3 else None)
            padded = [l + [l[-1]]*(max_len-len(l)) if len(l) < max_len else l for l in cat_losses]
            avg_loss = np.mean(padded, axis=0)
            ax2.plot(avg_loss, 'r-', linewidth=2, label='Average')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('RMSE')
        ax2.set_title('CatBoost Training Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Loss curves saved to {save_path}")
        plt.close()


class ModelTrainer:
    """模型训练引擎 - 滚动训练版"""
    
    def __init__(self, panel_data: Dict[str, pd.DataFrame], 
                 factors_normalized: Dict[str, pd.DataFrame],
                 factor_matrix: pd.DataFrame,
                 selected_features: List[str] = None):
        self.returns = panel_data['returns']
        self.universe = panel_data['universe']
        self.market_index = panel_data['market_index']
        self.factors_normalized = factors_normalized
        self.factor_matrix = factor_matrix
        self.preselected_features = selected_features
        
        self.labels: pd.DataFrame = None
        self.selected_features: List[str] = selected_features.copy() if selected_features else None
        self.feature_importance: pd.DataFrame = None
        self.lgbm_model = None
        self.cat_model = None
        self.loss_recorder = LossRecorder()  # Loss记录器
        
    def build_labels(self) -> pd.DataFrame:
        """
        步骤1: 标签构建
        - 计算未来24小时收益率
        - 去大盘Beta
        - 转换为Rank
        """
        print("\n" + "=" * 60)
        print("PHASE 3: MODEL TRAINING")
        print("=" * 60)
        print("Building labels...")
        
        # 未来24小时收益率
        forward_returns = self.returns.shift(-FORWARD_HOURS).rolling(FORWARD_HOURS).sum()
        
        # 大盘指数未来收益
        index_forward_returns = self.market_index['returns'].shift(-FORWARD_HOURS).rolling(FORWARD_HOURS).sum()
        
        # 去Beta后的Alpha收益
        alpha_returns = forward_returns.sub(index_forward_returns, axis=0)
        
        # 截面Rank (0到1之间)
        def rank_row(row):
            valid = row.dropna()
            if len(valid) < 2:
                return row
            ranked = valid.rank(pct=True)
            return ranked.reindex(row.index)
        
        rank_labels = alpha_returns.apply(rank_row, axis=1)
        
        # 只保留票池内的标签
        rank_labels = rank_labels.where(self.universe)
        
        self.labels = rank_labels
        
        # 统计
        valid_labels = rank_labels.stack().dropna()
        print(f"  Forward hours: {FORWARD_HOURS}")
        print(f"  Valid labels count: {len(valid_labels):,}")
        print(f"  Labels mean: {valid_labels.mean():.4f}, std: {valid_labels.std():.4f}")
        
        return rank_labels
    
    def prepare_data(self) -> pd.DataFrame:
        """
        准备完整数据集 (用于滚动训练)
        """
        print("=" * 60)
        print("Preparing full dataset for rolling training...")
        
        # Stack标签
        labels_stacked = self.labels.stack()
        labels_stacked.name = 'label'
        labels_stacked.index.names = ['timestamp', 'symbol']
        
        # 找到共同索引 (内存高效方式)
        common_idx = self.factor_matrix.index.intersection(labels_stacked.index)
        print(f"  Common index size: {len(common_idx):,}")
        
        # 使用loc选择数据而不是join
        data = self.factor_matrix.loc[common_idx].copy()
        data['label'] = labels_stacked.loc[common_idx]
        
        # 只删除label为NaN的行，因子NaN由模型处理
        data = data.dropna(subset=['label'])
        
        self.full_data = data
        available_features = [c for c in data.columns if c != 'label']
        if self.preselected_features:
            available_features = [c for c in self.preselected_features if c in available_features]
        if not available_features:
            raise ValueError("No usable features available for training")
        self.feature_cols = available_features
        
        # 获取时间索引
        self.all_timestamps = data.index.get_level_values('timestamp').unique().sort_values()
        
        print(f"  Total timestamps: {len(self.all_timestamps)}")
        print(f"  Total samples: {len(data):,}")
        print(f"  Features: {len(self.feature_cols)}")
        if self.preselected_features:
            print(f"  Validated feature subset applied: {len(self.feature_cols)}")
        print(f"  Time range: {self.all_timestamps[0]} -> {self.all_timestamps[-1]}")
        
        return data
    
    def run_rolling_training(self) -> Tuple[pd.Series, pd.DatetimeIndex]:
        """
        滚动窗口训练 (Walk-Forward Analysis)
        
        核心逻辑:
        - 训练窗口: 过去6个月 (ROLLING_TRAIN_SIZE)
        - 测试窗口: 未来1个月 (ROLLING_TEST_SIZE)
        - 隔离期: 48小时 (ROLLING_EMBARGO)，防止Label泄露
        - 每月滚动一次，模拟真实世界的"定期更新模型"
        
        Returns:
            all_predictions: 所有测试期的预测值 (pd.Series)
            test_times: 测试集时间范围
        """
        print("=" * 60)
        print(">>> Starting Rolling Window Training (Walk-Forward Analysis)")
        print(f"    Train window: {ROLLING_TRAIN_SIZE} hours (~6 months)")
        print(f"    Test window: {ROLLING_TEST_SIZE} hours (~1 month)")
        print(f"    Embargo: {ROLLING_EMBARGO} hours")
        print("=" * 60)
        
        timestamps = self.all_timestamps
        all_predictions = []
        all_test_times = []
        fold_diagnostics = []  # 过拟合诊断: 每fold的train/test IC
        
        # 从有足够数据的地方开始
        current_idx = ROLLING_TRAIN_SIZE + ROLLING_EMBARGO
        fold_num = 0
        
        while current_idx < len(timestamps):
            fold_num += 1
            
            # 1. 定义时间切片
            train_start_idx = current_idx - ROLLING_EMBARGO - ROLLING_TRAIN_SIZE
            train_end_idx = current_idx - ROLLING_EMBARGO
            
            test_start_idx = current_idx
            test_end_idx = min(current_idx + ROLLING_TEST_SIZE, len(timestamps))
            
            train_start = timestamps[train_start_idx]
            train_end = timestamps[train_end_idx - 1]
            test_start = timestamps[test_start_idx]
            test_end = timestamps[test_end_idx - 1]
            
            print(f"\n  [Fold {fold_num}]")
            print(f"    Training: {train_start} -> {train_end}")
            print(f"    Testing : {test_start} -> {test_end}")
            
            # 2. 切分数据
            train_times = timestamps[train_start_idx:train_end_idx]
            test_times = timestamps[test_start_idx:test_end_idx]
            
            train_mask = self.full_data.index.get_level_values('timestamp').isin(train_times)
            test_mask = self.full_data.index.get_level_values('timestamp').isin(test_times)
            
            X_train = self.full_data.loc[train_mask, self.feature_cols]
            y_train = self.full_data.loc[train_mask, 'label']
            X_test = self.full_data.loc[test_mask, self.feature_cols]
            
            # 3. 分出验证集 (训练集的最后20%作为验证)
            train_ts = X_train.index.get_level_values('timestamp').unique()
            val_split = int(len(train_ts) * 0.8)
            val_times_fold = train_ts[val_split:]
            train_times_fold = train_ts[:val_split]
            
            train_mask_fold = X_train.index.get_level_values('timestamp').isin(train_times_fold)
            val_mask_fold = X_train.index.get_level_values('timestamp').isin(val_times_fold)
            
            X_train_fold = X_train.loc[train_mask_fold]
            y_train_fold = y_train.loc[train_mask_fold]
            X_val_fold = X_train.loc[val_mask_fold]
            y_val_fold = y_train.loc[val_mask_fold]
            
            print(f"    Train samples: {len(X_train_fold):,}, Val samples: {len(X_val_fold):,}")
            
            # 4. 训练模型 (轻量级参数，加速滚动训练)
            lgbm_params_light = LGBM_PARAMS.copy()
            cat_params_light = CATBOOST_PARAMS.copy()
            
            # 训练LGBM - 每50次迭代打印一次
            lgbm_eval_results = {}
            lgbm_model = lgb.LGBMRegressor(**lgbm_params_light)
            lgbm_model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=50),  # 每50次迭代打印
                    lgb.record_evaluation(lgbm_eval_results)
                ]
            )
            # 记录LGBM Loss
            if 'valid_0' in lgbm_eval_results and 'rmse' in lgbm_eval_results['valid_0']:
                self.loss_recorder.add_fold_loss(fold_num, 'lgbm', lgbm_eval_results['valid_0']['rmse'])
            
            # 训练CatBoost - 每50次迭代打印一次
            train_pool = Pool(X_train_fold, y_train_fold, feature_names=self.feature_cols)
            val_pool = Pool(X_val_fold, y_val_fold, feature_names=self.feature_cols)
            
            cat_params_light['verbose'] = 50  # 每50次迭代打印
            cat_model = CatBoostRegressor(**cat_params_light)
            cat_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
            # 记录CatBoost Loss
            cat_evals = cat_model.get_evals_result()
            if 'validation' in cat_evals and 'RMSE' in cat_evals['validation']:
                self.loss_recorder.add_fold_loss(fold_num, 'catboost', cat_evals['validation']['RMSE'])
            
            # 5. 预测测试集
            pred_lgbm = lgbm_model.predict(X_test)
            pred_cat = cat_model.predict(X_test)
            
            # Z-Score融合
            df_preds = pd.DataFrame({
                'lgbm': pred_lgbm,
                'cat': pred_cat
            }, index=X_test.index)
            
            def zscore(x):
                return (x - x.mean()) / (x.std() + 1e-10)
            
            df_preds['lgbm_z'] = df_preds.groupby(level='timestamp')['lgbm'].transform(zscore)
            df_preds['cat_z'] = df_preds.groupby(level='timestamp')['cat'].transform(zscore)
            
            w_lgbm = ENSEMBLE_WEIGHTS['lgbm']
            w_cat = ENSEMBLE_WEIGHTS['catboost']
            final_pred = w_lgbm * df_preds['lgbm_z'] + w_cat * df_preds['cat_z']
            
            all_predictions.append(final_pred)
            all_test_times.extend(test_times.tolist())
            
            # 计算Fold IC (test)
            y_test = self.full_data.loc[test_mask, 'label']
            fold_test_ic = final_pred.groupby(level='timestamp').apply(
                lambda x: x.corr(y_test.loc[x.index])
            ).mean()
            
            # 计算Fold IC (train) — 过拟合诊断
            train_pred_lgbm = lgbm_model.predict(X_train_fold)
            train_pred_cat = cat_model.predict(X_train_fold)
            train_preds_df = pd.DataFrame({
                'lgbm': train_pred_lgbm, 'cat': train_pred_cat
            }, index=X_train_fold.index)
            train_preds_df['lgbm_z'] = train_preds_df.groupby(level='timestamp')['lgbm'].transform(zscore)
            train_preds_df['cat_z'] = train_preds_df.groupby(level='timestamp')['cat'].transform(zscore)
            train_final = w_lgbm * train_preds_df['lgbm_z'] + w_cat * train_preds_df['cat_z']
            fold_train_ic = train_final.groupby(level='timestamp').apply(
                lambda x: x.corr(y_train_fold.loc[x.index])
            ).mean()
            
            overfit_ratio = fold_train_ic / (fold_test_ic + 1e-10) if pd.notna(fold_test_ic) else np.nan
            fold_diagnostics.append({
                'fold': fold_num,
                'train_ic': fold_train_ic,
                'test_ic': fold_test_ic,
                'overfit_ratio': overfit_ratio,
            })
            print(f"    Train IC: {fold_train_ic:.4f}  Test IC: {fold_test_ic:.4f}  "
                  f"Overfit ratio: {overfit_ratio:.2f}x")
            
            # 6. 滚动到下一个窗口
            current_idx += ROLLING_TEST_SIZE
            
            # 保存最后一个Fold的模型用于后续分析
            if current_idx >= len(timestamps):
                self.lgbm_model = lgbm_model
                self.cat_model = cat_model
        
        # 合并所有预测
        all_predictions = pd.concat(all_predictions)
        all_test_times = pd.DatetimeIndex(sorted(set(all_test_times)))
        
        self.test_times = all_test_times
        self.rolling_predictions = all_predictions
        
        print(f"\n>>> Rolling Training Completed")
        print(f"    Total folds: {fold_num}")
        print(f"    Test period: {all_test_times[0]} -> {all_test_times[-1]}")
        print(f"    Total predictions: {len(all_predictions):,}")
        
        # 计算整体IC
        y_all_test = self.full_data.loc[all_predictions.index, 'label']
        overall_ic = all_predictions.groupby(level='timestamp').apply(
            lambda x: x.corr(y_all_test.loc[x.index])
        )
        print(f"    Overall Rank IC: {overall_ic.mean():.4f} (std: {overall_ic.std():.4f})")
        print(f"    Overall ICIR: {overall_ic.mean() / overall_ic.std():.4f}")
        
        # 过拟合诊断汇总
        if fold_diagnostics:
            diag_df = pd.DataFrame(fold_diagnostics)
            diag_df.to_csv(OUTPUT_DIR / 'fold_overfit_diagnostics.csv', index=False)
            avg_train = diag_df['train_ic'].mean()
            avg_test = diag_df['test_ic'].mean()
            avg_ratio = diag_df['overfit_ratio'].mean()
            print(f"\n    === OVERFIT DIAGNOSTICS ===")
            print(f"    Avg Train IC: {avg_train:.4f}")
            print(f"    Avg Test IC:  {avg_test:.4f}")
            print(f"    Avg Overfit Ratio: {avg_ratio:.2f}x")
            if avg_ratio > 3.0:
                print(f"    ⚠️ WARNING: High overfit ratio ({avg_ratio:.1f}x). "
                      f"Consider stronger regularization or fewer features.")
            elif avg_ratio > 2.0:
                print(f"    ⚠️ CAUTION: Moderate overfit ratio ({avg_ratio:.1f}x).")
            else:
                print(f"    ✅ Overfit ratio acceptable ({avg_ratio:.1f}x).")
        
        return all_predictions, all_test_times
    
    def analyze_feature_importance(self) -> pd.DataFrame:
        """
        特征重要性分析 (基于最后一个Fold的模型)
        """
        print("=" * 60)
        print("Analyzing feature importance (from last fold)...")
        
        if self.lgbm_model is None or self.cat_model is None:
            print("  Warning: Models not available for importance analysis")
            return None
        
        # 归一化 LGBM 重要性
        lgbm_imp = self.lgbm_model.feature_importances_.astype(float)
        lgbm_imp = lgbm_imp / lgbm_imp.sum()
        
        # 归一化 CatBoost 重要性
        cat_imp = self.cat_model.get_feature_importance().astype(float)
        cat_imp = cat_imp / cat_imp.sum()
        
        # 加权平均
        avg_imp = (lgbm_imp + cat_imp) / 2
        
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'lgbm_importance': lgbm_imp,
            'catboost_importance': cat_imp,
            'ensemble_importance': avg_imp
        }).sort_values('ensemble_importance', ascending=False)
        
        importance_df['importance_pct'] = importance_df['ensemble_importance'] / importance_df['ensemble_importance'].sum() * 100
        
        self.feature_importance = importance_df
        
        print("\n  Top 20 features by ensemble importance:")
        print(importance_df[['feature', 'lgbm_importance', 'catboost_importance', 'ensemble_importance']].head(20).to_string(index=False))
        
        # 识别低重要性特征
        low_importance = importance_df[importance_df['importance_pct'] < 1.0]
        print(f"\n  Features with importance < 1%: {len(low_importance)}")
        
        return importance_df
    
    def run_pipeline(self) -> Tuple['RollingPredictor', List[str], pd.DatetimeIndex]:
        """运行模型训练流水线 (滚动训练版)
        
        Returns:
            rolling_predictor: 滚动预测器 (包含预测结果)
            selected_features: 选择的特征列表
            test_times: 测试集时间范围 (用于回测)
        """
        # 构建标签
        self.build_labels()
        
        # 准备数据
        self.prepare_data()
        
        # 执行滚动训练
        predictions, test_times = self.run_rolling_training()
        
        # 特征重要性分析 (基于最后一个Fold)
        self.analyze_feature_importance()
        
        # 绘制Loss曲线
        print("=" * 60)
        print("Plotting loss curves...")
        self.loss_recorder.plot_loss_curves(save_path=OUTPUT_DIR / 'loss_curves.png')
        
        # 创建滚动预测器 (用于策略回测)
        rolling_predictor = RollingPredictor(predictions, self.feature_cols)
        
        # 保存特征列表
        self.selected_features = list(self.feature_cols)
        
        # 保存结果
        self._save_rolling_results(predictions)
        
        print("\n" + "=" * 60)
        print("Phase 3 completed! (Rolling Walk-Forward Training)")
        print("=" * 60)
        
        return rolling_predictor, self.selected_features, self.test_times
    
    def _save_rolling_results(self, predictions: pd.Series):
        """保存滚动训练结果"""
        # 保存预测结果
        predictions.to_frame('prediction').to_parquet(OUTPUT_DIR / 'rolling_predictions.parquet')
        
        # 保存特征列表
        pd.Series(self.feature_cols).to_csv(
            OUTPUT_DIR / 'selected_features.csv', index=False, header=False
        )
        
        # 保存特征重要性
        if self.feature_importance is not None:
            self.feature_importance.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)
        
        print(f"  Rolling results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    from data_engineering import DataEngine
    from factor_engineering import FactorEngine
    
    # Phase 1
    data_engine = DataEngine()
    panel_data = data_engine.run_pipeline()
    
    # Phase 2
    factor_engine = FactorEngine(panel_data)
    factors_normalized, factor_matrix = factor_engine.run_pipeline()
    
    # Phase 3
    trainer = ModelTrainer(panel_data, factors_normalized, factor_matrix)
    rolling_predictor, selected_features, test_times = trainer.run_pipeline()
