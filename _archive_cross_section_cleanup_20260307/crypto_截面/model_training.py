"""
Phase 3: Model Training 模型训练模块
- 标签构建 (Label Generation)
- LightGBM训练
- 特征筛选与重要性分析
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from config import (
    FORWARD_HOURS, MODEL_PARAMS, TRAIN_RATIO, OUTPUT_DIR
)


class ModelTrainer:
    """模型训练引擎"""
    
    def __init__(self, panel_data: Dict[str, pd.DataFrame], 
                 factors_normalized: Dict[str, pd.DataFrame],
                 factor_matrix: pd.DataFrame):
        self.returns = panel_data['returns']
        self.universe = panel_data['universe']
        self.market_index = panel_data['market_index']
        self.factors_normalized = factors_normalized
        self.factor_matrix = factor_matrix
        
        self.labels: pd.DataFrame = None
        self.model: lgb.LGBMRegressor = None
        self.feature_importance: pd.DataFrame = None
        self.selected_features: List[str] = None
        
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
    
    def prepare_train_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        准备训练数据
        按时间顺序切分，70%训练，30%验证
        """
        print("=" * 60)
        print("Preparing train/validation data...")
        
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
        # 对因子填充0（LightGBM可以处理NaN，但为安全起见填充）
        feature_cols = [c for c in data.columns if c != 'label']
        data[feature_cols] = data[feature_cols].fillna(0)
        
        # 获取时间索引
        timestamps = data.index.get_level_values('timestamp').unique().sort_values()
        
        # 按时间切分
        split_idx = int(len(timestamps) * TRAIN_RATIO)
        train_times = timestamps[:split_idx]
        val_times = timestamps[split_idx:]
        
        # 保存测试集时间范围 (回测只在这个范围内进行)
        self.test_times = val_times
        self.train_times = train_times
        
        print(f"  Train period: {train_times[0]} to {train_times[-1]}")
        print(f"  Validation/Test period: {val_times[0]} to {val_times[-1]}")
        print(f"  *** 回测将只在测试集时间段进行 ***")
        
        # 分离特征和标签
        feature_cols = [c for c in data.columns if c != 'label']
        
        train_mask = data.index.get_level_values('timestamp').isin(train_times)
        val_mask = data.index.get_level_values('timestamp').isin(val_times)
        
        X_train = data.loc[train_mask, feature_cols]
        y_train = data.loc[train_mask, 'label']
        X_val = data.loc[val_mask, feature_cols]
        y_val = data.loc[val_mask, 'label']
        
        print(f"  Train samples: {len(X_train):,}")
        print(f"  Validation samples: {len(X_val):,}")
        print(f"  Features: {len(feature_cols)}")
        
        return X_train, y_train, X_val, y_val
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame, y_val: pd.Series) -> lgb.LGBMRegressor:
        """
        步骤2: 训练LightGBM模型
        """
        print("=" * 60)
        print("Training LightGBM model...")
        
        # 创建模型
        model = lgb.LGBMRegressor(**MODEL_PARAMS)
        
        # 训练
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='l2',
        )
        
        # 预测
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # 评估
        train_mse = mean_squared_error(y_train, y_train_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        # 计算Rank IC
        val_data = pd.DataFrame({
            'actual': y_val.values,
            'predicted': y_val_pred
        }, index=y_val.index)
        
        rank_ic = val_data.groupby(level='timestamp').apply(
            lambda x: x['actual'].corr(x['predicted'])
        )
        
        print(f"  Train MSE: {train_mse:.6f}, R2: {train_r2:.4f}")
        print(f"  Val MSE: {val_mse:.6f}, R2: {val_r2:.4f}")
        print(f"  Val Rank IC mean: {rank_ic.mean():.4f}")
        print(f"  Val Rank IC std: {rank_ic.std():.4f}")
        print(f"  Val Rank ICIR: {rank_ic.mean() / rank_ic.std():.4f}")
        
        self.model = model
        
        return model
    
    def analyze_feature_importance(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """
        步骤3: 特征重要性分析
        """
        print("=" * 60)
        print("Analyzing feature importance...")
        
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        })
        importance_df['importance_pct'] = importance_df['importance'] / importance_df['importance'].sum() * 100
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        self.feature_importance = importance_df
        
        print("  Top 20 features by importance:")
        print(importance_df.head(20).to_string(index=False))
        
        # 识别低重要性特征
        low_importance = importance_df[importance_df['importance_pct'] < 1.0]
        print(f"\n  Features with importance < 1%: {len(low_importance)}")
        
        return importance_df
    
    def select_features(self, min_importance_pct: float = 1.0, max_features: int = 35) -> List[str]:
        """
        特征筛选
        - 剔除重要性极低的因子
        - 保留前N个因子
        """
        print("=" * 60)
        print("Selecting features...")
        
        # 筛选重要性足够的特征
        selected = self.feature_importance[
            self.feature_importance['importance_pct'] >= min_importance_pct
        ]['feature'].tolist()
        
        # 限制最大特征数
        if len(selected) > max_features:
            selected = selected[:max_features]
        
        self.selected_features = selected
        
        print(f"  Selected {len(selected)} features")
        print(f"  Features: {selected}")
        
        return selected
    
    def calc_factor_correlation(self, X: pd.DataFrame) -> pd.DataFrame:
        """计算因子相关性矩阵"""
        print("=" * 60)
        print("Calculating factor correlations...")
        
        corr_matrix = X.corr()
        
        # 找出高相关因子对
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.7:
                    high_corr_pairs.append({
                        'factor1': corr_matrix.columns[i],
                        'factor2': corr_matrix.columns[j],
                        'correlation': corr
                    })
        
        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs)
            high_corr_df = high_corr_df.sort_values('correlation', ascending=False)
            print(f"  High correlation pairs (|corr| > 0.7):")
            print(high_corr_df.head(10).to_string(index=False))
        else:
            print("  No highly correlated factor pairs found")
        
        return corr_matrix
    
    def retrain_with_selected_features(self, X_train: pd.DataFrame, y_train: pd.Series,
                                       X_val: pd.DataFrame, y_val: pd.Series) -> lgb.LGBMRegressor:
        """使用筛选后的特征重新训练"""
        print("=" * 60)
        print("Retraining with selected features...")
        
        X_train_selected = X_train[self.selected_features]
        X_val_selected = X_val[self.selected_features]
        
        # 创建新模型
        model = lgb.LGBMRegressor(**MODEL_PARAMS)
        
        model.fit(
            X_train_selected, y_train,
            eval_set=[(X_val_selected, y_val)],
            eval_metric='l2',
        )
        
        # 评估
        y_val_pred = model.predict(X_val_selected)
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        # 计算Rank IC
        val_data = pd.DataFrame({
            'actual': y_val.values,
            'predicted': y_val_pred
        }, index=y_val.index)
        
        rank_ic = val_data.groupby(level='timestamp').apply(
            lambda x: x['actual'].corr(x['predicted'])
        )
        
        print(f"  Retrained Val MSE: {val_mse:.6f}, R2: {val_r2:.4f}")
        print(f"  Retrained Val Rank IC mean: {rank_ic.mean():.4f}")
        print(f"  Retrained Val Rank ICIR: {rank_ic.mean() / rank_ic.std():.4f}")
        
        self.model = model
        
        return model
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """预测"""
        if self.selected_features:
            X = X[self.selected_features]
        return pd.Series(self.model.predict(X), index=X.index)
    
    def save_model(self, path: Optional[str] = None):
        """保存模型"""
        import joblib
        if path is None:
            path = OUTPUT_DIR / 'lgbm_model.pkl'
        # 使用joblib保存，避免中文路径编码问题
        joblib.dump(self.model, path)
        print(f"  Model saved to {path}")
        
        # 保存特征重要性
        self.feature_importance.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)
        
        # 保存选择的特征
        pd.Series(self.selected_features).to_csv(
            OUTPUT_DIR / 'selected_features.csv', index=False, header=False
        )
    
    def run_pipeline(self) -> Tuple[lgb.LGBMRegressor, List[str], pd.DatetimeIndex]:
        """运行模型训练流水线
        
        Returns:
            model: 训练好的模型
            selected_features: 选择的特征列表
            test_times: 测试集时间范围 (用于回测)
        """
        # 构建标签
        self.build_labels()
        
        # 准备数据
        X_train, y_train, X_val, y_val = self.prepare_train_data()
        
        # 训练模型
        self.train_model(X_train, y_train, X_val, y_val)
        
        # 特征分析
        self.analyze_feature_importance(X_train)
        self.calc_factor_correlation(X_train)
        
        # 特征筛选
        self.select_features()
        
        # 重新训练
        self.retrain_with_selected_features(X_train, y_train, X_val, y_val)
        
        # 保存
        self.save_model()
        
        print("\n" + "=" * 60)
        print("Phase 3 completed!")
        print("=" * 60)
        
        return self.model, self.selected_features, self.test_times


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
    model, selected_features = trainer.run_pipeline()
