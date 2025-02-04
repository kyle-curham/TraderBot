import pandas as pd
import numpy as np
np.float=float
from fredapi import Fred
from finrl.meta.data_processors.processor_alpaca import AlpacaProcessor as FinrlAlpaca
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Updated FRED series with new frequency aliases to address deprecation warnings
FRED_SERIES = {
    'GDPC1': {'name': 'Real GDP', 'buffer': '3QE'},          # Quarterly (3 quarters buffer)
    'CPIAUCSL': {'name': 'CPI Inflation', 'buffer': '3ME'},  # Monthly (3 months buffer)
    'UNRATE': {'name': 'Unemployment Rate', 'buffer': '3ME'},
    'FEDFUNDS': {'name': 'Fed Funds Rate', 'buffer': '3ME'},
    'SP500': {'name': 'S&P 500', 'buffer': '1ME'},          # Daily but needs buffer for holidays
    'VIXCLS': {'name': 'VIX Volatility', 'buffer': '1ME'},
    'DGS10': {'name': '10-Year Treasury Yield', 'buffer': '1ME'},
    'INDPRO': {'name': 'Industrial Production', 'buffer': '3ME'},
    'RSXFSN': {'name': 'Retail Sales', 'buffer': '3ME'}
}

class TradingDataProcessor(FinrlAlpaca):
    def __init__(self, 
                 credentials_path: str = "credentials.json",
                 base_frequency: str = "1D"):
        credentials_file = Path(credentials_path)

        if not credentials_file.exists():
            raise FileNotFoundError(f"Credentials file not found at {credentials_path}")
        
        try:
            with open(credentials_file) as f:
                credentials = json.load(f)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in credentials file")
        
        required_keys = {"API_KEY_ALPACA", "API_SECRET_ALPACA", 
                        "API_BASE_URL_ALPACA", "API_KEY_FRED"}
        
        if not required_keys.issubset(credentials):
            missing = required_keys - credentials.keys()
            raise KeyError(f"Missing required credentials: {missing}")

        super().__init__(
            credentials["API_KEY_ALPACA"],
            credentials["API_SECRET_ALPACA"],
            credentials["API_BASE_URL_ALPACA"]
        )

        # get fred data
        self.fred = Fred(api_key=credentials["API_KEY_FRED"])
        self.tech_indicator_list = []
        self.macro_indicators = [config['name'] for config in FRED_SERIES.values()]
        self.base_frequency = base_frequency

        self.scalers = {
            'price': StandardScaler(),
            'macro': StandardScaler()
        }

    def download_data(
        self, 
        ticker_list: list[str],
        start_date: str,
        end_date: str,
        time_interval: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Raw data collection without processing"""
        price_df = super().download_data(ticker_list, start_date, end_date, time_interval)
        
        # Get macro data (already buffered and forward-filled)
        macro_df = self._get_macro_data(start_date, end_date)
        
        return price_df, macro_df

    def process_data(self, raw_data: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
        """Main processing pipeline"""
        price_df, macro_df = raw_data
        
        # Clean and merge
        merged = self._clean_and_merge(price_df, macro_df)
        
        # Feature engineering
        processed = self._create_features(merged)
        
        # Handle missing values with tailored fill methods per feature type
        processed = self._handle_missing_values(processed)
        
        # Normalization
        normalized = self._normalize_features(processed)

        # Create time index grouped by ticker
        normalized['time_idx'] = normalized.groupby('tic').cumcount()
        
        return normalized

    def _clean_and_merge(self, price_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
        """Merge and clean raw datasets with consistent timezone information.

        The price_df already contains tz-aware timestamps. We extract its timezone and then
        localize the macro_df timestamps accordingly. After that, we merge the two on the
        timestamp column.
        """
        # Use price_df as-is (tz-aware)
        price_clean = price_df.copy()

        # Extract timezone from the price data's timestamp
        # (Assumes all price timestamps have the same tz.)
        tz_info = price_clean['timestamp'].iloc[0].tzinfo

        # Prepare macro data: rename 'date' to 'timestamp' and convert to datetime
        macro_clean = macro_df.rename(columns={'date': 'timestamp'})
        macro_clean['timestamp'] = pd.to_datetime(macro_clean['timestamp'])

        # Localize macro timestamps if they are tz-naive
        if macro_clean['timestamp'].dt.tz is None:
            macro_clean['timestamp'] = macro_clean['timestamp'].dt.tz_localize(tz_info)

        # Merge on timestamp with forward fill/backward fill for any missing macro data
        merged = price_clean.merge(
             macro_clean,
             on='timestamp',
             how='left'
        ).ffill().bfill()

        return merged

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete feature engineering pipeline, including price, volatility, volume,
        turbulence, macro derivatives, and technical indicators.

        This method sequentially applies helper methods to generate a wide range of features and then
        calls add_technical_indicator to enrich the dataset with additional technical indicators.
        """
        # Price-based features
        df = self._calculate_returns(df)
        
        # Volatility features
        df = self._calculate_volatility(df)
        
        # Volume-related features
        df = self._calculate_volume_features(df)
        
        # Macro-based features: turbulence from macro indicators
        df = self._calculate_turbulence(df)
        
        # Macro derivatives (for daily macros and price-based derivative features)
        df = self._add_macro_derivatives(df)
        
        # Add additional price technical indicators that complement the above.
        # Choosing indicators from the actual TECHNICAL_INDICATORS_LIST:
        # "macd": trend/momentum measure,
        # "rsi_30": momentum measure,
        # "close_30_sma": short-term trend,
        # "close_60_sma": medium-term trend
        tech_indicator_list = [
            "macd",
            "rsi_30",
            "close_30_sma",
            "close_60_sma"
        ]
        df = self.add_technical_indicator(df, tech_indicator_list)
        
        return df

    def _calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate normalized returns with PPO-specific features and diagnostics"""
        # 1. Calculate log returns with NaN tracking
        df['log_return'] = np.log(df['close'] / df.groupby('tic')['close'].shift(1))
        
        # 2. Add candlestick body features (PPO benefits from price action context)
        df['hl_spread'] = (df['high'] - df['low']) / df['low']  # Normalized high-low spread
        df['co_change'] = (df['close'] - df['open']) / df['open']  # Close-open change
        
        # 3. Market-relative features (important for policy networks)
        df['market_return'] = (
            df.groupby('timestamp')['log_return']
            .transform('mean')
            .fillna(0)
        )
        df['alpha'] = df['log_return'] - df['market_return']
        
        return df

    def _calculate_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate multiple volatility measures with proper annualization and per-ticker min-max scaling.
        
        This method computes several volatility metrics and scales them on a per-ticker basis using min-max normalization
        """
        # 1. Calculate base volatility metrics
        group = df.groupby('tic')['log_return']
        

        # Annualized rolling volatility (sqrt(T) correction)
        df['vol_21d_ann'] = group.transform(
            lambda x: x.rolling(21, min_periods=1).std() * np.sqrt(252)
        )
        
        # Realized volatility (sum of squared returns)
        df['realized_vol_5d'] = group.transform(
            lambda x: x.rolling(5).apply(lambda s: np.sqrt(np.sum(s**2)) * np.sqrt(252/5))
        )
        
        # Parkinson volatility (high-low estimator)
        df['parkinson_vol'] = group.transform(
            lambda x: np.sqrt(
                1 / (4 * np.log(2)) *
                (np.log(df['high'] / df['low'])**2).rolling(21, min_periods=1).mean()
            ) * np.sqrt(252)
        )
        
        # 2. Volatility regime features: volatility ratio (short-term vs long-term)
        df['vol_ratio_5d_21d'] = df['realized_vol_5d'] / (df['vol_21d_ann'] + 1e-8)
        
        # 3. Per-ticker min-max scaling for each volatility metric
        vol_cols = ['vol_21d_ann', 'realized_vol_5d', 'parkinson_vol', 'vol_ratio_5d_21d']
        df[vol_cols] = df.groupby('tic')[vol_cols].transform(

            lambda x: (x - x.min()) / ((x.max() - x.min()) + 1e-8)
        )
        
        return df

    def _calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-related features optimized for RL trading using per-ticker min-max scaling.
        
        This method computes several features from volume data and scales them for each ticker using min-max normalization.
        """
        group = df.groupby('tic')['volume']
        

        # 1. Core normalized volume feature
        df['volume_z'] = group.transform(
            lambda x: (x - x.rolling(21, min_periods=1).mean()) / 
                      (x.rolling(21, min_periods=1).std() + 1e-8)
        )
        
        # 2. Relative volume indicator
        df['relative_volume'] = group.transform(
            lambda x: (x / x.rolling(21, min_periods=1).mean()) - 1
        )
        
        # 3. Volume momentum features
        df['volume_momentum_5d'] = group.transform(
            lambda x: x.pct_change(5).fillna(0)
        )
        
        # 4. Volume volatility
        df['volume_volatility_21d'] = group.transform(
            lambda x: x.rolling(21, min_periods=1).std().fillna(0)
        )
        
        # 5. Apply per-ticker min-max scaling to the volume features
        volume_features = ['volume_z', 'relative_volume', 'volume_momentum_5d', 'volume_volatility_21d']
        df[volume_features] = df.groupby('tic')[volume_features].transform(
            lambda x: (x - x.min()) / ((x.max() - x.min()) + 1e-8)
        )

        
        return df

    def _calculate_turbulence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate a continuous market turbulence metric using a rolling Mahalanobis distance 
        computed on macroeconomic indicators, then map it to a value between 0 and 1.

        For each unique timestamp, the method computes the Mahalanobis distance of the current 
        (average) macro indicator values relative to a historical window. After computing this 
        distance for all relevant dates, we perform a min-max normalization over non-NaN values, 
        thereby mapping the turbulence metric to a continuous scale between 0 and 1.

        Args:
            df (pd.DataFrame): DataFrame containing at least a 'timestamp' column and the macro 
                               indicator columns as specified by self.macro_indicators.
        
        Returns:
            pd.DataFrame: The input DataFrame with a new 'turbulence' column containing the normalized 
                          turbulence metric.
        """
        # Ensure that macro indicator columns exist
        macro_cols = self.macro_indicators
        if not set(macro_cols).issubset(df.columns):
            raise ValueError("Not all required macro indicator columns are present in the dataframe.")
        
        # Get a sorted list of unique timestamps from the DataFrame
        unique_dates = df['timestamp'].drop_duplicates().sort_values().reset_index(drop=True)

        # Set rolling window parameters
        rolling_window: int = 252  # roughly one trading year
        min_history: int = 30      # require at least 30 days of history for robust calculations

        turbulence_by_date: dict = {}

        # Loop over unique timestamps to calculate Mahalanobis distance
        for idx, cur_date in unique_dates.items():
            # Define the historical window (all days strictly before the current date in the rolling window)
            start_idx = max(0, idx - rolling_window)
            history_dates = unique_dates.iloc[start_idx:idx]
            
            if len(history_dates) < min_history:
                turbulence_by_date[cur_date] = np.nan
                continue
            
            # Use historical macro data: average macro values over the rolling window
            history_data = df[df['timestamp'].isin(history_dates)][macro_cols]
            
            # Skip if insufficient data after filtering
            if history_data.shape[0] < min_history:
                turbulence_by_date[cur_date] = np.nan
                continue

            # Compute historical mean and covariance matrix
            mu: np.ndarray = history_data.mean().values
            cov: np.ndarray = history_data.cov().values

            # Compute pseudo-inverse of covariance matrix for stability
            inv_cov = np.linalg.pinv(cov)

            # Current day's macro snapshot: average across rows for the current date
            current_data = df[df['timestamp'] == cur_date][macro_cols].mean().values

            # Calculate the Mahalanobis distance (the turbulence metric)
            diff = current_data - mu
            distance = np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))
            turbulence_by_date[cur_date] = distance

        # Min-max normalize the turbulence metrics to a [0, 1] scale over non-NaN values
        turbulence_values = np.array([v for v in turbulence_by_date.values() if not np.isnan(v)])
        min_t = turbulence_values.min() if turbulence_values.size > 0 else 0
        max_t = turbulence_values.max() if turbulence_values.size > 0 else 1

        for date, value in turbulence_by_date.items():
            if not np.isnan(value):
                if max_t - min_t == 0:
                    turbulence_by_date[date] = 0.0
                else:
                    turbulence_by_date[date] = (value - min_t) / (max_t - min_t)

        # Map the normalized turbulence metric back onto the original DataFrame
        df = df.copy()
        df['turbulence'] = df['timestamp'].map(turbulence_by_date)

        return df

    def _add_macro_derivatives(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate percentage change derivatives for select daily macro indicators.

        This method adds derivative columns only for the macro indicators with a daily frequency 
        (as indicated by a '1M' buffer in the FRED_SERIES mapping).
        
        Args:
            df (pd.DataFrame): DataFrame containing the original features.
            
        Returns:
            pd.DataFrame: Modified DataFrame that includes derivative columns.
        """
        # Process macro derivatives only for daily macros
        daily_macro_names = [
            cfg['name'] for key, cfg in FRED_SERIES.items() if cfg['buffer'] in ('1M', '1ME')
        ]
        for col in daily_macro_names:
            if col in df.columns:
                returns: pd.Series = df[col].pct_change().fillna(0)
                df[f'{col}_delta'] = returns
        
        return df

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize selected features while leaving pre-normalized engineered features unchanged.

        Normalization steps:
          1. Price columns ("open", "high", "low", "close") are normalized per tic using min-max scaling.
             The close min and max are stored for potential denormalization.
          2. Volume columns ("volume", "vwap") are normalized per tic using min-max scaling.
          3. The turbulence column is globally normalized using min-max scaling.
          4. Macro data (e.g. "real gdp", "cpi inflation", "unemployment rate", "fed funds rate",
             "s&p 500", "vix volatility", "10-year treasury yield", "industrial production", "retail sales"),
             technical indicators (e.g. "macd", "rsi_30", "close_30_sma", "close_60_sma") and
             "trade_count" are normalized globally via min-max scaling.
          5. Remaining numeric features are winsorized (clipped to the 1st-99th percentile) to limit
             the influence of extreme outliers.
        """
        df = df.copy()

        # (1) Normalize OHLC price columns on a per-tic basis.
        price_cols = ["open", "high", "low", "close"]
        ohlc_stats = df.groupby("tic")["close"].agg(["min", "max"]).rename(
            columns={"min": "close_min", "max": "close_max"}
        )
        df = df.merge(ohlc_stats, on="tic", how="left")
        df[price_cols] = df.groupby("tic")[price_cols].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
        )

        # (2) Normalize volume columns on a per-tic basis.
        volume_cols = ["volume", "vwap"]
        df[volume_cols] = df.groupby("tic")[volume_cols].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
        )

        # (3) Normalize the turbulence metric globally.
        if "turbulence" in df.columns:
            t_min = df["turbulence"].min()
            t_max = df["turbulence"].max()
            df["turbulence"] = (df["turbulence"] - t_min) / (t_max - t_min + 1e-8)

        # (4) Global min-max normalization for macro features and technical indicators.
        global_cols = [
            "Real GDP", "CPI Inflation", "Unemployment Rate", "Fed Funds Rate",
            "S&P 500", "VIX Volatility", "10-Year Treasury Yield",
            "Industrial Production", "Retail Sales",
            "S&P 500_delta", "VIX Volatility_delta", "10-Year Treasury Yield_delta",
            "macd", "rsi_30", "close_30_sma", "close_60_sma",
            "turbulence"
        ]
        for col in global_cols:
            if col in df.columns:
                col_min = df[col].min()
                col_max = df[col].max()
                df[col] = (df[col] - col_min) / (col_max - col_min + 1e-8)

        # (5) Normalize trade_count per tic.
        if "trade_count" in df.columns:
            df["trade_count"] = df.groupby("tic")["trade_count"].transform(
                lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
            )

        # (6) For all remaining numeric features that are not already normalized,
        # apply winsorization (clip to 1st-99th percentile) to reduce the influence of outliers.
        skip_cols = set(price_cols + volume_cols + ["turbulence", "timestamp", "tic", "close_min", "close_max"] + global_cols + ["trade_count"])
        other_numeric_cols = [
            col for col in df.select_dtypes(include=[np.number]).columns if col not in skip_cols
        ]
        for col in other_numeric_cols:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower, upper)
        
        if df.isna().any().any():
            raise ValueError("NaNs detected after normalization")
        
        return df

    def add_technical_indicator(
        self, 
        df: pd.DataFrame, 
        tech_indicator_list: list[str]
    ) -> pd.DataFrame:
        """Extend with FINRL's technical indicators"""
        # Call parent method first
        df = super().add_technical_indicator(df, tech_indicator_list)

        self.tech_indicator_list = tech_indicator_list
        return df

    def _get_macro_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch macroeconomic data with buffer periods"""
        macro_data = []
        
        for series_id, config in FRED_SERIES.items():
            # Calculate buffer start date
            buffer_start = (pd.to_datetime(start_date) - 
                           pd.tseries.frequencies.to_offset(config['buffer'])).strftime('%Y-%m-%d')
            
            try:
                # Fetch data with buffer period
                series = self.fred.get_series(series_id, buffer_start, end_date)
                if series.empty:
                    raise ValueError(f"No data available for {series_id}")
                
                # Resample to base frequency and forward fill
                series = series.resample(self.base_frequency).ffill().bfill()
                # Ensure no NaNs using direct ffill and bfill methods
                if series.isna().any():
                    series = series.ffill().bfill()
                macro_data.append(series.rename(config['name']).loc[start_date:end_date])
                
            except Exception as e:
                raise ValueError(f"Failed to fetch {config['name']}: {str(e)}") from e

        return pd.concat(macro_data, axis=1).reset_index().rename(columns={'index': 'date'})

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using tailored strategies for different feature types.

        For features derived from percentage changes or derivatives (i.e., columns with 'delta' or 
        "return"), missing values are filled with 0—assuming no change when data is not available. For 
        the remaining features, a per-tic forward fill followed by a backward fill is applied to preserve 
        the time-series continuity.

        Args:
            df (pd.DataFrame): DataFrame containing engineered features.

        Returns:
            pd.DataFrame: DataFrame with missing values handled.
        """
        # Sort by tic and timestamp to ensure proper sequential ordering
        df = df.sort_values(["tic", "timestamp"])

        # Apply filling per tic group
        def fill_group(g: pd.DataFrame) -> pd.DataFrame:
            for col in g.columns:
                # Skip non-numeric or key columns
                if col in ["tic", "timestamp"]:
                    continue
                # For derivative and change features, assume no change by filling with 0
                if "delta" in col or "return" in col:
                    g[col] = g[col].fillna(0)
                else:
                    # For other features, forward-fill then back-fill any remaining missing values.
                    g[col] = g[col].ffill().bfill()
            return g

        df = df.groupby("tic", group_keys=False).apply(fill_group).reset_index(drop=True)
        return df
