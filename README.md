# Machine learning - RidgeClassifier (log futures close prices)

This trading strategy is designed for the [Quantiacs](https://quantiacs.com/contest) platform, which hosts competitions
for trading algorithms. Detailed information about the competitions is available on
the [official Quantiacs website](https://quantiacs.com/contest).

## How to Run the Strategy

### In an Online Environment

The strategy can be executed in an online environment using Jupiter or JupiterLab on
the [Quantiacs personal dashboard](https://quantiacs.com/personalpage/homepage). To do this, clone the template in your
personal account.

### In a Local Environment

To run the strategy locally, you need to install the [Quantiacs Toolbox](https://github.com/quantiacs/toolbox).

## Strategy Overview

**Strategy idea**: We will open cryptofutures positions as predicted by the RidgeClassifier.

**Features for learning** - the logarithm of closing prices for the last 18 days of the futures "F_O", "F_LN", "F_KC"

This strategy leverages machine learning, specifically using a RidgeClassifier, to predict the positions in cryptocurrency futures based on the logarithm of the closing prices of certain futures markets over the past 18 days. The selected futures markets for generating features are "F_O", "F_LN", and "F_KC". The strategy involves loading both futures and cryptocurrency futures data, preprocessing this data to extract relevant features, and then using these features to train a RidgeClassifier. The classifier aims to predict whether to open a position in cryptofutures based on the direction (up or down) it predicts for the market.

The preprocessing step includes taking the logarithm of the closing prices to remove any trend and fill missing values, ensuring a consistent dataset for the model. This model predicts daily weights for cryptofutures positions, attempting to capitalize on the anticipated market movement direction.

To implement this strategy, the notebook includes a function to load and preprocess data (load_data), a function to construct the dataset for a single time step (build_data_for_one_step), and the main prediction function (predict_weights) that utilizes the machine learning model to determine position weights. The backtesting framework provided (qnbt.backtest) is used to evaluate the strategy's performance over time, starting from January 1, 2014, with a lookback period of 18 days to reflect the features' timeframe.

This approach exemplifies the integration of machine learning into trading strategies, showcasing how historical price data can be utilized to make informed decisions in the cryptofutures market.

```python
import xarray as xr

import qnt.backtester as qnbt
import qnt.data as qndata
import numpy as np
import pandas as pd


def load_data(period):
    futures = qndata.futures.load_data(tail=period, assets=["F_O", "F_LN", "F_KC"])
    crypto = qndata.cryptofutures.load_data(tail=period)
    return {"futures": futures, "crypto": crypto}, futures.time.values


def build_data_for_one_step(data, max_date: np.datetime64, lookback_period: int):
    min_date = max_date - np.timedelta64(lookback_period, "D")
    return {
        "futures": data["futures"].sel(time=slice(min_date, max_date)),
        "crypto": data["crypto"].sel(time=slice(min_date, max_date)),
    }


def predict_weights(market_data):
    def get_ml_model():
        from sklearn.linear_model import RidgeClassifier
        model = RidgeClassifier(random_state=18)
        return model

    def get_features(data):
        def remove_trend(prices_pandas_):
            prices_pandas = prices_pandas_.copy(True)
            assets = prices_pandas.columns
            for asset in assets:
                prices_pandas[asset] = np.log(prices_pandas[asset])
            return prices_pandas

        price = data.sel(field="close").ffill('time').bfill('time').fillna(0) # fill NaN
        for_result = price.to_pandas()
        features_no_trend_df = remove_trend(for_result)
        return features_no_trend_df

    def get_target_classes(data):

        price_current = data.sel(field="close").dropna('time')
        price_future = price_current.shift(time=-1).dropna('time')

        class_positive = 1
        class_negative = 0

        target_is_price_up = xr.where(price_future > price_current, class_positive, class_negative)
        return target_is_price_up.to_pandas()

    futures = market_data["futures"].copy(True)
    crypto = market_data["crypto"].copy(True)

    asset_name_all = crypto.coords['asset'].values
    features_all_df = get_features(futures)
    target_all_df = get_target_classes(crypto)

    predict_weights_next_day_df = crypto.sel(field="close").isel(time=-1).to_pandas()

    for asset_name in asset_name_all:
        target_for_learn_df = target_all_df[asset_name]
        feature_for_learn_df = features_all_df[:-1] # last value reserved for prediction

        # align features and targets
        target_for_learn_df, feature_for_learn_df = target_for_learn_df.align(feature_for_learn_df, axis=0, join='inner')

        model = get_ml_model()

        try:
            model.fit(feature_for_learn_df.values, target_for_learn_df)

            feature_for_predict_df = features_all_df[-1:]

            predict = model.predict(feature_for_predict_df.values)
            predict_weights_next_day_df[asset_name] = predict
        except:
            logging.exception("model failed")
            # if there is exception, return zero values
            return xr.zeros_like(crypto.isel(field=0, time=0))


    return predict_weights_next_day_df.to_xarray()


weights = qnbt.backtest(
    competition_type="cryptofutures",
    load_data=load_data,
    lookback_period=18,
    start_date='2014-01-01',
    strategy=predict_weights,
    window=build_data_for_one_step,
    analyze=True,
    build_plots=True
)
```
