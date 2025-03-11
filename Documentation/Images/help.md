## Program Description
This browser application provides a flexible environment for experimenting with new state of the art AI methods inside the area of algorithmic trading (AT) 
 while using high quality data of 13 different crypto currencies.

Currently the application features experiments in two different sub-areas of AT:
- inside of long-term series forecasting (LTSF) one can produce multiple forecasts at once with [NLinear](https://arxiv.org/abs/2205.13504v3) and NDLinear (our own variation)
- (WIP) inside a reinforcement learning (RL) environment one can train a [PPO](https://arxiv.org/abs/1707.06347) agent to make profitable buying and selling decisions automatically.

## Program Structure

To start any experiment, one follows the figure below and firstly defines the dataset settings and optionally fetches new datasets or updates them. Model and benchmark parameters are defined later and will impact the model's evaluation results which are visualized at the end of the experiment like in this depiction of the program structure:

![project_structure](https://github.com/user-attachments/assets/efded7a9-bc6a-48ff-a5b8-0d62b8780e14)

Note that in this depiction all RL systems are marked as WIP and that a second project with a real-world Binance backtesting environment 
is planned to backtest the RL system in more realistic conditions to also capture slippage and order delay.
### Datasets 

When changing the datasets, these experiments mentioned above are performed in slightly different environments that mostly change the feature preparation process. 

In the main menu one selects if he wants to make experiments on either:
- a scalping or day trading dataset with frequencies ranging from 1 *min* to 2 *h* where bid-ask features are not considered for simplicity.
- a high-frequency trading (HFT) dataset fetched in real time with frequencies ranging from 10 *ms* (WIP: currently only 250 *ms*) to 1 min where bid-ask features are considered. 
  For this dataset the calculation of technical indicators is strongly reduced to a small collection of reliable indicators.

Additionally, this program provides its datasets for downloading so that they can be  used for other purposes.

### Feature Engineering

The feature engineering will change depending on the dataset and the model choice.
For every setting there will be a basic feature engineering with around 100 custom 
features and also a feature engineering with 101 extra alpha factors from [WorldQuant](https://github.com/yli188/WorldQuant_alpha101_code/blob/master/101Alpha_code_1.py),
where most of these formulaic alphas are efficiently used in today's real-world trading systems. 
Besides that, RL models receive technical indicators whilst LTSF models do not because most of these provide no meaningful information.

It is strongly recommended to only use 
at most 40-50 additional features besides the these features to prevent overfitting. 
All features have to be selected in groups of about 5 or more related features for simplicity and they can be deselected later on.

Keep in mind for time series data that patterns may change with time and earlier patterns do not necessarily hold for later times.

### NLinear and NDLinear

Generally speaking, both of these models are very simple and currently the most effective models for LTSF: Their calculations involve only one to two matrix multiplications of a classical neural network and 
 capture temporal patterns by looking at a larger look-back window as depicted below. 

Applied with an GPU, these models can complement any AT or RL system with its forecasts as one model execution takes only about 0.4 *ms* for up to 768 simultaneous forecaststs.

A depiction of the [Linear](https://arxiv.org/abs/2205.13504v3) model:

![Linear](https://github.com/user-attachments/assets/954bb53d-2472-4ed9-b3dc-5e6624f3b474)
 
##### NLinear

- The NLinear represents Linear with an additional time series normalization by subtracting the viewed part of time series by its last value.
- Normally data can be standardized by subtracting it by its mean and dividing by its variance to achive a mean of 0 and unit variance. 
However, when the time series data changes, its mean changes too, causing a distribution shift for the normalized data and thus the
model to face unknown data. 
By subtracting the time series by its last value the standardized data distribution gets shifted back inplace.

##### NDLinear

- DLinear was the former version Linear and NLinear and decomposes the time series into a trend and remainder component by subtracting a moving average from it.
- Two separate networks predict each component separately, reducing the pattern complexity for the model to learn. 
 The result is finally calculated by taking the sum of the two component predictions.
- NLinear and DLinear perform similary while NLinear remains the simpler model.

Here is a depiction of the proposed NDLinear (figure reused from [DLinear](https://arxiv.org/abs/2205.13504v2)):

![NDLinear](https://github.com/user-attachments/assets/9a5c2352-661f-4fae-a43b-a2bb1df3c7ef)

- However, we still think that a decomposition of the time series will benefit NLinear.
- We also we think that by normalizing a time series by subtracting its last value like NLinear, 
 the bias value of the viewed time series shifts too often with each time new step, producing too complex bias patterns for the model to learn.

This is why we propose a variant of the model called NDLinear which now ovoids a bias-oscillating by normalizing 
the trend component instead of the whole time series, avoiding oscillation through the noisy remainder component.
<br>

##### PPO 

[Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) is one of many RL algorithms that can be used for algorithmic trading. We prefer to use this model due to its simplicity and stable training process. As a policy gradient method it needs slightly more training data than value-based methods 
but can conversely handle more complex environments with numerous states, which is more practical to leverage feature engineering. 

Under the hood, PPO uses a basic neural network to predict the most valuable actions based on a reward function and further RL techniques. 
Nevertheless, when looking at the following [paper](https://www.ijcai.org/Proceedings/2020/627), this neural network can be replaced by LSTMs to better adapt the model to time series data.
Likewise, we are looking forward to implement LSTM and DLinear models as the core model for PPO like in this depiction a RL environment:

