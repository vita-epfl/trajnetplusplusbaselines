Training LSTMs
==============

`python -m trajnetbaselines.lstm.trainer`


## Average L2 [m]
                               |  Lin | LSTM
                    biwi_eth/* | 0.71 | 0.85
                  biwi_hotel/* | 0.56 | 0.71
               crowds_zara01/* | 0.61 | 0.65
               crowds_zara02/* | 0.67 | 0.65
         crowds_uni_examples/* | 0.73 | 0.81

## Average L2 (non-linear sequences) [m]
                               |  Lin | LSTM
                    biwi_eth/* | 1.19 | 1.00
                  biwi_hotel/* | 0.98 | 0.96
               crowds_zara01/* | 1.13 | 0.86
               crowds_zara02/* | 1.26 | 0.94
         crowds_uni_examples/* | 1.45 | 1.18

## Final L2 [m]
                               |  Lin | LSTM
                    biwi_eth/* | 1.28 | 1.29
                  biwi_hotel/* | 0.97 | 1.09
               crowds_zara01/* | 1.08 | 1.05
               crowds_zara02/* | 1.19 | 1.11
         crowds_uni_examples/* | 1.30 | 1.36


Good Models
===========

```
vanilla_lstm_20180728_202445.pkl, vanilla_lstm_20180807_055813.pkl wo syi
occupancy_lstm_20180803_055811.pkl
directional_lstm_20180729_082221.pkl
social_lstm_20180727_122804.pkl
```
