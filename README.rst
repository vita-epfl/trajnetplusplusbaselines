Training LSTMs
==============

The training script and its help menu:
``python -m trajnetbaselines.lstm.trainer --help``

Create the table below with:
``python -m trajnetbaselines.eval``


.. code-block::

    ## Average L2 [m]
                                   |   N  |  Lin | LSTM | O-LSTM | D-LSTM | S-LSTM
                    val/biwi_hotel |    6 | 1.15 | 0.86 |  0.83  |  0.87  |  0.78
                 val/crowds_zara02 |  158 | 0.58 | 0.46 |  0.46  |  0.46  |  0.47
                 val/crowds_zara03 |   35 | 0.58 | 0.42 |  0.39  |  0.41  |  0.39
            val/crowds_students001 |  378 | 0.69 | 0.44 |  0.44  |  0.44  |  0.43
            val/crowds_students003 |  217 | 0.79 | 0.51 |  0.50  |  0.51  |  0.50
                      val/dukemtmc |  144 | 1.16 | 0.80 |  0.79  |  0.80  |  0.80

    ## Average L2 (non-linear sequences) [m]
                                   |   N  |  Lin | LSTM | O-LSTM | D-LSTM | S-LSTM
                    val/biwi_hotel |    3 | 1.45 | 0.86 |  0.82  |  0.80  |  0.72
                 val/crowds_zara02 |   62 | 1.11 | 0.74 |  0.73  |  0.75  |  0.76
                 val/crowds_zara03 |   13 | 1.05 | 0.56 |  0.57  |  0.55  |  0.56
            val/crowds_students001 |  151 | 1.13 | 0.60 |  0.60  |  0.60  |  0.59
            val/crowds_students003 |   94 | 1.20 | 0.65 |  0.65  |  0.65  |  0.64
                      val/dukemtmc |   58 | 2.05 | 1.16 |  1.15  |  1.17  |  1.15

    ## Final L2 [m]
                                   |   N  |  Lin | LSTM | O-LSTM | D-LSTM | S-LSTM
                    val/biwi_hotel |    6 | 2.51 | 1.99 |  1.93  |  2.03  |  1.76
                 val/crowds_zara02 |  158 | 1.11 | 0.99 |  0.99  |  0.99  |  1.03
                 val/crowds_zara03 |   35 | 1.08 | 0.91 |  0.86  |  0.90  |  0.87
            val/crowds_students001 |  378 | 1.31 | 0.97 |  0.97  |  0.96  |  0.95
            val/crowds_students003 |  217 | 1.49 | 1.11 |  1.08  |  1.10  |  1.09
                      val/dukemtmc |  144 | 2.16 | 1.71 |  1.70  |  1.71  |  1.69

    ## Collision [-]
                                   |   N  |  Lin | LSTM | O-LSTM | D-LSTM | S-LSTM
                    val/biwi_hotel |    6 |    0 |    0 |     0  |     0  |     0
                 val/crowds_zara02 |  177 |    0 |    0 |     0  |     0  |     0
                 val/crowds_zara03 |   33 |    0 |    0 |     0  |     0  |     0
            val/crowds_students001 |  404 |    0 |    0 |     0  |     0  |     0
            val/crowds_students003 |  224 |    0 |    0 |     0  |     0  |     0
                      val/dukemtmc |  144 |    0 |    0 |     0  |     0  |     0
Good Models
===========

* vanilla_lstm_20180728_202445.pkl, vanilla_lstm_20180807_055813.pkl wo syi
* occupancy_lstm_20180803_055811.pkl
* directional_lstm_20180729_082221.pkl
* social_lstm_20180727_122804.pkl
