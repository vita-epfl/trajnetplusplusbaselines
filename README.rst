Explaining Trajectory Forecasting Models: Layer-wise Relevance Propagation
=========================================================================

.. image:: docs/train/LRP.gif

Visualizing the decision-making of directional pooling (D-Grid) using layer-wise relevance propagation. The darker the yellow circles, the more is the weight provided by the primary pedestrian (blue) to the corresponding neighbour (yellow). Our proposed directional pooling, driven by domain knowledge, outputs human-like trajectories with more intuitive focus on surrounding neighbours as compared to social pooling (S-Grid).

Dependencies
============

1. Celluloid


Running LRP
===========

Once an LSTM model is trained, use the following script for visualizing LRP:

``python -m evaluator.fast_evaluator --path <dataset_name> --output <model_pkl_file>``

Animations are saved in the *anims* folder

The script for generating the above animation (in anims folder):

``python -m evaluator.fast_evaluator --path crowds_zara02 --output lstm_directional_one_12_6.pkl``


Citation
========

If you find this code useful in your research then please cite

.. code-block::

    @inproceedings{Kothari2020HumanTF,
      title={Human Trajectory Forecasting in Crowds: A Deep Learning Perspective},
      author={Parth Kothari and Sven Kreiss and Alexandre Alahi},
      year={2020}
    }


Acknowledgements
================

Our LRP code is inspired from `LRP for LSTM <https://github.com/ArrasL/LRP_for_LSTM>`_
