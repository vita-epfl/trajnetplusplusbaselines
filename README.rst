
Faster Implementation for LSTMs
===============================

Allows for faster training and testing under following two constraints (in data):

- Number of pedestrians per scene is fixed
- The pedestrians must be present at all times in the scene

Link to synthetic data (satisfying the two constraints): TBA Soon.

Curently, this implementation supports only nearest neighbour (NN) interaction pooling.


Training LSTMs
==============

The training script and its help menu:
``python -m trajnetbaselines.lstm.trainer --help``


Evaluation
==========

The evaluation script and its help menu: ``python -m evaluator.trajnet_evaluator --help``

More details regarding TrajNet++ evaluator are provided `here <https://github.com/vita-epfl/trajnetplusplusbaselines/blob/master/evaluator/README.rst>`_

Evaluation on datasplits is based on the following categorization:

.. image:: docs/train/Categorize.png

Citation
========

If you find this code useful in your research then please cite

.. code-block::

    @inproceedings{Kothari2020HumanTF,
      title={Human Trajectory Forecasting in Crowds: A Deep Learning Perspective},
      author={Parth Kothari and Sven Kreiss and Alexandre Alahi},
      year={2020}
    }

