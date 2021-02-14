
Faster Implementation for LSTMs
===============================

Allows for faster training and testing under following two constraints (in data):

- Number of pedestrians per scene is fixed
- The pedestrians must be present at all times in the scene

Link to synthetic data (satisfying the above two constraints): `here <https://github.com/vita-epfl/trajnetplusplusdata/releases/tag/v3.1>`_

Curently, this implementation supports only parallelized nearest neighbour (NN) interaction pooling.

Data Setup
==========

The detailed step-by-step procedure for setting up the TrajNet++ framework can be found `here <https://thedebugger811.github.io/posts/2020/03/intro_trajnetpp/>`_

Download the `synthetic data <https://github.com/vita-epfl/trajnetplusplusdata/releases/tag/v3.1>`_

Move the provided datasets to DATA_BLOCK/

Move the accompanied goal_file (.pkl file) to goal_files/


Training Models
===============

LSTM
----

The training script and its help menu:
``python -m trajnetbaselines.lstm.trainer --help``

**Run Example**

.. code-block::

   ## Vanilla LSTM with Goals
   python -m trajnetbaselines.lstm.trainer --path five_parallel_synth --augment --goals --epochs 15 --step_size 6

   ## NN LSTM with Goals 
   python -m trajnetbaselines.lstm.trainer --path five_parallel_synth --type nn --augment --goals --epochs 15 --step_size 6

Evaluation
==========

The evaluation script and its help menu: ``python -m evaluator.trajnet_evaluator --help``

**Run Example**

.. code-block::

   ## TrajNet++ evaluator (saves model predictions. Useful for submission to TrajNet++ benchmark)
   python -m evaluator.trajnet_evaluator --path five_parallel_synth --output OUTPUT_BLOCK/five_parallel_synth/parallel_lstm_goals_nn_None.pkl

   ## Fast Evaluator (does not save model predictions)
   python -m evaluator.fast_evaluator --path five_parallel_synth --output OUTPUT_BLOCK/five_parallel_synth/parallel_lstm_goals_nn_None.pkl

More details regarding TrajNet++ evaluator are provided `here <https://github.com/vita-epfl/trajnetplusplusbaselines/blob/master/evaluator/README.rst>`_

Evaluation on datasplits is based on the following `categorization <https://github.com/vita-epfl/trajnetplusplusbaselines/blob/master/docs/train/Categorize.png>`_


Benchmarking Models
===================

We host the `Trajnet++ Challenge <https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge>`_ on AICrowd allowing researchers to objectively evaluate and benchmark trajectory forecasting models on interaction-centric data. We rely on the spirit of crowdsourcing, and encourage researchers to submit their sequences to our benchmark, so the quality of trajectory forecasting models can keep increasing in tackling more challenging scenarios.

Citation
========

If you find this code useful in your research then please cite

.. code-block::

    @article{Kothari2020HumanTF,
      title={Human Trajectory Forecasting in Crowds: A Deep Learning Perspective},
      author={Parth Kothari and S. Kreiss and Alexandre Alahi},
      journal={ArXiv},
      year={2020},
      volume={abs/2007.03639}
    }

