Link to the Challenge: `Trajnet++ Challenge <https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge>`_

Starter Guide (NEW): `Introducing Trajnet++ Framework <https://thedebugger811.github.io/posts/2020/03/intro_trajnetpp/>`_

Data Setup
==========

Data Directory Setup
--------------------

All Datasets are stored in DATA_BLOCK

All Models after training are stored in OUTPUT_BLOCK: ``mkdir OUTPUT_BLOCK``

Data Conversion
---------------

For data conversion, refer to trajnetplusplusdataset.

After conversion, copy the converted dataset to DATA_BLOCK

Training LSTMs
==============

The training script and its help menu:
``python -m trajnetbaselines.lstm.trainer --help``

Training GANs
==============

The training script and its help menu:
``python -m trajnetbaselines.sgan.trainer --help``

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

