Data Setup
==========

Data Directory Setup
--------------------

All Datasets are stored in DATA_BLOCK

All Models after training are stored in OUTPUT_BLOCK: ``mkdir OUTPUT_BLOCK``

Data Conversion
---------------

For data conversion, refer to Trajnetdataset.

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

Evaluation on datasplits is based on the following categorization:

.. image:: docs/train/Categorize.png
