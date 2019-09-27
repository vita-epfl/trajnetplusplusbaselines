Venv Setup (Temporary: For VITA)
================================

Create a common virtual environment.

Go to trajnettools. 

``python3 setup.py sdist bdist_wheel`` 

``pip install .`` 


Data Setup
==========

For data conversion, refer to Trajnetdataset. 

DATA_BLOCK contains different training sets: ``mkdir DATA_BLOCK`` 

Move the converted dataset for training to DATA_BLOCK:
``mkdir DATA_BLOCK\trajdata``

``cp -r <converted_dataset> DATA_BLOCK\trajdata``


The corresponding outputs after training are stored in OUTPUT_BLOCK: ``mkdir OUTPUT_BLOCK``

Requirements
============

Social Force: ``pip install 'socialforce[test,plot]'`` 

Training LSTMs
==============

The training script and its help menu:
``python -m trajnetbaselines.lstm.trainer --help``


Evaluation on datasplits is based on the following categorization:

.. image:: docs/train/Categorize.png

Create the table below with:
``python -m evaluator.trajnet_evaluator --output <path_to_trained_model>``

.. image:: docs/train/Eval.png