Evaluation on TrajNet++
=======================

Description of the full evaluation procedure (outputs all metrics):
-------------------------------------------------------------------

The evaluation script and its help menu: ``python -m evaluator.trajnet_evaluator --help``

1.  Make sure that (a) test scenes (only containing the observations) are present in the 'test' folder (b) groundtruth scenes (containing the observations as well as the groundtruth predictions) are present in the 'test_private.' These conditions are true if the dataset was generated using trajnetplusplusdataset.

2. The full evaluation procedure generates a 'test_pred' folder containing the predictions of your model corresponding to the files in the test folder. This process is carried out using 'write.py' (See this file for more details.) NOTE: If Model predictions already exist in 'test_pred' from a previous run, this process SKIPS WRITING the new model predictions if the model name is the SAME. 

3. Once the predictions are written in 'test_pred', our trajnet_evaluator compares the model predictions in 'test_pred' with groundtruth predictions in 'test_private' providing a complete table of evaluation metrics as Results.png.

Eg: ``python -m evaluator.trajnet_evaluator --data trajdata --output OUTPUT_BLOCK/trajdata/occupancy.pkl``


Description of the FAST evaluation procedure (outputs only ADE/FDE metrics):
----------------------------------------------------------------------------

The evaluation script and its help menu: ``python -m evaluator.fast_evaluator --help``

1.  This procedure does not generate a test_pred file. It quickly provides only the ADE/FDE metric in the terminal. 

Eg: ``python -m evaluator.fast_evaluator --data trajdata --output OUTPUT_BLOCK/trajdata/occupancy.pkl``
