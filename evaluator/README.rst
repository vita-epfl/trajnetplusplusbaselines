Evaluation on TrajNet++
=======================

Description of the full evaluation procedure:
-------------------------------------------------------------------

The evaluation script and its help menu: ``python -m trajnetbaselines.lstm.trajnet_evaluator --help``

1.  Make sure that (a) test scenes (only containing the observations) are present in the 'test' folder (b) groundtruth scenes (containing the observations as well as the groundtruth predictions) are present in the 'test_private.' These conditions are true if the dataset was generated using trajnetplusplusdataset.

2. The full evaluation procedure generates a 'test_pred' folder containing the predictions of your model for the files in the test folder. NOTE: If Model predictions already exist in 'test_pred' from a previous run of the same model, the evaluator command SKIPS WRITING the new model predictions. In other words, already existing model predictions are not overwritten. 

3. Once the predictions are written in 'test_pred', our trajnet_evaluator compares the model predictions in 'test_pred' with groundtruth predictions in 'test_private' providing a complete table of evaluation metrics as Results.png.

Eg: ``python -m trajnetbaselines.lstm.trajnet_evaluator --path trajdata --output OUTPUT_BLOCK/trajdata/lstm_occupancy.pkl``
