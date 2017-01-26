Python implementation of Vector Auto Regressive model for time-series prediction.

This code is ported from MATLAB code by [Igor](http://www-users.cs.umn.edu/~melnyk/index.html?2).

The data is expected in multiple pickle files, each file representing one complete time-series. To run,

`python buildVAR 1 traindata/ testdata/`

First argument is `p`, the order of the VAR model.
Second argument is the directory containing the training data files.
Third argument is the directory containing the test data files.

Test set mean RMSE is printed out to the screen.
