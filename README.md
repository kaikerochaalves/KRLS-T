# KRLS-T (kernel recursive least squares tracker)

The kernel recursive least squares tracker (KRLS-T) is a model proposed by Vaerenbergh et al. [1].

- [KRLS-T](https://github.com/kaikerochaalves/KRLS-T/blob/65a33bc8c494efa0220efd0a1d32c78ca1a91579/Model/KRLS_T.py) is the KRLS-T model.

- [GridSearch_AllDatasets](https://github.com/kaikerochaalves/KRLS-T/blob/65a33bc8c494efa0220efd0a1d32c78ca1a91579/GridSearch_AllDatasets.py) is the file to perform a grid search for all datasets and store the best hyper-parameters.

- [Runtime_AllDatasets](https://github.com/kaikerochaalves/KRLS-T/blob/65a33bc8c494efa0220efd0a1d32c78ca1a91579/Runtime_AllDatasets.py) perform 30 simulations for each dataset and compute the mean runtime and the standard deviation.

- [MackeyGlass](https://github.com/kaikerochaalves/KRLS-T/blob/65a33bc8c494efa0220efd0a1d32c78ca1a91579/MackeyGlass.py) is the script to prepare the Mackey-Glass time series, perform simulations, compute the results and plot the graphics. 

- [Nonlinear](https://github.com/kaikerochaalves/KRLS-T/blob/65a33bc8c494efa0220efd0a1d32c78ca1a91579/Nonlinear.py) is the script to prepare the nonlinear dynamic system identification time series, perform simulations, compute the results and plot the graphics.

- [LorenzAttractor](https://github.com/kaikerochaalves/KRLS-T/blob/65a33bc8c494efa0220efd0a1d32c78ca1a91579/LorenzAttractor.py) is the script to prepare the Lorenz Attractor time series, perform simulations, compute the results and plot the graphics. 

[1] S. Van Vaerenbergh, M. L ́azaro-Gredilla, I. Santamar ́ıa, Kernel recursive least-squares tracker for time-varying regression, IEEE Transactions on Neural Networks and Learning Systems 23 (8) (2012) 1313–1326. doi: https://doi.org/10.1109/TNNLS.2012.2200500.
