# Optimization for machine learning - Mini Project
# Accomodating learning rates and warm up for deep learning using large minibatches - scalability limitations

The aim of this project has been to investigate and access the challenges in relation to scalability gains, when using
large mini batches for training deep neural networks. More specifically the project gives a look into the effects of scaling the learning rate
to the batch size. The experiments were carried out on the CIFAR-10 dataset, using a standard CNN (model 3 obtained at this site https://shonit2096.medium.com/cnn-on-cifar10-data-set-using-pytorch-34be87e09844) and a simple validation approach (for it to be comparable to the studies of Goyal et al. 2017, Luschi et al. 2018).
The models trained with scaled learning rates for the various batch-sizes were compared to a base learning rate for all batch sizes to determine the accuracy and generalization gains of this adjustment.
In addition to this, we accessed whether various warm up strategies could improve predictions even further. To compare and verify the results of the models trained with large mini batches, these results were compared to
the results from similar baseline analyses with the use of small mini-batch sizes (as mini-batch of sizes between 2-32 had performed best in previous studies of Luschi et al. 2018, trained on the AlexNet and ResNet).
Using the accuracy of models trained with small batch sizes as a threshold for satisfactory generalization and accuracy assisted us in comparing and verifying the results obtained by training the models with large mini-batch sizes.
The end goal was to compare the trade-off between accuracy and training-time, to determine the breaking point of when we start to loose our scalability gains.

## Description
The results of the projects were computed in the 4 main Jupyter Notebooks - 1. baseline_small_minibatch.ipynb (computing standard threshold), 2. baseline_large_minibatch.ipynb (baseline to compare gains of learning rate scaling and warm up), 3. scaled_lr_large_minibatch.ipynb, 4. gradual_warmup_scaled_lr_large_minibatch.ipynb (incl. both warm-up and batch-scaling of learning rate).
For visualizing the results the notebook Results&Visualizations.ipynb must be run.
In addition to the notebooks, a python-file (models.py) containing the CNN that is shared between all the notebooks can also be found in the repository.
All of the model training notebooks has more or less the same functionality and structure, and only a few lines differ to accomodate the different modes of training (explained in the notebooks). The below sections are described and discussed further down.
* 1. Loading libraries, setting directories and testing for GPU availability (Colab)
* 2. Loading and reading CIFAR-10 dataset (no data cleaning or exploration was done for the dataset)
* 3. Training the network
* 4. Saving results


### Libraries and Colab
The training setup and the models were exclusively build on the PyTorch library.
All experiments were carried out on Google Colab's free GPU service.

### Loading the CIFAR-10 dataset
Training for all models was carried out on the full dataset, containing 50.000 labeled training images (10 classes evenly distributed) and 10.000 test images. No data-cleaning or -exploration was carried out for the dataset.

### Training
Training was carried in a simple validation setting (no CV to compare it to other studies of Luschi et al. 2018). The CNN used throughout the project consisted of 3 convolutional blocks in connection with a fully connected layer.
Each convolutional block containing 2 conv. layers using batch normalization, the rectified linear unit activation function and max pooling, along with dropout in various layers - see models.py for more details.

### Saving results
The following results were extracted for every epoch of every model-training: {batch size, epoch number, accuracy, generalization error, running time}

### Plotting
Plots were generated with the notebook Results&Visualizations.ipynb to support the discussion in the report.

## Run code
To run the code open the various notebooks and specify your file-paths where it is prompted in the code.

## File overview
README.md
models.py
baseline_small_minibatch.ipynb
baseline_large_minibatch.ipynb
scaled_lr_large_minibatch.ipynb
gradual_warmup_scaled_lr_large_minibatch.ipynb
Results&Visualizations.ipynb
report.pdf
* DATA FOLDER: As the results and plots capture information about running time and other parameters, it is necessary to run the notebooks on Google Colab or VMs with similar hardware for reproducibility. As running all batch sizes in a loop for every notebook would trigger Colabs user-limitation to free GPU service and it would be too tiresome for the reader to change filenames manually from run to run (doing a single batch size at a time to avoid Colab from crashing/limiting free GPU), in order to produce the .csv files him/herself and run the plots, all of the results (.csv files) have been stored in the data-folder. These can be used to run the Results&Visualization.ipynb.

## Conclusion
For insights to the discussion and conclusion we refer to the report.
