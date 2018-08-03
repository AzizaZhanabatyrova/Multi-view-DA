# Multi-view DA

The code utilized in this work uses AdaptSegNet's code as starting point. See [AdaptSegNet](https://github.com/wasidennis/AdaptSegNet).

# 1. Requirements

- The PyTorch machine learning library for Python, version 0.4 as well as Python programming language, version 2.7.14 were used for code development. The code has not been tested with earlier or later versions of Python and Pytorch, thus forward and backward compatibilities are not guaranteed, however, they are expected.

- Other necessary packages are matplotlib (2.2.2), numpy (1.14.5), pillow (5.1.0), opencv (3.4.1), scipy (1.1.0), tensorboardX (1.2), torchvision (0.2.0).

- To run the trainings, it is required to have a CUDA enabled graphics processor unit. In our case, we have used a NVIDIA GeForce GTX 1080 Ti GPU with 11GB RAM to train the models.

- Cityscapes, ISPRS, CrowdAI and Airsim datasets. All these datasets should be available for download on their websites, except for Airsim, which can be obtained by contacting the responsible people at the Autonomous Systems
Lab at ETH. See [Autonomous Systems Lab - ETH](http://www.asl.ethz.ch/)

# 2. Preparing the datasets

Before running the trainings, it is necessary to have not only the datasets, buttheir respective text files containing the path to every image and label used during training, validation and testing phases. Such text files are included here and can be easily changed using any text editor.

# 3. Usage

### 3.1 Training

To run the trainings, there are two different files, one for configuration 1 and another one for configuration 2 and 3. They include many options that can be set using the terminal such as the absolute path to the folders containing the datasets and the dataset text files, number of training steps, batch size, and so on. A full list of such options and their description can be obtained by running the command given as follows:

```sh
$ python train_conf1.py -h 
```

or 

```sh
$ python train_conf2_conf3.py -h 
```

The results of the training can be visualized using the TensorboardX interface by running the command given below. The path to the saved log files is passed as an argument when running the train.py file.

```sh
$ tensorboard -logdir=path_to_saved_log_files -host=localhost -port=port_number
```

### 3.2 Evaluation

Similarly, the evaluation can be performed by running the evaluate.py file with the appropriate options. Such options can be found by running:

```sh
$ python evaluation.py -h
```

When the evaluation is finished, the mIoU is informed in the console.