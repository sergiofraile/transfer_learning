# Transfer Learning

This project has been based on the transfer learning excercise by [@lmoroney](https://twitter.com/lmoroney) that I found in his Tensorflow [course](https://www.coursera.org/learn/convolutional-neural-networks-tensorflow/).

Everything is segregated in small files that together brings the whole transfer learning excercise live. 

## Structure

All the code is in the src directory. There are two files called pipe (a .py and a .sh) that do exactly the same. They will basically run all the development pipe, from downloading the dataset to run the network and plot the history.

If you prefer to run them manually, *pre.sh* will download the inception network weights and the dataset and move them to the respective datasets and models folders (will be created if they don't exist).

The network is located in *main.py*.

All the constants have been abstracted to a *constants.py* file so tunning the model parameters is fairly easy and quick.

## How to run it?

Simply run one of the pipe files from the top level directory:

```
python src/pipe.py
```

or

```
sh src/pipe.sh
```

Only makes sure that your python installation has access to the libraries indicated in *requirements.txt*. You can install them with pip (preferible in an isolated environment) with:

```
pip install -r requirements.tx
```
