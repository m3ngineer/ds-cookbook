# Creating art through using generative adversarial networks

General adversarial networks consist of two convolutional neural networks (CCNs): a generator and a discriminator. The generator create new data based on a training set, while the disciminator attempts to differentiate between the training and data that the generator made.

The generator and discriminator have an adversarial and productive relationship whereby both try to outsmart the other.

## About the data
The data used for this project is from [WikiArt](https://www.wikiart.org), a non-profit collaboration that aims to make art accessible to anyone and everywhere. The organization has made over 100,000 paintings sorted by different styles and artists. A dataset of artworks by genre can be downloaded [here](https://www.kaggle.com/ipythonx/wikiart-gangogh-creating-art-gan).

## Training the model on Spell

Spell is a machine learning platform that can be useful for running modeling training where your local computer may be insufficient.

Install Spell
```pip install spell```

Authenticate Spell
```spell login```

Upload the necessary data to Spell
```spell upload <filename>```

Run the model training on Spell's GPU infrastructure
```spell run python art_gan.py -t V100 -m uploads/art_gan/<portrait_data.npy>```

Download the outputs
```Spell run python art_gan.py -t V100 -m uploads/art_gan/cubism_data.npy```
