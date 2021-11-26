import torch
import os
from torch import nn

import math
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(111)

BINARIES_PATH = os.path.join(os.getcwd(), 'models', 'binaries')
# Location where semi-trained models (during the training) will get saved to
CHECKPOINTS_PATH = os.path.join(os.getcwd(), 'models', 'checkpoints')
# All data both input (MNIST) and generated will be saved here
DATA_DIR_PATH = os.path.join(os.getcwd(), 'data')
# We'll be saving images here during GAN training
DEBUG_IMAGERY_PATH = os.path.join(DATA_DIR_PATH, 'debug_imagery')
# MNIST images have 28x28 resolution, it's just convenient to put this
# into a constant you'll see later why
MNIST_IMG_SIZE = 28
batch_size = 128

# Decide which device we want to run on
device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

print('Downloading datset...')
train_set = torchvision.datasets.MNIST(
    root=DATA_DIR_PATH, train=True, download=True, transform=transform
)

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataroot = "data/MNIST" # Root directory for dataset
dataset_path = 'artclub_gan'
# workers = 2 # Number of workers for dataloader
# batch_size = 128 # Batch size during training
# image_size = 64 # Spatial size of training images. All images will be resized to this size using a transformer.
# nc = 3 # Number of channels in the training images. For color images this is 3
# nz = 100 # Size of z latent vector (i.e. size of generator input)
# ngf = 64 # Size of feature maps in generator
# ndf = 64 # Size of feature maps in discriminator
# num_epochs = 5 # Number of training epochs
# lr = 0.0002 # Learning rate for optimizers
# beta1 = 0.5 # Beta1 hyperparam for Adam optimizers
# ngpu = 1 # Number of GPUs available. Use 0 for CPU mode.
# dataset = dset.ImageFolder(root=dataroot,
#                            transform=transforms.Compose([
#                                transforms.Resize(image_size),
#                                transforms.CenterCrop(image_size),
#                                transforms.ToTensor(),
#                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                            ]))
#
# Create data_loader
print('Loading data...')
# Create a data loader which will shuffle the data from train_set
# and return batches of 32 samples that you’ll use to train the neural networks
# batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)

# Size of the generator's input vector.
# Generator will eventually learn how to map these into meaningful images!
LATENT_SPACE_DIM = 100


# This one will produce a batch of those vectors
def get_gaussian_latent_batch(batch_size, device):
    return torch.randn((batch_size, LATENT_SPACE_DIM), device=device)


# It's cleaner if you define the block like this - bear with me
def vanilla_block(in_feat, out_feat, normalize=True, activation=None):
    layers = [nn.Linear(in_feat, out_feat)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_feat))
    # 0.2 was used in DCGAN, I experimented with other values like 0.5 didn't notice significant change
    layers.append(nn.LeakyReLU(0.2) if activation is None else activation)
    return layers

class GeneratorNet(torch.nn.Module):
    """Simple 4-layer MLP generative neural network.

    By default it works for MNIST size images (28x28).

    There are many ways you can construct generator to work on MNIST.
    Even without normalization layers it will work ok. Even with 5 layers it will work ok, etc.

    It's generally an open-research question on how to evaluate GANs i.e. quantify that "ok" statement.

    People tried to automate the task using IS (inception score, often used incorrectly), etc.
    but so far it always ends up with some form of visual inspection (human in the loop).

    Fancy way of saying you'll have to take a look at the images from your generator and say hey this looks good!

    """

    def __init__(self, img_shape=(MNIST_IMG_SIZE, MNIST_IMG_SIZE)):
        super().__init__()
        self.generated_img_shape = img_shape
        num_neurons_per_layer = [LATENT_SPACE_DIM, 256, 512, 1024, img_shape[0] * img_shape[1]]

        # Now you see why it's nice to define blocks - it's super concise!
        # These are pretty much just linear layers followed by LeakyReLU and batch normalization
        # Except for the last layer where we exclude batch normalization and we add Tanh (maps images into [-1, 1])
        self.net = nn.Sequential(
            *vanilla_block(num_neurons_per_layer[0], num_neurons_per_layer[1]),
            *vanilla_block(num_neurons_per_layer[1], num_neurons_per_layer[2]),
            *vanilla_block(num_neurons_per_layer[2], num_neurons_per_layer[3]),
            *vanilla_block(num_neurons_per_layer[3], num_neurons_per_layer[4], normalize=False, activation=nn.Tanh())
        )

    def forward(self, latent_vector_batch):
        img_batch_flattened = self.net(latent_vector_batch)
        # just un-flatten using view into (N, 1, 28, 28) shape for MNIST
        return img_batch_flattened.view(img_batch_flattened.shape[0], 1, *self.generated_img_shape)

# You can interpret the output from the discriminator as a probability and the question it should
# give an answer to is "hey is this image real?". If it outputs 1. it's 100% sure it's real. 0.5 - 50% sure, etc.
class DiscriminatorNet(torch.nn.Module):
    """Simple 3-layer MLP discriminative neural network. It should output probability 1. for real images and 0. for fakes.

    By default it works for MNIST size images (28x28).

    Again there are many ways you can construct discriminator network that would work on MNIST.
    You could use more or less layers, etc. Using normalization as in the DCGAN paper doesn't work well though.

    """

    def __init__(self, img_shape=(MNIST_IMG_SIZE, MNIST_IMG_SIZE)):
        super().__init__()
        num_neurons_per_layer = [img_shape[0] * img_shape[1], 512, 256, 1]

        # Last layer is Sigmoid function - basically the goal of the discriminator is to output 1.
        # for real images and 0. for fake images and sigmoid is clamped between 0 and 1 so it's perfect.
        self.net = nn.Sequential(
            *vanilla_block(num_neurons_per_layer[0], num_neurons_per_layer[1], normalize=False),
            *vanilla_block(num_neurons_per_layer[1], num_neurons_per_layer[2], normalize=False),
            *vanilla_block(num_neurons_per_layer[2], num_neurons_per_layer[3], normalize=False, activation=nn.Sigmoid())
        )

    def forward(self, img_batch):
        # Flatten from (N,1,H,W) into (N, HxW)
        img_batch_flattened = img_batch.view(img_batch.shape[0], -1)
        return self.net(img_batch_flattened)

# VANILLA_000000.pth is the model I pretrained for you,
# feel free to change it if you trained your own model (last section)!
model_path = os.path.join(BINARIES_PATH, 'VANILLA_000000.pth')
assert os.path.exists(model_path), f'Could not find the model {model_path}. You first need to train your generator.'

# Hopefully you have some GPU ^^ (you do if you’re using Spell)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# let's load the model, this is a dictionary containing model weights but also some metadata
# commit_hash - simply tells me which version of my code generated this model (hey you have to learn git!)
# gan_type - this one is "VANILLA" but I also have "DCGAN" and "cGAN" models
# state_dict - contains the actual neural network weights
model_state = torch.load(model_path)
print(f'Model states contains this data: {model_state.keys()}')

gan_type = model_state["gan_type"]
print(f'Using {gan_type} GAN!')

# Let's instantiate a generator net and place it on GPU (if you have one)
generator = GeneratorNet().to(device)
# Load the weights, strict=True just makes sure that the architecture corresponds to the weights 100%
generator.load_state_dict(model_state["state_dict"], strict=True)
# Puts some layers like batch norm in a good state so it's ready for inference <- fancy name right?
generator.eval() 

# This is where we'll dump images
generated_imgs_path = os.path.join(DATA_DIR_PATH, 'generated_imagery')  os.makedirs(generated_imgs_path, exist_ok=True)

#
# This is where the magic happens!
#

print('Generating new MNIST-like images!')
generated_img = generate_from_random_latent_vector(generator)
save_and_maybe_display_image(generated_imgs_path, generated_img, should_display=True)
