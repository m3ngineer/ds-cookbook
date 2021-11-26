import torch
from torch import nn

import math
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(111)

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
    root="uploads/artclub_gan/MNIST/raw", train=True, download=False, transform=transform
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
batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)

# Plot examples of training data
real_samples, mnist_labels = next(iter(train_loader))
for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(real_samples[i].reshape(28, 28), cmap="gray_r")
    plt.xticks([])
    plt.yticks([])

# Generate neural network
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), 784)
        output = self.model(x)
        return output

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, x):
        output = self.model(x)
        output = output.view(x.size(0), 1, 28, 28)
        return output

def save_images(self, generated_images, epoch_no, batch_no):
    """
    Shows and saves generated images
    :param generated_images:
    :param epoch_no:
    :param batch_no:
    :return:
    """
    plt.figure(figsize=(8, 8), num=2)
    gs1 = gridspec.GridSpec(8, 8)
    gs1.update(wspace=0, hspace=0)

    for i in range(64):
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect('equal')
        image = generated_images[i, :, :, :]
        image += 1
        image *= 127.5
        fig = plt.imshow(image.astype(np.uint8))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    plt.tight_layout()
    save_name = 'new_images/generated_epoch' + str(
        epoch_no + 1) + '_batch' + str(batch_no + 1) + '.png'
    if not os.path.exists('new_images'):
        os.mkdir('new_images')
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
    plt.pause(0.0000000001)
    plt.show()

discriminator = Discriminator()
generator = Generator().to(device=device)

# Training model
lr = 0.0001
num_epochs = 50
loss_function = nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

for epoch in range(num_epochs):
    for n, (real_samples, mnist_labels) in enumerate(train_loader):
        # Data for training the discriminator
        real_samples = real_samples.to(device=device)
        real_samples_labels = torch.ones((batch_size, 1)).to(
            device=device
        )
        latent_space_samples = torch.randn((batch_size, 100)).to(
            device=device
        )
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((batch_size, 1)).to(
            device=device
        )
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat(
            (real_samples_labels, generated_samples_labels)
        )

        # Training the discriminator
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(
            output_discriminator, all_samples_labels
        )
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Data for training the generator
        latent_space_samples = torch.randn((batch_size, 100)).to(
            device=device
        )

        # Training the generator
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(
            output_discriminator_generated, real_samples_labels
        )
        loss_generator.backward()
        optimizer_generator.step()

        # Show loss
        if n == batch_size - 1:
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")

            # Save models
            model_state = {
                        'epoch': epoch + 1,
                        'loss_gen':  loss_generator,
                        'loss_dis': loss_discriminator,
                        # 'model_gen_state_dict': netG.state_dict(),
                        # 'model_dis_state_dict': netD.state_dict(),
                        'optimizer_gen_state_dict': optimizer_generator.state_dict(),
                        'optimizer_dis_state_dict': optimizer_discriminator.state_dict(),
                    }
            print('Saving model...')
            save_model(model_state, dataroot)

        current_batch_size = real_samples.shape[0]
        # Each 50 batches show and save images
        if ((n + 1) % 5 == 0 and current_batch_size == batch_size):
            print('Saving images...')
            self.save_images(generated_samples, epoch, n)

# check samples genreated by GAN
latent_space_samples = torch.randn(batch_size, 100).to(device=device)
generated_samples = generator(latent_space_samples)

generated_samples = generated_samples.cpu().detach()
for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(generated_samples[i].reshape(28, 28), cmap="gray_r")
    plt.xticks([])
    plt.yticks([])
