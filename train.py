import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

from models import Generator, Discriminator

# Set random seed for reproducibility
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
print("Random Seed: ", manualSeed)

############################ Inputs ############################
dataroot = r"C:/Users/andre/Desktop/Facultatea alternativa de inteligenta/Modulul 8/gan/data/myface" 

batch_size = 64 # Batch size during training
image_size = 64 # Spatial size of training images. 
nc = 3 # Number of channels in the training images. For color images this is 3
nz = 100 # Size of z latent vector (i.e. size of generator input)
ngf = 64 # Size of feature maps in generator
ndf = 64 # Size of feature maps in discriminator

num_epochs = 500 # Number of training epochs
lr = 0.0002 # Learning rate for optimizers
beta1 = 0.5 # Beta1 hyperparam for Adam optimizers
####################################################################################

############################ Dataset settings ############################
# resize (174, 162) -> (68, 64), for keeping the aspect ratio). Center crop cropps (64, 64) 
transforms = transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])

dataset = dset.ImageFolder(root=dataroot, transform=transforms)
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Plot some training images
# real_batch = next(iter(dataloader))
# print(real_batch[0].to(device).shape)
# plt.figure(figsize=(8, 8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
# plt.show()
################################################################################################################

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Create the generator #
netG = Generator(nz, ngf, nc).to(device)
netG.apply(weights_init) # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.02.

# Create the Discriminator #
netD = Discriminator(ndf, nc).to(device)
netD.apply(weights_init) # Apply the weights_init function to randomly initialize all weightsto mean=0, stdev=0.2.

# Initialize BCELoss function #
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

#### Training Loop ####

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader):
        ####################################################################################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        #################################################################################

        ## Train with all-real batch
        netD.zero_grad()

        # Format batch. The only the image from dataloader, not with gr folder
        real_images = data[0].to(device)
        label_true = torch.full((batch_size, ), 1, dtype=torch.float, device=device)

        # Forward pass real batch through D
        output = netD(real_images).view(-1)
        
        # Calculate loss on all-real batch
        errD_real = criterion(output, label_true)

        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(batch_size, nz, 1, 1, device=device)

        # Generate fake image batch with G
        fake = netG(noise)
        label_false = torch.full((batch_size, ), 0, dtype=torch.float, device=device)

        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)

        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label_false)

        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        # Update D
        optimizerD.step()
        #################################################################################

        ########################################################
        # (2) Update G network: maximize log(D(G(z)))
        ######################################################
        netG.zero_grad()
        label_true = torch.full((batch_size, ), 1, dtype=torch.float, device=device)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)

        # Calculate G's loss based on this output
        errG = criterion(output, label_true)

        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()

        # Update G
        optimizerG.step()
        ######################################################

        # Output training stats
        if i % 10 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD_fake.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD_fake.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 100 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                _fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(_fake, padding=2, normalize=True))
            
            # Save generated images
            if not os.path.exists("generated"):
                os.mkdir("generated")
            
            img_to_save = np.transpose(img_list[-1].numpy(), (1, 2, 0))
            plt.imsave(os.path.join("generated", f"Epoch{epoch}_Iter{iters}.png"), img_to_save)
        
        iters += 1
    
    # Save generator        
    torch.save(netG.state_dict(), os.path.join("generated", f"Epoch{epoch}"))

    
