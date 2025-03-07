import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

class Discriminator(nn.Module):
    def __init__(self, img_dim): # Using MNIST data (img_dim = 784)
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim,128),
            nn.LeakyReLU(0.1),  # in GANs leakyReLu is often times a better choice
            nn.Linear(128,1),   # fake=0, real=1
            nn.Sigmoid()        # ensure its between 0 and 1
        )

    def forward(self,x):
        return self.disc(x)
    

class Generator(nn.Module):
    def __init__(self,z_dim,img_dim):   # z_dim: the dimension of the latent noise that Genrator gonna take as input
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim,256),
            nn.LeakyReLU(0.1),
            nn.Linear(256,img_dim),    # 28*28*1 --> 784
            nn.Tanh(),  # make sure the pixel value is between -1 and 1, 
        )               # since we will normalize the input image into (-1,1)
    
    def forward(self,x):
        return self.gen(x)
    

if __name__ =="__main__":
    # Hyperparameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 3e-4
    z_dim = 64  # 128, 256
    # GANs are super sensitive to hyperparameters
    image_dim = 28 * 28 *1
    batch_size = 32
    num_epochs = 50

    disc = Discriminator(img_dim = image_dim).to(device = device)
    gen = Generator(z_dim = z_dim, img_dim = image_dim).to(device = device)
    fixed_noise = torch.randn((batch_size, z_dim)).to(device = device) 
        # so we will be able to see how it changes across the epochs
        # get us good viusalization in tensorboard
    transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))]
    )
    dataset = datasets.MNIST(root = "dataset/",transform=transforms,download=True)
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    opt_disc = optim.Adam(disc.parameters(),lr = lr)
    opt_gen = optim.Adam(gen.parameters(),lr = lr)
    criterion = nn.BCELoss()
    writer_fake = SummaryWriter(f"run/GAN_MNIST/fake")
    writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
    step = 0 # step for the tensorboard

    for epoch in range(num_epochs):
        for batch_idx, (real,_) in enumerate(loader):
            real = real.view(-1,784)  # reshape: keep the number of example in our batch and then flat the rest
            batch_size = real.shape[0]  # first dimension

            ### Train Discriminator: max long(D(real)) + log(1 - D(G(z)))   
                # z is some random noise
            # randn Gaussian distribution
            noise = torch.randn(batch_size,z_dim).to(device=device)
            fake = gen(noise) # max G(z)
            disc_real = disc(real).view(-1) # flatten everything
            lossD_real = criterion(disc_real,torch.ones_like(disc_real))
            disc_fake = disc(fake).view(-1)
            lossD_fake = criterion(disc_fake, torch.ones_like(disc_fake)) 
            lossD = (lossD_real + lossD_fake)/ 2
            disc.zero_grad()
            lossD.backward(retain_graph=True)
            opt_disc.step()

            #### Train Generator min log(1 - D(G(z))) <--> max log(D(G(z)))
            output = disc(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()

            # code for tensorboard
            if batch_idx == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}] \ "
                    f"Loss D:{lossD:.4f}, Loss G:{lossG:.4f}"
                )

                with torch.no_grad():
                    fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                    data = real.reshape(-1, 1, 28, 28)
                    img_grid_fake = torchvision.utils.make_grid(fake, normalize = True)
                    img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                    writer_fake.add_image(
                        "Mnist Fake Images", img_grid_fake, global_step=step
                    )
                    writer_real.add_image(
                        "Mnist real Images",img_grid_real, global_step=step
                    )

                    step += 1
            






