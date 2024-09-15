# try normalising in the loader
# try noise instead of cloning og img for gen img

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

device = torch.device('cpu')
img_size=356

class modVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.chosen_feat=["0", "5", "10", "19", "28"]
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self,x):
        features=[]

        for layer_num,layer in enumerate(self.model):
            x=layer(x)

            if str(layer_num) in self.chosen_feat:
                features.append(x)

        return features
    
loader = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]
)
            
def image_loader(path):
    image = Image.open(path)
    image_us = loader(image).unsqueeze(0)
    return image_us.to(device)

og_img = image_loader('annahathaway.png')
style_img = image_loader('style.jpg')
gen_img = og_img.clone().requires_grad_(True)

steps=6000
lr =0.001
alpha = 1
beta = 0.01
optimizer = optim.Adam([gen_img],lr=lr)

model = modVGG().to(device).eval()

for s in range(steps):
    gen_feats = model(gen_img)
    og_feats = model(og_img)
    style_feats=model(style_img)

    for gf,of,sf in zip(gen_feats,og_feats,style_feats):
        og_loss=0
        og_loss += torch.mean((gf-of)**2)

        style_loss=0

        batch_size, c, h, w = gf.shape
        gram_gen = gf.view(c,h*w).mm(gf.view(c,h*w).t())

        gram_style = sf.view(c,h*w).mm(sf.view(c,h*w).t())

        style_loss= torch.mean((gram_gen-gram_style)**2)

    fin_loss = alpha*og_loss + beta*style_loss
    optimizer.zero_grad()
    fin_loss.backward()
    optimizer.step()

    if s % 200 ==0:
        print(fin_loss)
        save_image(gen_img,'gen_img.jpg')


