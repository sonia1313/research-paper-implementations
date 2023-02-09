import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

class VGG(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_maps = ["0","5","10","19","28"]
        self.model = models.vgg19(pretrained=True).features[:29]

        print(self.model)

    def forward(self,x):
        features = []

        for n_layer, layer in enumerate(self.model):
            x = layer(x) #?

            if str(n_layer) in self.feature_maps:
                features.append(x)
        return features


def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)

    return image.to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_size = 200

loader = transforms.Compose([transforms.Resize((img_size,img_size)),
                            transforms.ToTensor()]
                            )
orignial_image = load_image("annahathaway_content_image.png")
style_image = load_image("style_img.jpg")

generated = orignial_image.clone().requires_grad_(True)
model = VGG().to(device).eval()

#Hyper-parameters
total_steps = 6000
lr = 0.001
alpha = 1
beta = 0.01
optimizer = optim.Adam([generated], lr = lr)

for step in range(total_steps):
    generated_features = model(generated)
    original_img_features = model(orignial_image)
    style_features = model(style_image)

    style_loss = original_loss = 0

    for gen_feature, orig_feature, style_feature in zip(generated_features,original_img_features,style_features):
        batch_size, channel, h, w = gen_feature.shape
        original_loss += torch.mean((gen_feature-orig_feature)**2)

        G = gen_feature.view(channel,h * w).mm(
            gen_feature.view(channel,h*w).t()
        )

        A = style_feature.view(channel,h*w).mm(
            style_feature.view(channel,h*w).t()
        )

        style_loss += torch.mean((G-A)**2)

    total_loss = alpha * original_loss + beta * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(total_loss)
        save_image(generated,f"generated_{step}.png")