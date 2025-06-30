# -Neural-Style-Transfer
pip install torch torchvision pillow matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load image
def load_image(img_path, max_size=400, shape=None):
    image = Image.open(img_path).convert('RGB')

    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    in_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    image = in_transform(image).unsqueeze(0)
    return image

# Display image
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    image = image.clip(0, 1)
    return image

# Load content and style images
content = load_image('content.jpg')
style = load_image('style.jpg', shape=content.shape[-2:])

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
content = content.to(device)
style = style.to(device)

# Load pretrained VGG19
vgg = models.vgg19(pretrained=True).features.to(device).eval()

# Freeze parameters
for param in vgg.parameters():
    param.requires_grad_(False)

# Get features
def get_features(image, model, layers=None):
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  # content layer
            '28': 'conv5_1'
        }

    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features

# Gram Matrix
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# Get features
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# Calculate style grams
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# Create target image
target = content.clone().requires_grad_(True).to(device)

# Loss weights
style_weight = 1e6
content_weight = 1

# Optimizer
optimizer = optim.Adam([target], lr=0.003)

# Training
epochs = 2000
for i in range(epochs):
    target_features = get_features(target, vgg)
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

    style_loss = 0
    for layer in style_grams:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        layer_style_loss = torch.mean((target_gram - style_gram) ** 2)
        _, d, h, w = target_feature.shape
        style_loss += layer_style_loss / (d * h * w)

    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print(f"Epoch {i}, Total loss: {total_loss.item():.4f}")

# Display final output
final_img = im_convert(target)
plt.imshow(final_img)
plt.axis("off")
plt.title("Stylized Image")
plt.show()
