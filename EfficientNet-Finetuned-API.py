import torch
import torch.nn as nn
import torchvision.transforms as transforms
from fastapi import FastAPI, UploadFile
from PIL import Image
import io
import torch.nn.functional as F

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets, transforms, models
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


# 作ったモデルの呼び出し

effmodel = models.efficientnet_v2_s()

effmodel.classifier = torch.nn.Identity()

# シードの固定
torch.manual_seed(4931)


class EffNetFT(nn.Module):
    def __init__(self, dropout_rate=0.55):
        super(EffNetFT, self).__init__()
        self.conv = effmodel
        self.fc = nn.Linear(1280, 10)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


net = EffNetFT()
net.eval()

# Load the state dictionary into the model
model_state_dict_path = (
    "/Users/asato/Desktop/CA Tech CV課題/CA Tech CV 1/efficient_net_cifar10_finetuned.pth"
)


# Load the state dictionary into the model without DataParallel
state_dict = torch.load(model_state_dict_path, map_location=torch.device("cpu"))
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
net.load_state_dict(state_dict)


def predict(model, image_tensor):
    transform_pred = transforms.Compose(
        [
            transforms.Resize(400),
            transforms.CenterCrop(384),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    image_tensor = transform_pred(image_tensor)

    # Add an extra batch dimension since PyTorch treats all inputs as batches
    image_tensor = image_tensor.unsqueeze(0)

    # Forward pass through the model
    outputs = model(image_tensor)

    # Apply softmax to get probabilities (学習中の交差エントロピー損失は、Softmax機能が備わっていた)
    probabilities = F.softmax(outputs, dim=1)

    # Get the predicted class and its confidence
    _, predicted_class = torch.max(probabilities, 1)
    confidence = probabilities[0][predicted_class].item()

    return predicted_class.item(), confidence


# レスポンスボディの指定


app = FastAPI()  # サーバーのインスタンスを作成


@app.post("/results")
async def results(file: UploadFile):
    # Read image data as bytes
    image_data = await file.read()
    # Convert bytes data to a PIL Image
    image = Image.open(io.BytesIO(image_data))

    transform = transforms.ToTensor()

    # Apply the transformation to convert the PIL Image to a tensor
    image_tensor = transform(image)

    predicted_class, confidence = predict(net, image_tensor)

    return {f"predictions: {predicted_class} score: {confidence}"}
