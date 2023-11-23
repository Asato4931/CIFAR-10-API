import torch
import torch.nn as nn
import torchvision.transforms as transforms
from fastapi import FastAPI, UploadFile
from PIL import Image
import io
import torch.nn.functional as F


# 作ったモデルの呼び出し

# シードの固定
torch.manual_seed(4931)


class SimpleCNN(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(x)
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(x)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x


net = SimpleCNN()
net.eval()

# Load the state dictionary into the model
model_state_dict_path = (
    "/Users/asato/Desktop/CA Tech CV課題/CA Tech CV 1/simple_cnn_model_state_dict.pth"
)

# Load the state dictionary into the model without DataParallel
state_dict = torch.load(model_state_dict_path, map_location=torch.device("cpu"))
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
net.load_state_dict(state_dict)


def predict(model, image_tensor):
    transform_pred = transforms.Compose(
        [
            transforms.Resize(32),
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
