from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import os
from django.conf import settings


labelEncoder = {'5Hundrednote': 0, '1Hundrednote': 1, 'Twentynote': 2, '2Hundrednote': 3, 'Fiftynote': 4, 'Tennote': 5, '2Thousand': 6 }

reverseLabel = { v:k for k,v in labelEncoder.items() }


testTransforms = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor()
])

class CurrencyModel(nn.Module):
    def __init__(self, input_size=3, output_size=len(labelEncoder)):
        super().__init__()

        self.ConvBLK1 = nn.Sequential(
            nn.Conv2d(input_size, 32, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )

        self.ConvBLK2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=524288, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=output_size),

        )

    def forward(self, x):
        x = self.ConvBLK1(x)
        x = self.ConvBLK2(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


model = CurrencyModel()
model_path = os.path.join(settings.BASE_DIR, 'predict', 'models', 'currency(6).pth')
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

def predictPokemon(image):  # PIL.Image
    img = image.convert('RGB')
    transformed = testTransforms(img).unsqueeze(0)
    with torch.no_grad():
        output = model(transformed)
        predicted_index = output.argmax(dim=1).item()
    predicted_label = reverseLabel[predicted_index]
    readable = predicted_label.replace("note", " rupees note").replace("Hundred", " hundred").replace("Thousand", " thousand")
    return f"The given note is a {readable.replace('Tennote', '10 rupees note').replace('Twentynote', '20 rupees note')}."




if __name__ == "__main__":
    image_path = input("Enter the path to the image file: ").strip()
    predictPokemon(image_path)