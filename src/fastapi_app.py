import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torchvision.transforms as transforms
import uvicorn
from fastapi import FastAPI
from src.predict import predict
from fastapi import File
from fastapi import UploadFile
import yaml
import numpy as np
from io import BytesIO
from PIL import Image
import skimage
import cv2


def get_model():
    """TODO: replace the hardcoded linee

    Returns:
        _type_: _description_
    """
    cfg_file = open("cfg/cfg.yaml")
    cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)
    trained_model_path = cfg["save_model_path"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(trained_model_path, map_location=torch.device(device))
    
    return model


async def load_image(file: UploadFile) -> np.ndarray:
    """Loads the images asychronously

    Args:
        file (UploadFile): File from Fast API

    Returns:
        np.ndarray: Loaded image
    """
    contents = await file.read()
    # image = skimage.io.imread(BytesIO(contents)) # why this no work?
    image = np.asarray(Image.open(BytesIO(contents)))
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # print (image.shape)
    return image


def process_pil_image(img_array: np.ndarray) -> torch.Tensor:
    """Processes an image that was read as PIL image and converted to np array.

    Processing -
    1. Makes it tensor
    2. Permute it so that it can be fed as (3, H, W)
    3. Resize it to 224 x 224
    4. Reshape to be (1,3,224,224)
    5. Normalise the channel intensity

    Args:
        img_array (np.ndarray): _description_

    Returns:
        torch.Tensor: _description_
    """
    img_array = torch.tensor(img_array)
    img_array = torch.permute(img_array, (2, 0, 1))
    img_array = transforms.functional.resize(img_array, (224, 224))
    img_array = torch.reshape(img_array, (1, 3, 224, 224))
    img_array = img_array / 255
    return img_array


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}


@app.post("/predict")
async def run_inference(file: UploadFile):
    img_array = await load_image(file)
    img_array = process_pil_image(img_array)
    model = get_model()
    pred = predict(model, img_array)
    return str(int(pred))


if __name__ == "__main__":
    uvicorn.run("src.fastapi_app:app")
