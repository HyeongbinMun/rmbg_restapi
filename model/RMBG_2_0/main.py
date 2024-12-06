import torch, cv2
import numpy as np
import warnings
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

warnings.filterwarnings("ignore", category=FutureWarning, module="kornia.feature.lightglue")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.registry")

class RMBG2:
    def __init__(self, model_name='briaai/RMBG-2.0', image_size=(1024, 1024), device='cuda'):
        self.device = device
        self.model = AutoModelForImageSegmentation.from_pretrained(model_name, trust_remote_code=True)
        torch.set_float32_matmul_precision('high')
        self.model.to(self.device)
        self.model.eval()

        self.image_size = image_size
        self.transform_image = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def inference(self, image: Image.Image):
        """
        Perform inference on the given image and return the result as a numpy array.
        :param image: Input PIL Image
        :return: Processed image with background removed (numpy array)
        """
        out_images = []
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        input_image = self.transform_image(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self.model(input_image)[-1].sigmoid().cpu()

        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        image.putalpha(mask)
        out_images.append(image)

        return out_images


# analyzer = RMBG2()
# image = cv2.imread('/workspace/model/RMBG_2_0/images/34.jpg')
# out_images = analyzer.inference(image)
# out_images[0].save('/workspace/model/RMBG_2_0/images/results.png')