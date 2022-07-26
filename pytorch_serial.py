import torch
from torch.utils.data import DataLoader, Subset

import torchvision
from torchvision import transforms
from torchvision.models.detection.ssd import ssd300_vgg16, SSD300_VGG16_Weights

import torch.multiprocessing as mp

from util import draw_bounding_boxes, save_images

import pandas as pd

from tqdm import tqdm

def run_inference(data, results):    
    subset_size = len(data) // 50
    data = Subset(data, indices=list(range(subset_size)))
    
    data_loader = DataLoader(data, batch_size=96)
    
    device = torch.device(f"cuda:0")
    
    model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
    model = model.to(device)
    model.eval()
    
    for i, batch in tqdm(enumerate(data_loader)):
        images, _ = batch
        images = images.to(device)
        output = model(images)
        for d in output:
            for k, v in d.items():
                d[k] = v.cpu().detach().numpy()
        results.extend(output)
    

    
if __name__ == "__main__":
    world_size = 4
    results = mp.Manager().list()
    
    torchvision_transforms = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor()])
    data = torchvision.datasets.ImageFolder("/home/ec2-user/SageMaker/movie_posters", transform=torchvision_transforms)
    
    run_inference(data, results)
    
    results = results[:5]
    
    df = pd.DataFrame(results)
    # Add back original images
    original_images = Subset(data, range(len(df)))
    df["image"] = [image[0].numpy() for image in original_images]
    
    objects = []
    
    for i in range(len(df)):
        objects.append(draw_bounding_boxes(df.iloc[i]))
        
    save_images(objects, "./object_detections_serial")