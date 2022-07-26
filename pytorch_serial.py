import torch
from torch.utils.data import DataLoader, Subset

import torchvision
from torchvision import transforms
from torchvision.models.detection.ssd import ssd300_vgg16

import torch.multiprocessing as mp

from util import draw_bounding_boxes, save_images

import pandas as pd

from tqdm import tqdm

def run_inference(rank, world_size, data, results):    
    partition_size = len(data) // world_size
    data = Subset(data, indices=list(range(rank*partition_size, (rank+1)*partition_size)))
    
    data_loader = DataLoader(data, batch_size=128, num_workers=1)
    
    device = torch.device(f"cuda:{rank}")
    
    model = ssd300_vgg16(pretrained=True)
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
    
    mp.spawn(run_inference,
        args=(world_size,data,results),
        nprocs=world_size,
        join=True)
    
    results = results[:5]
    
    df = pd.DataFrame(results)
    # Add back original images
    original_images = Subset(data, range(len(df)))
    df["image"] = [image[0].numpy() for image in original_images]
    
    objects = []
    
    for i in range(len(df)):
        objects.append(draw_bounding_boxes(df.iloc[i]))
        
    save_images(objects, "./object_detections_serial")