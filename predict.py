import argparse
import json
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from PIL import Image

### python predict.py ./flowers/test/7/image_08099.jpg ./checkpoint.pth --topk 5 --category_names cat_to_name.json --gpu

def process_image(image):
    pil_image = Image.open(image).convert("RGB")    
    image_transforms = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], 
                [0.229, 0.224, 0.225]
            )
        ]
    )
    
    return image_transforms(pil_image)


def predict(image_path, model, use_gpu, topk):        
    topk=int(topk)
    with torch.no_grad():
        image = process_image(image_path)
        image = torch.from_numpy(np.asarray(image))
        image.unsqueeze_(0)
        image = image.float()
        
        if use_gpu:
           image = image.cuda()
           model = model.cuda()
        else:
            image = image.cpu()
            model = model.cpu()
            
        outputs = model(image)
        probs, classes = torch.exp(outputs).topk(topk)
        probs, classes = probs[0].tolist(), classes[0].add(1).tolist()
        
        return probs, classes

def main():
    parser = argparse.ArgumentParser()    
    parser.add_argument('input', type=str)
    parser.add_argument('checkpoint_path', type=str)
    parser.add_argument('--topk', type=int)
    parser.add_argument('--category_names', type=str)
    parser.add_argument('--gpu', dest='gpu', action='store_true', default=False)
    args = parser.parse_args()

    print('---' * 20)
    print('Loading Model Start ...')
    chp = torch.load(args.checkpoint_path)
    model = models.vgg16(pretrained=True)
    model.class_to_idx = chp['class_to_idx']
    model.classifier = chp['classifier']
    model.load_state_dict(chp['state_dict'])
    num_of_epochs = chp['num_of_epochs']
    print('---' * 20)    
    print('Loading Model End!!')  
    
    use_gpu = torch.cuda.is_available() and args.gpu
    device = torch.device("cuda" if use_gpu else "cpu")

    print('---' * 20)    
    print('current device: ', device)
    
    print('---' * 20)
    if args.category_names:
        print('Loading category names ...')
        with open(args.category_names, 'r') as f:
            categories = json.load(f)
            print("The category names are loaded. Total: ", len(categories))

    print('---' * 20)
    print('Prediction Start ...')
    print('---' * 20)
            
    optional_topk = args.topk if args.topk else 0
    
    probs, classes = predict(args.input, model, use_gpu, args.topk)
    
    if optional_topk == 0:
        print("class: {:<30}, probability: {:.2f}".format(categories[classes[0]] if args.category_names else classes[0], probs[0]))
        return Void
    
    i = 0
    for c in classes:
        key = str(c)
        if key in categories.keys():
            label = categories[key]
            print("class: {:<30}, probability: {:.2f}".format(label, probs[i]))
        i += 1
    
    print('---' * 20)
    print('Prediction End!!')
    print('---' * 20)
    
    
if __name__ == "__main__":
    main()