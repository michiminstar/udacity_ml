import argparse
import numpy as np
import torch
from functions import predict, load_checkpoint
from utils import process_image
import json

def main():
    parser = argparse.ArgumentParser(description='This program predicts a flower name from an image')
    parser.add_argument('image_path', type=str, help='Image path')
    parser.add_argument('checkpoint', type=str, help='Checkpoint')
    parser.add_argument('--top_k', type=int, default='1', help='Top k probablities')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Mappings of indices and class names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU', default=False)
    
    args = parser.parse_args()    
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu') if args.gpu else 'cpu'
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

        model_load, optimizer_load, scheduler_load, epoch= load_checkpoint(device, args.checkpoint)
        
        probs, classes, flowers = predict(model_load, process_image(args.image_path), cat_to_name, args.top_k)
        
        print("Predictions for {}: {}".format(args.image_path, flowers))
                
        probs = np.array(probs)[0]
        print("Probablities: {}".format(probs))
        

if __name__ == '__main__':
    main() 
