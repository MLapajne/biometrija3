import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score
import torch.nn as nn
import os
import cv2
from PIL import Image
from skimage import feature
from scipy.spatial.distance import cosine
from skimage.feature import hog

# Load the trained model
model = models.resnet50(pretrained=True)
model.fc = torch.nn.Linear(in_features=2048, out_features=136)
model.load_state_dict(torch.load('best_model.pth',  map_location=torch.device('cpu')))
feature_extractor = torch.nn.Sequential(*list(model.children())[:-3])
# Set the model to evaluation mode
model.eval()
feature_extractor.eval()


base_path_images = os.path.join('datasets', 'ears', 'images-cropped','train')
 # Replace with your base directory
# Load the test images and compute the ResNet feature vectors
# Get a sorted list of all image files



def calculate_resnet(img_name, resize_size, crop_size, normalize_mean, normalize_std):
    
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])
    img = Image.open(img_name)
    img = img.resize((100, 100))
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    out = feature_extractor(batch_t)
    return out.detach().numpy().flatten()



def test_calculate_resnet():
    # Define ranges of parameters to test
    resize_sizes = [(256, 256), (300, 300)]
    crop_sizes = [224, 200]
    normalize_means = [[0.485, 0.456, 0.406], [0.5, 0.5, 0.5]]
    normalize_stds = [[0.229, 0.224, 0.225], [0.5, 0.5, 0.5]]
    accuracy = 0
    args = []
    for resize_size in resize_sizes:
        for crop_size in crop_sizes: 
            for normalize_mean in normalize_means:
                for normalize_std in normalize_stds:
                    output = compare_img([resize_size, crop_size, normalize_mean, normalize_std])
                    print(f"Output for parameters resize_size={resize_size}, crop_size={crop_size}, normalize_mean={normalize_mean}, normalize_std={normalize_std}: {output}")
                    if accuracy < output:
                        accuracy = output
                        args = [resize_size, crop_size, normalize_mean, normalize_std]
    return [output, args]

def test_calculate_hog():
    # Define ranges of parameters to test

    orientations = [8, 9]
    pixels_per_cells = [(8, 8), (16, 16)]
    cells_per_blocks = [(2, 2), (4, 4)]
    block_norms = ['L2', 'L2-Hys']
    accuracy = 0
    args = []
    for orientation in orientations:
        for pixels_per_cell in pixels_per_cells:
            for cells_per_block in cells_per_blocks:
                for block_norm in block_norms:
                    #hog_features = calculate_hog(img_name, orientations, pixels_per_cell, cells_per_block, block_norm)
                    output = compare_img([orientation, pixels_per_cell, cells_per_block, block_norm])
                    print(f"orientation={orientation}, pixels_per_cell={pixels_per_cell}, cells_per_block={cells_per_block}, block_norm={block_norm}: {output}")

                    if accuracy < output:
                        accuracy = output
                        args = [orientation, pixels_per_cell, cells_per_block, block_norm]
                        #print(f"orientation={orientations}, pixels_per_cell={pixels_per_cell}, cells_per_block={cells_per_block}, block_norm={block_norm}: {output}")
    return [output, args]
     
def calculate_hog(img_name, orientations, pixels_per_cell, cells_per_block, block_norm):
    img = cv2.imread(img_name)
    img = cv2.resize(img, (100, 100), interpolation = cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_features = hog(img, orientations, pixels_per_cell, cells_per_block, block_norm)
    hog_ver = hog_features.flatten()
    return hog_ver

class ImageComparator:
    def __init__(self):
        self.num_detected_img = 0
        
    


    def compare_images(self, distance):
        if 0 < distance < 0.2:
            self.num_detected_img += 1

    def calculate_accuracy(self, all_img):
        if self.num_detected_img > 0:
            lbp_accuracy_my = (self.num_detected_img / all_img) * 100
        else:
            lbp_accuracy_my = 0
        return lbp_accuracy_my


def calculate_lbp(img_name, resize_size, crop_size, normalize_mean, normalize_std):
    img = cv2.imread(img_name)
    img = cv2.resize(img, (100, 100), interpolation = cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp_pattern1 = feature.local_binary_pattern(img)
    lbp_pattern1 = lbp_pattern1.flatten()
    return lbp_pattern1


def compare_img(args):
    image_files = sorted([f for f in os.listdir(base_path_images) if f.endswith('.jpg') or f.endswith('.png')])
    number_of_images = 0
    #image_comparator_lib = ImageComparator()
    image_comparator_res = ImageComparator()
    #image_comparator_hog = ImageComparator()


    for image_file1 in image_files:
        image_path1 = os.path.join(base_path_images, image_file1)

        #lbp_output1 = calculate_lbp(image_path1, args)
        output_res1 = calculate_resnet(image_path1, *args)
        #output_hog1 = calculate_hog(image_path1, *args)
        for image_file2 in image_files:
            image_path2 = os.path.join(base_path_images, image_file2)

            
            


            if image_file1.split('-')[0] == image_file2.split('-')[0]:
                if (image_file1 != image_file2):
                    
                    
                    number_of_images += 1
                    #lbp_output2 = calculate_lbp(image_path2, P, R, method)
                    output_res2 = calculate_resnet(image_path2, *args)
                    #output_hog2 = calculate_hog(image_path2, *args)
                    #distance_lib = cosine(lbp_output1, lbp_output2)
                    distance_res = cosine(output_res1, output_res2)
                    #distance_hog = cosine(output_hog1, output_hog2)

                    #image_comparator_lib.compare_images(distance_lib)
                    image_comparator_res.compare_images(distance_res)
                    #image_comparator_hog.compare_images(distance_hog)

    #lbp_accuracy = image_comparator_lib.calculate_accuracy(number_of_images)
    res_accuracy = image_comparator_res.calculate_accuracy(number_of_images)
    #accuracy_hog = image_comparator_hog.calculate_accuracy(number_of_images)

    return res_accuracy



#values_detected = test_calculate_resnet()
#print("Values_detected:")
#print(f"Output for parameters resize_size={values_detected[1][0]}, crop_size={values_detected[1][1]}, normalize_mean={values_detected[1][2]}, normalize_std={values_detected[1][3]}: {values_detected[0]}")

values_detected = test_calculate_resnet()
print("Values_detected:")
print(values_detected)

