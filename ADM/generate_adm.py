import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from gradcam_utils import GradCAM, show_cam_on_image
from model import resnet34
import pickle
import cv2
import glob
import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if os.path.exists("datasets/thresholds.pkl"):
    with open("datasets/thresholds.pkl", "rb") as f:
        thresholds = pickle.load(f)


def get_model():
    model = resnet34(num_classes=14).cuda()
    model_weight_path = "./weights/MIMIC_best_weight.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location="cpu"))
    return model.eval()


def main(model, image_path, seg_path, mask_path, array_path):
    target_layers = [model.layer4]
    data_transform = transforms.Compose([transforms.Resize(300),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    assert os.path.exists(image_path), "file: '{}' dose not exist.".format(image_path)
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img, dtype=np.uint8)
    img_tensor = data_transform(img)
    input_tensor = torch.unsqueeze(img_tensor, dim=0).cuda()
    logit = model(input_tensor)  # [64, 1, 768]
    thresholded_predictions = 1 * (logit.detach().cpu().numpy() > thresholds)
    indices = np.where(thresholded_predictions[0] == 1)[0]

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    mask_arr_ass = np.zeros((300, 300, 3))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    for target_category in list(indices):
        grayscale_cam = cam(input_tensor=input_tensor, target_category=int(target_category))

        grayscale_cam = grayscale_cam[0, :]
        _, heatmap = show_cam_on_image(img_np.astype(dtype=np.float32) / 255.,
                                       grayscale_cam,
                                       use_rgb=True)
        threshold = 0.6
        mask = cv2.threshold(heatmap, threshold, 1, cv2.THRESH_BINARY)[1]
        mask = cv2.dilate(mask, kernel)
        mask = mask.astype(np.uint8) * 255
        mask_arr = np.asarray(mask)
        mask_arr_ass += mask_arr
    mask_arr_ass = np.any(mask_arr_ass, axis=2)
    np.save(array_path, mask_arr_ass)
    mask = Image.fromarray((mask_arr_ass * 255).astype(np.uint8)).convert('L')
    mask.save(mask_path)
    img = Image.open(image_path)
    img = mask * np.asarray(img)
    img = Image.fromarray(img)
    img.save(seg_path)


if __name__ == '__main__':
    image_list = glob.glob("../dataset/mimic_cxr/images300/*/*/*/*.jpg")
    model = get_model()
    bar = tqdm.tqdm(image_list)
    for image_path in bar:
        seg_path = image_path.replace("images300", "resnet34_300/images300_seg")
        mask_path = image_path.replace("images300", "resnet34_300/images300_mask")
        array_path = image_path.replace("images300", "resnet34_300/images300_array").replace(".jpg", ".npy")

        if not os.path.exists(os.path.dirname(seg_path)):
            os.makedirs(os.path.dirname(seg_path))
            os.makedirs(os.path.dirname(mask_path))
            os.makedirs(os.path.dirname(array_path))
        if not os.path.exists(seg_path):
            main(model, image_path, seg_path, mask_path, array_path)
