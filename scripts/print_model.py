import torch
import torchvision

def load_and_print_model(model_type, weights_path, device=None):
    """
    Load and print given model.

    Parameters:
    - model_type (str): "segmentation" or "video".
    - weights_path (str): path of weights file.
    - device (torch.device, optional): cuda or cpu.

    Return:
    - model: the loaded model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "segmentation":
        model_name = "deeplabv3_resnet50"
        model = torchvision.models.segmentation.__dict__[model_name](pretrained=False, aux_loss=False)
        model.classifier[-1] = torch.nn.Conv2d(
            model.classifier[-1].in_channels,
            1, 
            kernel_size=model.classifier[-1].kernel_size
        )

    elif model_type == "video":
        model_name = "r2plus1d_18" 
        model = torchvision.models.video.__dict__[model_name](pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 1) 

    else:
        raise ValueError("Invalid model_type. Choose 'segmentation' or 'video'.")

    if device.type == "cuda" and model_type == "segmentation":
        model = torch.nn.DataParallel(model)

    model.to(device)

    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint['state_dict']

    if model_type == "segmentation":
        model.load_state_dict(state_dict)
    elif model_type == "video":
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # remove 'module.' prefix
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

    model.eval()
    print(model)
    return model

if __name__ == "__main__":
    model_type = "video" # segmentation or video
    weights = 'E:/Ultrasound/EchoNet-Dynamic/dynamic-master-gpu/output/segmentation/deeplabv3_resnet50_random/best.pt'
    if model_type == "video":
        weights = 'E:/Ultrasound/EchoNet-Dynamic/dynamic-master-gpu/output/video/save/best.pt'
    load_and_print_model(model_type, weights)