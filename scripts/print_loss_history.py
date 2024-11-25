import os
import torch
import pandas as pd
import matplotlib.pyplot as plt

def print_one_loss_history(model_type):
    """
    Draw figure of loss history over epoch and save to local.
    Save loss history to local excel file.

    Parameters:
    - model_type (str): "segmentation" or "video".
    """
    # Load checkpoint
    if model_type == "segmentation":
        checkpoint_path = 'E:/Ultrasound/EchoNet-Dynamic/dynamic-master-gpu/output/segmentation/deeplabv3_resnet50_random/checkpoint.pt' # for segmentation
    else:
        checkpoint_path ='E:/Ultrasound/EchoNet-Dynamic/dynamic-master-gpu/output/video/r2plus1d_18_32_2_pretrained/checkpoint.pt'
    
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))

    epoch = checkpoint.get('epoch', None)
    loss = checkpoint.get('loss_history', None)

    # Save loss history to local excel file
    loss_df = pd.DataFrame({'Epoch': list(range(1, len(loss) + 1)), 'Loss': loss})
    output_dir = 'loss_history'
    os.makedirs(output_dir, exist_ok=True)
    excel_path = os.path.join(output_dir, model_type + '_loss_history.xlsx')
    loss_df.to_excel(excel_path, index=False)

    # Draw the figure
    plt.plot(loss_df['Epoch'], loss_df['Loss'], marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epoch - ' + model_type)
    plt.grid(True)
    figure_path = os.path.join(output_dir, model_type + '_loss_history.png')
    plt.savefig(figure_path)
    plt.show()

def print_two_loss_history():
    """
    Draw figure of normalized loss history over epoch of both models and save to local.
    Save both models' normalized loss history to local excel file.
    """
    # Load two models
    checkpoint_path_1 = 'E:/Ultrasound/EchoNet-Dynamic/dynamic-master-gpu/output/segmentation/deeplabv3_resnet50_random/checkpoint.pt'
    checkpoint_path_2 = 'E:/Ultrasound/EchoNet-Dynamic/dynamic-master-gpu/output/video/r2plus1d_18_32_2_pretrained/checkpoint.pt'
    checkpoint_1 = torch.load(checkpoint_path_1, map_location=torch.device('cuda'))
    checkpoint_2 = torch.load(checkpoint_path_2, map_location=torch.device('cuda'))
    loss_1 = checkpoint_1.get('loss_history', None)[:40]
    loss_2 = checkpoint_2.get('loss_history', None)[:40]

    # Find the minimum value and the corresponding epoch
    min_loss_1 = min(loss_1)
    min_loss_2 = min(loss_2)
    min_epoch_1 = loss_1.index(min_loss_1) + 1  # epoch starts from 1
    min_epoch_2 = loss_2.index(min_loss_2) + 1  
    print(f'Minimum Loss of Segmentation Model: {min_loss_1:.4f} with epoch {min_epoch_1}')
    print(f'Minimum Loss of Video Model: {min_loss_2:.4f} with epoch {min_epoch_2}')

    # Normalization
    loss_1_min, loss_1_max = min(loss_1), max(loss_1)
    loss_2_min, loss_2_max = min(loss_2), max(loss_2)
    loss_1_normalized = [(l - loss_1_min) / (loss_1_max - loss_1_min) for l in loss_1]
    loss_2_normalized = [(l - loss_2_min) / (loss_2_max - loss_2_min) for l in loss_2]

    # Save loss history to local excel file
    loss_df = pd.DataFrame({'Epoch': list(range(1, len(loss_1_normalized) + 1)), 
        'Loss_segmentation': loss_1, 
        'Loss_video': loss_2,
        'Loss_segmentation_normalized': loss_1_normalized, 
        'Loss_video_normalized': loss_2_normalized,})
    output_dir = 'loss_history'
    os.makedirs(output_dir, exist_ok=True)
    excel_path = os.path.join(output_dir, 'all_models_loss_history.xlsx')
    loss_df.to_excel(excel_path, index=False)

    # Draw the figure
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_1_normalized) + 1), loss_1_normalized, marker='o', color='blue', label='Left Ventricle Segmentation Model Loss (Normalized)')
    plt.plot(range(1, len(loss_2_normalized) + 1), loss_2_normalized, marker='x', color='orange', label='EF Prediction Model Loss (Normalized)')
    plt.xlabel('Epoch')
    plt.ylabel('Normalized Loss')
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    
    figure_path = os.path.join(output_dir, 'all_models_normalized_loss_history.png')
    plt.savefig(figure_path)
    plt.show()

if __name__ == "__main__":
    print_one_loss_history("video")
    # print_two_loss_history()