import torch
import torchvision
from dataset import BMWSemanticDataset
from torch.utils.data import DataLoader
from pre_processing import create_id_color_map

num_classes = 10
id_color_map = create_id_color_map()
id_color_map[num_classes-1] = (0,0,0) # black for background, unlabeled, other

def save_checkpoint(state, filename):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = BMWSemanticDataset(
        train_dir,
        train_maskdir,
        train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = BMWSemanticDataset(
        val_dir,
        val_maskdir,
        val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader
    
def calculate_metrics(predicted_masks, true_masks):
    smooth = 1e-6
    predicted_masks = predicted_masks.int()
    true_masks = true_masks.int()
    intersection = (predicted_masks & true_masks).float().sum((1, 2))
    union = (predicted_masks | true_masks).float().sum((1, 2))
    iou = (intersection + smooth) / (union + smooth)
    
    tp = (predicted_masks & true_masks).float().sum((1, 2))
    fp = (predicted_masks & ~true_masks).float().sum((1, 2))
    fn = (~predicted_masks & true_masks).float().sum((1, 2))

    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)

    return (
        iou.mean().item(),
        precision.mean().item(),
        recall.mean().item(),
    )

def conv_prediction_img(mask_prediction, id_color_map=id_color_map):
    """
    Converts a batch of class probability tensors to images.

    Args:
        mask_prediction: A torch tensor of shape [batch_size, num_classes, height, width] with class probabilities.
        id_color_map: A dictionary mapping class indices to color values.

    Returns:
        A torch tensor of shape [batch_size, rgb_channels=3, height, width] representing the images.
    """
    # Take the argmax over the class dimension to get the most probable class for each pixel
    max_class = torch.argmax(mask_prediction, dim=1)  # Shape [batch_size, height, width]

    # Initialize an output image tensor with the same batch size, height, and width, but with 3 channels for RGB
    batch_size, height, width = max_class.shape
    output_image = torch.zeros((batch_size, 3, height, width), dtype=torch.uint8)

    # Vectorized mapping of class indices to RGB colors for each image in the batch
    for class_idx, color in id_color_map.items():
        if isinstance(color, str):
            # Convert the color string to a tuple of integers
            color = tuple(map(int, color.split(',')))
        output_image[max_class == class_idx] = torch.tensor(color, dtype=torch.uint8)

    return output_image
    
def save_predictions_as_imgs(loader, model, folder, device, id_color_map = id_color_map):
    """
    Saves model predictions as RGB images after converting class probability tensors to semantic images.

    Args:
        loader: DataLoader providing batches of images and ground truth masks.
        model: The trained model used for making predictions.
        folder: Path to save the output prediction images.
        device: The device (CPU or GPU) on which computations will be carried out.
        id_color_map: A dictionary mapping class indices to RGB color values.

    Returns:
        None. Saves the predicted masks to the specified folder.
    """
    model.eval() # evaluation mode
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = model(x)
            preds_rgb = conv_prediction_img(preds, id_color_map)
        torchvision.utils.save_image(preds_rgb, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")