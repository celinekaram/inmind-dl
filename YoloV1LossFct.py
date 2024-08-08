import torch
from torch import nn
# from torchsummary import summary
# from utils import intersection_over_union
# from torchmetrics.detection import IntersectionOverUnion

def intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # compute the area of intersection rectangle
    interArea = (xB - xA) * (yB - yA)
    
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    # return the intersection over union value
    return iou

class YoloV1Loss(nn.Module):

    def __init__(self, S=7, B=2, C=20):
        super(YoloV1Loss, self).__init__()
        
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B # number of bounding boxes 
        self.C = C # number of class scores

        # Pay losses:
        # Increase loss from bounding box coordinate predictions
        self.lambda_coord = 5
        # Decrease loss from confidence predictions for boxes that don’t contain objects
        self.lambda_noobj = 0.5

    # target is ground_truth
    
    def forward(self, predictions, target):
        # predictions: BATCH_SIZE, S*S(C+B*5) => N, S, S, C+B*5
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        """ 
        * Prediction *:
        0 -> C-1 : class probabilities
        1st bounding box: C (confidence_score_1) | C+1 (x1) | C+2( y1) | C+3 (w1) | C+4 (h1)
        2nd bounding box: C+5 (confidence_score_2) | C+6 (x2) | C+7( y2) | C+8 (w2) | C+9 (h2)
        
        * Target * has only 1 bounding box, similar to indexes of 1st bounding box
        """

        # Calculate IoU for the 2 predicted bounding boxes with target bbox
        # ... : ellipsis, extract elements along the last axis for each item
        # [..., 2] = [: ,: ,: ,2]
        target_bb = target[..., self.C + 1 : self.C + 5]
        pred_bb1 = predictions[..., self.C + 1 : self.C + 5]
        pred_bb2 = predictions[..., self.C + 6 : self.C + 10]
        
        iou_b1 = intersection_over_union(pred_bb1,  target_bb )
        iou_b2 = intersection_over_union(pred_bb2,  target_bb )
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # Take the box with highest IoU out of the two prediction
        # bestbox will be indices of 0, 1 for which bbox was best
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., self.C].unsqueeze(3)  # identity_obj_i (is there an object in cell i ?)

        #   FOR BOX COORDINATES    #
        
        # if bestbox = 1 (2nd bb is the best), if bestbox = 0, (1st bb is the best)
        box_predictions = exists_box * (bestbox * pred_bb2 + (1 - bestbox) * pred_bb1)
        box_targets = exists_box * target_bb

        # Take sqrt of width, height of boxes (index 2 and 3)
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))
        # add 1e-6 in case there is a 0 value
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            # N*S*S, 4
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        pred_conf_bb1 = predictions[..., self.C : self.C + 1]
        pred_conf_bb2 = predictions[..., self.C + 5 : self.C + 6]
        target_conf_bb = target[..., self.C:self.C + 1]
        
        #   FOR OBJECT LOSS    # (if object exists ! )

        # pred_box is the confidence score for the bbox with highest IoU
        pred_box = bestbox * pred_conf_bb2 + (1 - bestbox) * pred_conf_bb1
        
        # Calculate loss with best box (with highest IOU)

        object_loss = self.mse(
            # N*S*S, 1
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target_conf_bb),
        )

        #   FOR NO OBJECT LOSS    # (if object does not exist ! )
        
        # Calculate loss of Bounding Box 1

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * pred_conf_bb1, start_dim=1),
            torch.flatten((1 - exists_box) * target_conf_bb, start_dim=1),
        )

        # Add loss of Bounding Box 2

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * pred_conf_bb2, start_dim=1),
            torch.flatten((1 - exists_box) * target_conf_bb, start_dim=1)
        )

        #   FOR CLASS LOSS   #

        class_loss = self.mse(
            # N, S, S, C -> N*S*S, C
            torch.flatten(exists_box * predictions[..., 0 : self.C], end_dim=-2,),
            torch.flatten(exists_box * target[..., 0 : self.C], end_dim=-2,),
        )

        loss = self.lambda_coord * box_loss + object_loss  + self.lambda_noobj * no_object_loss + class_loss
        return loss