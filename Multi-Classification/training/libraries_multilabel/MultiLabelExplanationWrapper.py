import torch
from torch import Tensor
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import numpy as np
import pandas as pd
import random
import warnings
from typing import Dict, Any
from bcos.common import BcosUtilMixin
import torch.nn.functional as F

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)


class CustomModelWrapper(BcosUtilMixin):
    def __init__(self, pretrained_model):
        super().__init__()
        self.model = pretrained_model

    def forward(self, x):
        return self.model(x)

    def modules(self):
        return self.model.modules()


    # need to change to sigmoid!!!! for actually using it
    # Override the explain method to use the model attribute correctly
    def explain(
        self,
        in_tensor,
        idx=None,
        explain_all_classes=False,
        threshold=0.5,
        **grad2img_kwargs,
    ) -> "Dict[str, Any]":
        # Call the explain method from CustomBcosMixin but use self.model for predictions
        if in_tensor.ndim == 3:
            raise ValueError("Expected 4-dimensional input tensor")
        if in_tensor.shape[0] != 1:
            raise ValueError("Expected batch size of 1")
        if not in_tensor.requires_grad:
            warnings.warn(
                "Input tensor did not require grad! Has been set automatically to True!"
            )
            in_tensor.requires_grad = True  # nonsense otherwise
        if self.model.training:  # noqa
            warnings.warn(
                "Model is in training mode! "
                "This might lead to unexpected results! Use model.eval()!"
            )

        result = dict()
        self.model.eval()
        with torch.enable_grad(), self.explanation_mode():
            # Manually set explanation mode for the wrapped model
            for m in self.model.modules():
                if hasattr(m, "set_explanation_mode"):
                    m.set_explanation_mode(True)

            # fwd + prediction using self.model
            out = self.model(in_tensor)  # Use self.model for predictions
            probs = torch.sigmoid(out)
            
            binary_preds = (probs > 0.5).int()

            result["probabilities"] = probs.detach().cpu()
            result["binary_predictions"] = binary_preds.detach().cpu()

            grads = []
            contribution_maps = []

            for class_idx in range(out.shape[1]):
                grad = torch.autograd.grad(
                    outputs=out[0, class_idx],
                    inputs=in_tensor,
                    retain_graph=True,
                    create_graph=False,
                    only_inputs=True,
                )[0]
                
                grads.append(grad)
                contribution_maps.append((in_tensor * grad).sum(1).squeeze(0))

            result["contribution_maps"] = torch.stack(contribution_maps)
            result["dynamic_linear_weights"] = torch.stack(grads)

            result["explanations"] = {}
            if explain_all_classes:
                # Generate explanations for all classes
                for idx in range(out.shape[1]):
                    result["explanations"][f"class_{idx}"] = self.gradient_to_image(
                        in_tensor[0], grads[idx][0], **grad2img_kwargs
                    )
            else:
                # Generate explanations only for predicted classes
                active_classes = torch.where(binary_preds)[0]
                for idx in active_classes:
                    result["explanations"][f"class_{idx}"] = self.gradient_to_image(
                        in_tensor[0], grads[idx][0], **grad2img_kwargs
                    )

            # Manually reset explanation mode for the wrapped model
            for m in self.model.modules():
                if hasattr(m, "set_explanation_mode"):
                    m.set_explanation_mode(False)

        return result