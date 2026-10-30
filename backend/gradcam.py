import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

class GradCAMPlusPlus:
    """Grad-CAM++: pondera el gradiente por píxel, así marca mejor varias zonas a la vez."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, target_class=None):
        """Devuelve el mapa de activación normalizado a [0, 1]."""
        self.model.eval()
        output = self.model(input_image)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()

        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        # pesos alpha de grad-cam++: por píxel, en vez del promedio simple de grad-cam
        grad_2 = gradients.pow(2)
        grad_3 = gradients.pow(3)

        sum_activations = activations.sum(dim=(1, 2), keepdim=True)

        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8
        alpha = alpha_num / alpha_denom

        weights = (alpha * F.relu(gradients)).sum(dim=(1, 2), keepdim=True)

        cam = (weights * activations).sum(dim=0)

        cam = F.relu(cam)
        cam = cam.cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam

    def overlay_heatmap(self, original_image, cam, alpha=0.4, colormap=cv2.COLORMAP_JET):
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)

        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlayed = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)

        return Image.fromarray(overlayed)


def create_gradcam_for_densenet(model):
    """Grad-CAM++ sobre el último bloque denso de DenseNet121."""
    target_layer = model.features.denseblock4
    return GradCAMPlusPlus(model, target_layer)