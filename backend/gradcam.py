import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

class GradCAMPlusPlus:
    """Grad-CAM++ - Mejor que Grad-CAM para múltiples objetos"""
    
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
        """
        Genera CAM usando Grad-CAM++ (mejor que Grad-CAM estándar)
        """
        self.model.eval()
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # ===== GRAD-CAM++ WEIGHTS (diferencia clave) =====
        # Calcular alpha (pesos adaptativos por píxel)
        grad_2 = gradients.pow(2)
        grad_3 = gradients.pow(3)
        
        # Sumar sobre dimensiones espaciales
        sum_activations = activations.sum(dim=(1, 2), keepdim=True)
        
        # Alpha weights (ecuación Grad-CAM++)
        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8
        alpha = alpha_num / alpha_denom
        
        # ReLU positivo
        weights = (alpha * F.relu(gradients)).sum(dim=(1, 2), keepdim=True)
        
        # Weighted combination
        cam = (weights * activations).sum(dim=0)
        
        # ReLU + Normalización
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam
    
    def overlay_heatmap(self, original_image, cam, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """Overlay mejorado con mejor colormap"""
        
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
        
        # Convertir a RGB si necesario
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Aplicar colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlayed = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
        
        return Image.fromarray(overlayed)


def create_gradcam_for_densenet(model):
    """Crea Grad-CAM++ para DenseNet121"""
    target_layer = model.features.denseblock4
    return GradCAMPlusPlus(model, target_layer)