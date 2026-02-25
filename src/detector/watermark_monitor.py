import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

class WatermarkMonitor:
    """
    DynaDetect-WM Tracing Module.
    Monitors the activation response of a model across intermediate layers
    when exposed to the repurposed 'poisoned' watermark probes.
    """
    def __init__(self, model, layer_names=None):
        self.model = model
        # Our ResNet18 wrapper stores the actual torchvision model under 'model' attribute
        # So the internal layers are named 'model.layer1', 'model.layer2', etc.
        self.layer_names = layer_names if layer_names else ['model.layer1', 'model.layer2', 'model.layer3', 'model.layer4']
        self.activations = {}
        self.hooks = []
        self._register_hooks()
        
        # Will store the expected activation signatures (from the authorized model)
        self.reference_signatures = {}

    def _register_hooks(self):
        """Registers forward hooks to capture intermediate layer outputs."""
        def get_activation(name):
            def hook(model, input, output):
                # Flatten the spatial dimensions: (B, C, H, W) -> (B, C)
                # Using global average pooling for a stable signature
                pooled = torch.mean(output, dim=[2, 3])
                self.activations[name] = pooled.detach().cpu().numpy()
            return hook

        for name, module in self.model.named_modules():
            if name in self.layer_names:
                hook = module.register_forward_hook(get_activation(name))
                self.hooks.append(hook)

    def remove_hooks(self):
        """Cleans up hooks when tracing is done."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_layer_activations(self, inputs, device):
        """Passes inputs through the model and returns the captured activations."""
        self.model.eval()
        self.activations = {}
        with torch.no_grad():
            _ = self.model(inputs.to(device))
        return self.activations

    def generate_reference_signatures(self, watermark_loader, device):
        """
        Calculates the expected average activation pattern for the watermarks 
        on the authorized (or clean) model.
        """
        print("Generating reference signatures for watermarks...")
        all_activations = {name: [] for name in self.layer_names}
        
        for inputs, _ in tqdm(watermark_loader, desc="Tracing Reference"):
            acts = self.get_layer_activations(inputs, device)
            for name in self.layer_names:
                all_activations[name].append(acts[name])
                
        # Average across the batch dimension to get a single signature vector per layer
        for name in self.layer_names:
            stacked = np.vstack(all_activations[name])
            self.reference_signatures[name] = np.mean(stacked, axis=0)

    def audit_model(self, target_model, watermark_loader, device):
        """
        Audits a potentially unauthorized 'stolen' model by comparing its
        activation responses to the watermarks against the reference signatures.
        Note: The target_model must have the same architecture/layer names for direct cosine comparison.
        """
        print("Auditing target model...")
        
        # Temporarily swap the model we are hooking into
        original_model = self.model
        self.remove_hooks()
        self.model = target_model
        self._register_hooks()
        
        target_activations = {name: [] for name in self.layer_names}
        
        for inputs, _ in tqdm(watermark_loader, desc="Auditing"):
            acts = self.get_layer_activations(inputs, device)
            for name in self.layer_names:
                target_activations[name].append(acts[name])
                
        # Clean up target hooks and restore original
        self.remove_hooks()
        self.model = original_model
        self._register_hooks()

        # Calculate alignment metrics (Cosine Similarity between target and reference)
        alignment_scores = {}
        for name in self.layer_names:
            stacked_target = np.vstack(target_activations[name])
            mean_target_signature = np.mean(stacked_target, axis=0)
            
            # Reshape for sklearn cosine_similarity (requires 2D arrays)
            ref = self.reference_signatures[name].reshape(1, -1)
            targ = mean_target_signature.reshape(1, -1)
            
            sim = cosine_similarity(ref, targ)[0][0]
            alignment_scores[name] = sim
            
        return alignment_scores
