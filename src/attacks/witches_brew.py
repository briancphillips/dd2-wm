import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm

class WitchesBrewPoisoner:
    """
    Implements a simplified version of the Witches' Brew gradient matching attack.
    The goal is to modify a set of 'poison' images so that their gradients match
    the gradients of a specific 'target' image when evaluated on a given model.
    """
    def __init__(self, model, epsilon=16/255, learning_rate=0.1, steps=250):
        self.model = model
        self.epsilon = epsilon
        self.lr = learning_rate
        self.steps = steps
        self.criterion = nn.CrossEntropyLoss()

    def _get_target_gradient(self, target_img, target_label, device):
        """
        Computes the gradient of the loss with respect to the model parameters
        for the target image we want the model to misclassify.
        """
        self.model.eval()
        self.model.zero_grad()
        
        target_img = target_img.unsqueeze(0).to(device)
        target_label = torch.tensor([target_label]).to(device)
        
        outputs = self.model(target_img)
        loss = self.criterion(outputs, target_label)
        
        # Calculate gradients
        target_grads = torch.autograd.grad(loss, self.model.parameters())
        
        # Flatten and concatenate all gradients into a single vector
        grad_vector = torch.cat([g.contiguous().view(-1) for g in target_grads])
        return grad_vector.detach()

    def generate_poisons(self, poison_images, poison_labels, target_img, target_label, device):
        """
        Optimizes the poison images to match the target gradient.
        """
        # Get the objective gradient we want to match
        target_grad_vector = self._get_target_gradient(target_img, target_label, device)
        
        # Setup the variables we will optimize
        poison_images = poison_images.clone().detach().to(device)
        poison_labels = poison_labels.to(device)
        
        # Save original images for projection (clipping)
        original_images = poison_images.clone().detach()
        
        # Enable gradients for the input images
        poison_images.requires_grad = True
        
        # We use Adam to optimize the images
        optimizer = optim.Adam([poison_images], lr=self.lr)
        
        self.model.eval()
        
        pbar = tqdm(range(self.steps), desc="Optimizing Poisons (Gradient Matching)")
        for step in pbar:
            optimizer.zero_grad()
            self.model.zero_grad()
            
            # Forward pass on poison images
            outputs = self.model(poison_images)
            loss = self.criterion(outputs, poison_labels)
            
            # Get gradients of the model parameters w.r.t the poison loss
            # create_graph=True allows us to take the derivative of these gradients
            poison_grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            poison_grad_vector = torch.cat([g.contiguous().view(-1) for g in poison_grads])
            
            # Our objective is to minimize the cosine distance (maximize cosine similarity)
            # between the poison gradients and the target gradient.
            # Using negative cosine similarity as the loss.
            cos_sim = nn.functional.cosine_similarity(poison_grad_vector, target_grad_vector, dim=0)
            matching_loss = -cos_sim
            
            # Backward pass to get gradients w.r.t the poison images
            matching_loss.backward()
            
            # Update poison images
            optimizer.step()
            
            # Project back to epsilon ball (L-infinity norm)
            with torch.no_grad():
                # Perturbation = optimized_image - original_image
                perturbation = poison_images - original_images
                # Clip perturbation to [-epsilon, epsilon]
                perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
                # Apply clipped perturbation and clamp to valid image range
                poison_images.copy_(original_images + perturbation)
                
            pbar.set_postfix({'cos_sim': f"{cos_sim.item():.4f}"})
            
        return poison_images.detach()

def create_poisoned_dataset(dataset, model, device, num_poisons=500, target_class=0, poison_class=1):
    """
    Utility function to select a target image and generate a set of poisons.
    """
    print(f"Selecting target image of class {target_class} and {num_poisons} base images of class {poison_class}...")
    
    target_img, target_label = None, None
    poison_indices = []
    poison_images_list = []
    poison_labels_list = []
    
    # Simple selection logic
    for i in range(len(dataset)):
        img, label = dataset[i]
        
        if label == target_class and target_img is None:
            target_img = img
            target_label = label
            
        elif label == poison_class and len(poison_indices) < num_poisons:
            poison_indices.append(i)
            poison_images_list.append(img)
            poison_labels_list.append(label)
            
        if target_img is not None and len(poison_indices) == num_poisons:
            break
            
    poison_images = torch.stack(poison_images_list)
    poison_labels = torch.tensor(poison_labels_list)
    
    poisoner = WitchesBrewPoisoner(model=model, epsilon=16/255, steps=250)
    
    # Generate the poisoned versions of the images
    optimized_poisons = poisoner.generate_poisons(
        poison_images=poison_images,
        poison_labels=poison_labels,
        target_img=target_img,
        target_label=target_label,
        device=device
    )
    
    return optimized_poisons, poison_indices, target_img, target_label
