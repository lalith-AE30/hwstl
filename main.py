import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

# Define the CVAE model (same as before)
class ConditionalVAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20, num_classes=10):
        super(ConditionalVAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Encoder
        self.fc1 = nn.Linear(input_dim + num_classes, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # mean
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # log variance
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim + num_classes, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x, c):
        inputs = torch.cat([x, c], dim=1)
        h1 = F.relu(self.fc1(inputs))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
        inputs = torch.cat([z, c], dim=1)
        h3 = F.relu(self.fc3(inputs))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, self.input_dim), c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

@st.cache_resource
def load_model():
    """Load the trained CVAE model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Try to load the saved model
        model = ConditionalVAE().to(device)
        model.load_state_dict(torch.load('cvae_weights.pth', map_location=device))
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error("Model weights file 'cvae_weights.pth' not found. Please train the model first.")
        return None, device

def generate_digit_samples(model, device, digit, num_samples=5):
    """Generate samples of a specific digit"""
    if model is None:
        return None
    
    model.eval()
    with torch.no_grad():
        # Create one-hot encoding for the specific digit
        class_vector = torch.zeros(num_samples, 10).to(device)
        class_vector[:, digit] = 1
        
        # Sample from latent space
        z = torch.randn(num_samples, model.latent_dim).to(device)
        
        # Generate images
        generated = model.decode(z, class_vector)
        generated = generated.view(num_samples, 28, 28)
        
        return generated.cpu().numpy()

def create_image_grid(images):
    """Create a grid of images for display"""
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    
    for i, img in enumerate(images):
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Convert matplotlib figure to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    
    return img

def main():
    st.set_page_config(
        page_title="CVAE Digit Generator",
        page_icon="🔢",
        layout="wide"
    )
    
    st.title("🔢 Conditional VAE Digit Generator")
    st.markdown("Generate MNIST-style digits using a trained Conditional Variational Autoencoder")
    
    # Load model
    model, device = load_model()
    
    if model is None:
        st.error("Please ensure you have trained the CVAE model and saved the weights as 'cvae_weights.pth'")
        st.markdown("""
        To train the model:
        1. Run the training script provided earlier
        2. Make sure the weights are saved as 'cvae_weights.pth'
        3. Place the weights file in the same directory as this Streamlit app
        """)
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Create input section
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input")
        digit = st.number_input(
            "Enter a digit (0-9):",
            min_value=0,
            max_value=9,
            value=7,
            step=1
        )
        
        generate_button = st.button("🎲 Generate Samples", type="primary")
        
        st.markdown("---")
        st.subheader("Settings")
        temperature = st.slider(
            "Generation Temperature:",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Higher values create more diverse samples"
        )
    
    with col2:
        st.subheader("Generated Samples")
        
        if generate_button:
            with st.spinner("Generating samples..."):
                # Generate samples with temperature scaling
                model.eval()
                with torch.no_grad():
                    # Create one-hot encoding for the specific digit
                    class_vector = torch.zeros(5, 10).to(device)
                    class_vector[:, digit] = 1
                    
                    # Sample from latent space with temperature
                    z = torch.randn(5, model.latent_dim).to(device) * temperature
                    
                    # Generate images
                    generated = model.decode(z, class_vector)
                    generated = generated.view(5, 28, 28)
                    
                    samples = generated.cpu().numpy()
                
                if samples is not None:
                    # Create and display image grid
                    image_grid = create_image_grid(samples)
                    st.image(image_grid, caption=f"5 Generated samples of digit {digit}")
                    
                    # Display individual samples
                    st.subheader("Individual Samples")
                    cols = st.columns(5)
                    
                    for i, sample in enumerate(samples):
                        with cols[i]:
                            fig, ax = plt.subplots(figsize=(2, 2))
                            ax.imshow(sample, cmap='gray')
                            ax.axis('off')
                            ax.set_title(f"Sample {i+1}")
                            
                            buf = io.BytesIO()
                            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                            buf.seek(0)
                            img = Image.open(buf)
                            plt.close()
                            
                            st.image(img, use_column_width=True)
                else:
                    st.error("Failed to generate samples. Please check the model.")
    
    # Sidebar with information
    st.sidebar.header("About")
    st.sidebar.markdown("""
    This app uses a **Conditional Variational Autoencoder (CVAE)** trained on the MNIST dataset.
    
    **How it works:**
    1. Enter a digit (0-9)
    2. The model generates 5 different samples of that digit
    3. Each sample is unique due to the random sampling in the latent space
    
    **Features:**
    - Generate any digit from 0-9
    - Adjustable generation temperature
    - Real-time generation
    """)
    
    st.sidebar.header("Model Info")
    if model is not None:
        st.sidebar.success("Model Status: Loaded ✅")
        st.sidebar.info(f"Device: {device}")
        st.sidebar.info(f"Latent Dimensions: {model.latent_dim}")
        st.sidebar.info(f"Hidden Dimensions: {model.hidden_dim}")
    else:
        st.sidebar.error("Model Status: Not Loaded ❌")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit and PyTorch")

if __name__ == "__main__":
    main()