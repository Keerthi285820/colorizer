import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import io

# 1. Simple Dummy Model (demo — grayscale ➜ faint “color” tint)
class DummyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 3, 3, 1, 1)  # placeholder layer
    def forward(self, x):
        # Fake colorization: just replicate grayscale with slight boost
        return x.repeat(1, 3, 1, 1) * 0.7 + 0.15

# 2. Local Placeholder Image (fixed font handling!)
def create_placeholder():
    img = Image.new('RGB', (300, 200), color='#333')
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()  # safe fallback font
    draw.text((50, 80), "Upload B&W Image", fill='white', font=font)
    return img

# 3. Streamlit UI
st.set_page_config(layout="wide", page_title="AI Colorizer")
st.title("🎨 AI Colorization Demo")
st.markdown("<h6>Upload B&W image | Adjust size | See demo result</h6>", unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    resize_val = st.slider("Resize to (px)", 128, 512, 256, help="Image size before processing")
    st.write("Model: Dummy Generator (Demo only)")
    st.info("Real colorization requires trained weights. See code comments for integration.")

# Load model once globally
model = DummyGenerator().eval()

# Image upload
uploaded = st.file_uploader("Upload B&W Image", type=["jpg","png","jpeg","bmp"], label_visibility="collapsed")

if uploaded:
    try:
        # Load & preprocess
        img = Image.open(uploaded).convert("L")
        orig_size = img.size
        img_resized = img.resize((resize_val, resize_val), Image.LANCZOS)
        
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
            T.Lambda(lambda x: x.unsqueeze(0))
        ])
        input_tensor = transform(img_resized)
        
        # Inference
        with torch.no_grad():
            output = model(input_tensor)
        output = torch.clamp(output, 0, 1)  # keep safe range
        output_img = T.ToPILImage()(output.squeeze(0))
        
        # Post-process: resize back to original
        final_img = output_img.resize(orig_size, Image.LANCZOS)
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original B&W")
            st.image(img, use_container_width=True, clamp=True)
        with col2:
            st.subheader("Demo Colorized")
            st.image(final_img, use_container_width=True, clamp=True)
            
        # Download button
        buf = io.BytesIO()
        final_img.save(buf, 'PNG')
        st.download_button("Download Demo Image", buf.getvalue(), 
                           file_name=f"demo_{uploaded.name}", mime="image/png")
    
    except Exception as e:
        st.error("Processing failed")
        st.exception(e)
else:
    st.info("⬆️ Upload a B&W image to start")
    placeholder = create_placeholder()
    st.image(placeholder, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Built with Streamlit + PyTorch | Demo mode only")

# 4. Real Model Integration (Optional)
# To use a trained model, replace DummyGenerator with your own model class
# and load weights from a file, e.g.:
#
# def load_real_model():
#     model = RealGenerator()
#     model.load_state_dict(torch.load('weights.pth', map_location='cpu'))
#     return model.eval()
#
# model = load_real_model()
