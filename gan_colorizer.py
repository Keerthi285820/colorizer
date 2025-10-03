import streamlit as st
import torch, torch.nn as nn, torchvision.transforms as T
from PIL import Image, ImageDraw
import io

# 1. Simple Dummy Model (No weights, no base64)
class DummyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 3, 3, 1, 1)  # Grayscale -> RGB
    def forward(self, x):
        # Fake colorization: repeat grayscale with slight color shift
        return x.repeat(1, 3, 1, 1) * 0.7 + 0.15  # Faint color tint

# 2. Local Placeholder Image (No internet)
def create_placeholder():
    img = Image.new('RGB', (300, 200), color='#333')
    draw = ImageDraw.Draw(img)
    draw.text((50, 80), "Upload B&W Image", fill='white', font=Image.font(size=20))
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

# Image upload
uploaded = st.file_uploader("Upload B&W Image", type=["jpg","png","jpeg","bmp"], label_visibility="collapsed")

if uploaded:
    try:
        # Load & preprocess
        img = Image.open(uploaded).convert("L")
        orig_size = img.size
        img_resized = img.resize((resize_val, resize_val), Image.LANCZOS)
        
        # Transform for model
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
            T.Lambda(lambda x: x.unsqueeze(0))
        ])
        input_tensor = transform(img_resized)
        
        # Run inference (dummy - no base64)
        model = DummyGenerator()
        with torch.no_grad():
            output = model(input_tensor)
        output_img = T.ToPILImage()(output.squeeze(0) * 0.5 + 0.5)
        
        # Post-process: resize to original
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
        st.error(f"Processing failed: {e}")
        st.exception(e)
else:
    st.info("⬆️ Upload a B&W image to start")
    placeholder = create_placeholder()
    st.image(placeholder, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Built with Streamlit + PyTorch | Demo mode only")

# 4. Real Model Integration (Optional)
# To use a real model, replace DummyGenerator with your trained model
# and load weights from a local file. Example:
#
# class RealGenerator(nn.Module):
#     def __init__(self): ...  # Your GAN architecture
#     def forward(self, x): ... # Real inference logic
#
# def load_real_model():
#     model = RealGenerator()
#     model.load_state_dict(torch.load("weights.pth", map_location="cpu"))
#     return model.eval()
#
# Replace model = DummyGenerator() with model = load_real_model()
