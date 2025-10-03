import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import io

# ----------------------------
# 1. Dummy Model (Demo only)
# ----------------------------
class DummyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 3, 3, 1, 1)  # placeholder layer

    def forward(self, x):
        # Fake colorization: replicate grayscale with faint tint
        return x.repeat(1, 3, 1, 1) * 0.7 + 0.15


# ----------------------------
# 2. Placeholder Image
# ----------------------------
def create_placeholder():
    img = Image.new('RGB', (300, 200), color='#333')
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()  # safe fallback font
    draw.text((50, 80), "Upload B&W Image", fill='white', font=font)
    return img


# ----------------------------
# 3. Streamlit UI
# ----------------------------
st.set_page_config(layout="wide", page_title="AI Colorizer")
st.title("🎨 AI Colorization Demo")
st.markdown("<h6>Upload B&W image | Adjust size | See demo result</h6>", unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    resize_val = st.slider("Resize to (px)", 128, 512, 256, help="Image size before processing")
    model_choice = st.radio("Choose model", ["Dummy", "Real (if available)"])
    st.info("Demo mode uses a dummy generator. Real mode requires your trained weights.")

# ----------------------------
# 4. Load Model
# ----------------------------
if model_choice == "Dummy":
    model = DummyGenerator().eval()
else:
    # Define your real model here
    class RealGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            # Example: replace with your GAN/Autoencoder layers
            self.conv = nn.Conv2d(1, 3, 3, 1, 1)
        def forward(self, x):
            return self.conv(x)

    def load_real_model():
        model = RealGenerator()
        # Replace 'weights.pth' with your weights file
        model.load_state_dict(torch.load("weights.pth", map_location="cpu"))
        return model.eval()

    try:
        model = load_real_model()
    except Exception:
        st.warning("⚠️ Real model weights not found. Falling back to Dummy.")
        model = DummyGenerator().eval()

# ----------------------------
# 5. File Upload
# ----------------------------
uploaded = st.file_uploader("Upload B&W Image", type=["jpg", "png", "jpeg", "bmp"], label_visibility="collapsed")

if uploaded:
    try:
        # Load & preprocess
        img = Image.open(uploaded).convert("L")
        orig_size = img.size
        img_resized = img.resize((resize_val, resize_val), Image.LANCZOS)

        transform = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: x.unsqueeze(0))  # add batch dim
        ])
        input_tensor = transform(img_resized)

        # Inference
        with torch.inference_mode():
            output = model(input_tensor)

        # Clamp output to [0,1]
        output = torch.clamp(output, 0, 1)
        output_img = T.ToPILImage()(output.squeeze(0))

        # Resize back to original
        final_img = output_img.resize(orig_size, Image.LANCZOS)

        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original B&W")
            st.image(img, use_container_width=True)
            st.caption(f"Original size: {orig_size}")

        with col2:
            st.subheader("Demo Colorized")
            st.image(final_img, use_container_width=True)
            st.caption(f"Resized to: {resize_val}x{resize_val} for processing")

        # Download button
        buf = io.BytesIO()
        final_img.save(buf, 'PNG')
        buf.seek(0)
        st.download_button("Download Demo Image", buf, file_name=f"colorized_{uploaded.name}", mime="image/png")

    except Exception as e:
        st.error("Processing failed")
        st.exception(e)
else:
    st.info("⬆️ Upload a B&W image to start")
    st.image(create_placeholder(), use_container_width=True)

# ----------------------------
# 6. Footer
# ----------------------------
st.markdown("---")
st.caption("Built with Streamlit + PyTorch | Demo mode (Dummy) or Real mode (if weights provided)")
