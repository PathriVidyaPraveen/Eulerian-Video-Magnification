import streamlit as st
import numpy as np
import cv2
import tempfile
import os

# =========================
# COLOR SPACE
# =========================
yiq_from_rgb = np.array(
    [[0.299, 0.587, 0.114],
     [0.5959, -0.2746, -0.3213],
     [0.2115, -0.5227, 0.3112]], dtype=np.float32
)
rgb_from_yiq = np.linalg.inv(yiq_from_rgb)

def rgb2yiq(img):
    return img.astype(np.float32) @ yiq_from_rgb.T

def yiq2rgb(img):
    return np.clip(img @ rgb_from_yiq.T, 0, 255).astype(np.uint8)

# =========================
# VIDEO IO
# =========================
def load_video(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame[:, :, ::-1])  # BGR → RGB

    cap.release()
    return np.array(frames), fps

def save_video(frames, path, fps):
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        out.write(f[:, :, ::-1])  # RGB → BGR
    out.release()



# =========================
# GAUSSIAN PYRAMID (CORRECT)
# =========================
def gaussian_pyramid(image, levels):
    pyramid = [image]
    for _ in range(levels):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid

def reconstruct_from_gaussian(lowest, pyramid):
    current = lowest
    for level in reversed(pyramid[:-1]):
        current = cv2.pyrUp(current)
        if current.shape != level.shape:
            current = cv2.resize(current, (level.shape[1], level.shape[0]))
    return current

# =========================
# TEMPORAL FILTER (IDEAL)
# =========================
def ideal_bandpass(signal, fps, f_lo, f_hi):
    fft = np.fft.fft(signal, axis=0)
    freqs = np.fft.fftfreq(signal.shape[0], d=1.0 / fps)

    mask = (freqs >= f_lo) & (freqs <= f_hi)
    fft[~mask] = 0

    return np.fft.ifft(fft, axis=0).real

# =========================
# EULERIAN VIDEO MAGNIFICATION
# =========================
def eulerian_magnification(frames, fps, levels, alpha, f_lo, f_hi, chroma_atten):
    yiq_frames = np.array([rgb2yiq(f) for f in frames])

    pyramids = []
    for f in yiq_frames:
        gp = gaussian_pyramid(f, levels)
        pyramids.append(gp[-1])  # lowest level only

    pyramids = np.array(pyramids)

    filtered = ideal_bandpass(pyramids, fps, f_lo, f_hi)
    filtered *= alpha
    filtered[:, :, :, 1:] *= chroma_atten

    output = []
    for i in range(len(frames)):
        gp = gaussian_pyramid(yiq_frames[i], levels)
        recon = reconstruct_from_gaussian(filtered[i], gp)
        out = yiq_frames[i] + recon
        output.append(yiq2rgb(out))

    return np.array(output)

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(layout="wide")
st.title("Eulerian Video Magnification (MIT CSAIL – Correct Gaussian EVM)")

st.markdown("""
**Exact Gaussian Eulerian Video Magnification pipeline**  
Spatial Gaussian pyramid → Temporal bandpass → Amplification → Reconstruction

No ML. No classification. Visualization only.
""")

uploaded = st.file_uploader("Upload a video (.mp4)", type=["mp4"])

col1, col2 = st.columns(2)

with col1:
    alpha = st.slider("Amplification α", 10, 200, 100)
    levels = st.slider("Gaussian Pyramid Levels", 2, 6, 4)

with col2:
    f_lo = st.slider("Low Frequency (Hz)", 0.4, 2.0, 0.83)
    f_hi = st.slider("High Frequency (Hz)", 0.9, 3.0, 1.0)
    chroma = st.slider("Chroma Attenuation", 0.1, 1.0, 0.6)

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    st.info("Processing video (Eulerian magnification is computationally heavy)")
    frames, fps = load_video(video_path)

    out = eulerian_magnification(
        frames,
        fps,
        levels=levels,
        alpha=alpha,
        f_lo=f_lo,
        f_hi=f_hi,
        chroma_atten=chroma
    )

    out_path = video_path.replace(".mp4", "_magnified.mp4")

    save_video(out, out_path, fps)

    st.success("Magnification complete")
    

    with open(out_path, "rb") as f:
        video_bytes = f.read()

    st.download_button(
        label="Download Magnified Video (MP4)",
        data=video_bytes,
        file_name="eulerian_magnified.mp4",
        mime="video/mp4"
    )


