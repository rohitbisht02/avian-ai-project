import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import tensorflow as tf
import pickle
import os
import hashlib
import altair as alt
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import noisereduce as nr
import io
import pathlib
import soundfile as sf
import matplotlib.pyplot as plt

# --- App Configuration ---
st.set_page_config(page_title="Species Identification AI", layout="wide")

# --- Load Resources ---
SCRIPT_DIR = pathlib.Path(__file__).parent
# MODEL_PATH = SCRIPT_DIR.parent / "models" / "bird_model.h5"
# MLB_PATH = SCRIPT_DIR.parent / "models" / "mlb.pkl"
# TAXONOMY_PATH = SCRIPT_DIR.parent / "data" / "birds" / "eBird_Taxonomy_v2021.csv"
MODEL_PATH = SCRIPT_DIR.parent/ "models" / "final_model_50_species.keras"
MLB_PATH = SCRIPT_DIR.parent / "models" / "final_model_50_species_mlb.pkl"
TAXONOMY_PATH = SCRIPT_DIR.parent/ "data" / "birds" / "eBird_Taxonomy_v2021.csv"

UPLOADS_DIR = SCRIPT_DIR.parent / "user_uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(MLB_PATH, 'rb') as f:
        mlb = pickle.load(f)
    taxonomy_df = pd.read_csv(TAXONOMY_PATH)
    name_map = dict(zip(taxonomy_df['SPECIES_CODE'], taxonomy_df['PRIMARY_COM_NAME']))
    return model, mlb, name_map

try:
    model, mlb, name_map = load_resources()
    CLASSES = mlb.classes_
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# --- Helper Functions ---
def get_file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()

def preprocess_audio(audio_bytes):
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=32000)
        target_length = 5 * sr
        if len(y) > target_length: y = y[:target_length]
        else: y = np.pad(y, (0, target_length - len(y)), 'constant')
        y_reduced = nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=0.85)
        S = librosa.feature.melspectrogram(y=y_reduced, sr=sr, n_mels=128, fmin=20, fmax=16000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        return S_dB, sr, y # Return audio waveform 'y' as well
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None, None, None

def convert_spectrogram_to_image(S_dB, sr):
    fig = plt.figure(figsize=[1, 1]); ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False); ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    librosa.display.specshow(S_dB, sr=sr, fmin=20, fmax=16000, ax=ax)
    buf = io.BytesIO(); plt.savefig(buf, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig); buf.seek(0)
    image = tf.io.decode_png(buf.getvalue(), channels=3)
    image = tf.image.resize(image, [128, 128]); image = image / 255.0
    return image

# --- UI Layout ---
st.sidebar.title("🐦 Species Identification AI")
st.sidebar.write("Upload an audio file or record live audio to identify bird species.")
option = st.sidebar.radio("Choose your input method:", ["Upload Audio File", "Record Live Audio"])

if option == "Upload Audio File":
    st.header("Upload Audio File")
    uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'ogg', 'mp3'])
    
    if uploaded_file is not None:
        audio_bytes = uploaded_file.getvalue()
        file_hash = get_file_hash(audio_bytes)
        
        st.audio(audio_bytes)
        
        if st.button("Identify from Uploaded File"):
            with st.spinner('Analyzing audio...'):
                S_dB, sr, audio_waveform = preprocess_audio(audio_bytes)
                if S_dB is not None:
                    image_tensor = convert_spectrogram_to_image(S_dB, sr)
                    
                    # --- Display Spectrogram and Waveform ---
                    st.subheader("Visualizations")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Waveform")
                        fig, ax = plt.subplots()
                        librosa.display.waveshow(audio_waveform, sr=sr, ax=ax)
                        st.pyplot(fig)
                    with col2:
                        st.write("Mel Spectrogram")
                        fig, ax = plt.subplots()
                        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
                        st.pyplot(fig)

                    # --- Prediction and Results ---
                    input_tensor = np.expand_dims(image_tensor, axis=0)
                    predictions = model.predict(input_tensor)[0]
                    
                    results = [(CLASSES[i], pred) for i, pred in enumerate(predictions)]
                    results.sort(key=lambda x: x[1], reverse=True)
                    
                    st.subheader("Identification Results")
                    
                    # Display top 5 predictions in a chart
                    results_df = pd.DataFrame(results[:5], columns=['Species Code', 'Confidence'])
                    results_df['Species Name'] = results_df['Species Code'].apply(lambda x: name_map.get(x, x))
                    
                    chart = alt.Chart(results_df).mark_bar().encode(
                        x=alt.X('Confidence', scale=alt.Scale(domain=[0, 1])),
                        y=alt.Y('Species Name', sort='-x'),
                        tooltip=['Species Name', 'Confidence']
                    ).properties(title="Top 5 Predictions")
                    st.altair_chart(chart, use_container_width=True)
                    
                    # Save the uploaded file
                    top_species_code = results[0][0]
                    top_species_name = name_map.get(top_species_code, top_species_code).replace(" ", "_")
                    save_path = UPLOADS_DIR / f"{top_species_name}_{file_hash[:8]}.ogg"
                    with open(save_path, "wb") as f:
                        f.write(audio_bytes)
                    st.success(f"Analysis complete. Audio saved as {save_path.name}")

if option == "Record Live Audio":
    st.header("Record Live Audio")
    st.write("Click 'Start' to record a 5-second audio clip.")
    # (Live recording code remains the same as before)
    class AudioRecorder(AudioProcessorBase):
        def __init__(self): self.audio_buffer = []
        def recv(self, frame): self.audio_buffer.append(frame.to_ndarray()); return frame
        
    webrtc_ctx = webrtc_streamer(key="audio-recorder", mode=WebRtcMode.SENDONLY,
                               audio_processor_factory=AudioRecorder,
                               media_stream_constraints={"video": False, "audio": True})
    
   if st.button("Identify from Uploaded File"):
    with st.spinner('Processing and identifying...'):
        print("--- DEBUG: Button clicked. Starting process. ---")

        spectrogram = preprocess_audio(uploaded_file.getvalue())

        if spectrogram is not None:
            print("--- DEBUG: Preprocessing successful. Preparing for prediction. ---")
            input_tensor = np.expand_dims(spectrogram, axis=0)

            print("--- DEBUG: Calling model.predict()... ---")
            predictions = model.predict(input_tensor)[0]
            print("--- DEBUG: model.predict() finished. ---")

            # Display results (this part remains the same)
            st.subheader("Identification Results")
            # ... (rest of the result display code) ...