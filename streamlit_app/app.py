# Final App Code: app.py

import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import pickle
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import noisereduce as nr
import io
import pathlib
import soundfile as sf
import requests
from bs4 import BeautifulSoup
import altair as alt
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import sqlite3
from datetime import datetime

# --- App Configuration ---
st.set_page_config(page_title="Avian AI", layout="wide", page_icon="🐦")

# --- [CRITICAL] File Paths ---
APP_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = APP_DIR.parent
WEIGHTS_PATH = PROJECT_ROOT / "models" / "final_model_50_species.weights.h5" # <-- Use the weights file
MLB_PATH = PROJECT_ROOT / "models" / "final_model_50_species_mlb.pkl"
TAXONOMY_PATH = PROJECT_ROOT / "data" / "birds" / "eBird_Taxonomy_v2021.csv"
CONFIG_PATH = APP_DIR / 'config.yaml'

# --- [CRITICAL] Resource Loading Function ---
@st.cache_resource
def load_resources():
    """Builds the model architecture from scratch and then loads the saved weights."""
    try:
        # 1. Define the Model Architecture EXACTLY as in the successful training log
        base_model = tf.keras.applications.EfficientNetV2B0(
            include_top=False,
            weights='imagenet', # Start with imagenet weights for the base
            input_shape=(128, 128, 3)
        )
        base_model.trainable = False # Not needed for inference

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(128, 128, 3)),
            tf.keras.layers.Rescaling(1./255), # This layer was in your trained model
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(50, activation='softmax')
        ])
        
        # 2. Load the Saved Weights into the architecture
        model.load_weights(str(WEIGHTS_PATH))
        
        # 3. Load Other Resources
        with open(MLB_PATH, 'rb') as f:
            mlb = pickle.load(f)
        taxonomy_df = pd.read_csv(TAXONOMY_PATH)
        name_map = dict(zip(taxonomy_df['SPECIES_CODE'], taxonomy_df['PRIMARY_COM_NAME']))
        
        return model, mlb, name_map
        
    except Exception as e:
        st.error(f"Fatal Error: Could not build model or load resources. Check file paths. Error: {e}")
        st.stop()

# --- User Authentication ---
try:
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=SafeLoader)
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )
    name, authentication_status, username = authenticator.login(fields={'Form name':'Login', 'Username':'Username', 'Password':'Password'})
except Exception as e:
    st.error(f"Could not load authentication configuration. Check 'config.yaml'. Error: {e}")
    authentication_status = None

# --- Main App Logic ---
def init_db():
    conn = sqlite3.connect('history.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS history
        (username TEXT, timestamp TEXT, bird_name TEXT, confidence REAL, image_url TEXT, scientific_name TEXT)
    ''')
    conn.commit()
    return conn

if authentication_status:
    model, mlb, name_map = load_resources()
    CLASSES = mlb.classes_
    db_conn = init_db() # Connect to DB after login

    authenticator.logout('Logout', 'sidebar')
    st.sidebar.title(f"Welcome *{name}*")

    # (Your other functions like get_history, get_bird_info, preprocess_audio, display_results go here without changes)
    # ...
    # (Your main app layout with tabs and buttons goes here without changes)
    # ...

    # The rest of your app code remains the same as what you provided.
    # Just ensure the load_resources function and the file paths at the top are replaced.
    def add_history(conn, username, bird_name, confidence, image_url, scientific_name):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c = conn.cursor()
        c.execute("INSERT INTO history (username, timestamp, bird_name, confidence, image_url, scientific_name) VALUES (?, ?, ?, ?, ?, ?)",
                  (username, timestamp, bird_name, confidence, image_url, scientific_name))
        conn.commit()

    def get_history(conn, username):
        c = conn.cursor()
        c.execute("SELECT timestamp, bird_name, confidence, image_url, scientific_name FROM history WHERE username = ? ORDER BY timestamp DESC", (username,))
        return c.fetchall()

    @st.cache_data(ttl=3600)
    def get_bird_info(common_name):
        try:
            search_url = f"https://en.wikipedia.org/wiki/{common_name.replace(' ', '_')}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            infobox = soup.find('table', {'class': 'infobox biota'})
            if not infobox: return "Not found", "No Wikipedia info box found.", None
            scientific_name_tag = infobox.find('span', {'class': 'binomial'})
            scientific_name = scientific_name_tag.text.strip() if scientific_name_tag else "Not found"
            first_p = infobox.find_next_sibling('p')
            description = first_p.text.strip() if first_p else "No description available."
            image_tag = infobox.find('img')
            image_url = "https:" + image_tag['src'] if image_tag and image_tag.has_attr('src') else None
            return scientific_name, description, image_url
        except Exception:
            return "Not found", f"Could not fetch details for {common_name}.", None

    def preprocess_audio(audio_bytes):
        try:
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=32000)
            target_length = 5 * sr
            if len(y) > target_length: y = y[:target_length]
            else: y = np.pad(y, (0, target_length - len(y)), 'constant')
            y_reduced = nr.reduce_noise(y=y, sr=sr)
            S = librosa.feature.melspectrogram(y=y_reduced, sr=sr, n_mels=128, fmin=20, fmax=16000)
            S_dB = librosa.power_to_db(S, ref=np.max)
            spec_arr = np.expand_dims(S_dB, axis=-1)
            spec_arr = np.repeat(spec_arr, 3, axis=-1)
            return spec_arr
        except Exception as e:
            st.error(f"Error processing audio: {e}")
            return None

    def display_results(predictions):
        st.subheader("AI Identification Results")
        results = sorted([(CLASSES[i], pred) for i, pred in enumerate(predictions)], key=lambda x: x[1], reverse=True)
        top_species_code, top_confidence = results[0]
        top_full_name = name_map.get(top_species_code, top_species_code)
        st.info(f"**Top Guess:** {top_full_name} ({top_confidence*100:.2f}% Confidence)")
        with st.spinner(f'Fetching details for {top_full_name}...'):
            scientific_name, description, image_url = get_bird_info(top_full_name)
        add_history(db_conn, username, top_full_name, top_confidence, image_url, scientific_name)
        if image_url: st.image(image_url, caption=f"{top_full_name} ({scientific_name})", use_column_width=True)
        if description:
            st.markdown(f"**Scientific Name:** *{scientific_name}*")
            st.markdown(f"**About:** {description}")
        st.markdown("---")
        st.write("Confidence Distribution (Top 5)")
        chart_data = pd.DataFrame({'Species': [name_map.get(c, c) for c, _ in results[:5]], 'Confidence': [p * 100 for _, p in results[:5]]})
        chart = alt.Chart(chart_data).mark_bar().encode(x=alt.X('Confidence:Q', title='Confidence (%)'), y=alt.Y('Species:N', title='Species', sort='-x'))
        st.altair_chart(chart, use_container_width=True)

    st.title("🐦 Avian AI")
    with st.sidebar.expander("🔍 Your Identification History", expanded=False):
        history = get_history(db_conn, username)
        if not history:
            st.write("No history yet.")
        else:
            for item in history:
                st.markdown(f"**{item[1]}** ({item[4]})")
                st.caption(f"_{item[0]}_ - {item[2]*100:.1f}% confidence")
                if item[3]: st.image(item[3], width=100)
                st.markdown("---")
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.header("Upload Audio File")
        uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'ogg', 'mp3'])
        if uploaded_file:
            st.audio(uploaded_file)
            if st.button("Identify from Uploaded File"):
                with st.spinner('AI is listening... 🎶'):
                    spectrogram = preprocess_audio(uploaded_file.getvalue())
                    if spectrogram is not None:
                        predictions = model.predict(np.expand_dims(spectrogram, axis=0))[0]
                        display_results(predictions)
    with col2:
        st.header("Record Live Audio")
        st.write("Click 'Start' to record a 5-second clip.")
        webrtc_ctx = webrtc_streamer(key="audio-recorder", mode=WebRtcMode.SENDONLY, audio_processor_factory=AudioProcessorBase, media_stream_constraints={"video": False, "audio": True})
        if st.button("Identify from Live Recording"):
            if webrtc_ctx.audio_processor and webrtc_ctx.audio_processor.audio_buffer:
                audio_frames = np.concatenate(webrtc_ctx.audio_processor.audio_buffer, axis=1)
                audio_int = (librosa.to_mono(audio_frames) * 32767).astype(np.int16)
                buffer = io.BytesIO()
                sf.write(buffer, audio_int.T, 48000, format='WAV'); buffer.seek(0)
                with st.spinner('AI is listening... 🎶'):
                    spectrogram = preprocess_audio(buffer.getvalue())
                    if spectrogram is not None:
                        predictions = model.predict(np.expand_dims(spectrogram, axis=0))[0]
                        display_results(predictions)
            else:
                st.warning("No audio recorded. Please start recording first.")

elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')