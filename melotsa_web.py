import os
import json
import random
import threading
import logging
import numpy as np
np.complex = complex  # Fix pour librosa
import librosa
import sounddevice as sd
import time
from flask import Flask, request, jsonify, send_from_directory, render_template
from scipy.signal import butter, lfilter
from openai import OpenAI
from werkzeug.utils import secure_filename

# === Configuration ===
OPENAI_API_KEY = "sk-"  # À remplacer
AUDIO_DIR = "audio_wav"
UPLOAD_FOLDER = AUDIO_DIR
ALLOWED_EXTENSIONS = {"mp3", "wav", "flac"}
SAMPLE_RATE = 44100
SWITCH_INTERVAL = 60  # Durée entre les changements de filtre (en secondes)
USE_ARDUINO = False

# Paramètres des filtres
FILTER_SETTINGS = {
    'lowpass_range': (400, 800),
    'highpass_range': (150, 300),
    'transition_duration': 0.1,  # secondes
    'random_switch_prob': 0.3,   # Probabilité de changement aléatoire
    'effect_duration': 3         # Durée des effets en secondes
}

# === Initialisation ===
os.makedirs(AUDIO_DIR, exist_ok=True)
app = Flask(__name__, template_folder="templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
client = OpenAI(api_key=OPENAI_API_KEY)
current_track = {"filename": None, "filter_params": {}}
vibration_enabled = False
tempo_cache = {}

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tomatis_simulation.log'),
        logging.StreamHandler()
    ]
)

# === Arduino ===
try:
    import serial
    arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
    USE_ARDUINO = True
    logging.info("Arduino connecté")
except Exception as e:
    arduino = None
    USE_ARDUINO = False
    logging.warning(f"Arduino non détecté: {str(e)}")

# === Utilitaires Audio ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def butter_lowpass(cutoff, sr, order=4):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype='low', analog=False)

def butter_highpass(cutoff, sr, order=4):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype='high', analog=False)

def apply_filter(data, b, a):
    return lfilter(b, a, data)

def enhanced_filter(y, sr):
    """Filtre avancé avec transitions progressives et variations aléatoires"""
    transition_samples = int(FILTER_SETTINGS['transition_duration'] * sr)
    segment_samples = int(SWITCH_INTERVAL * sr)
    filtered = np.zeros_like(y)
    
    current_filter = 'lowpass' if random.random() > 0.5 else 'highpass'
    
    for i in range(0, len(y), segment_samples):
        # Changement aléatoire de filtre
        if random.random() < FILTER_SETTINGS['random_switch_prob']:
            current_filter = 'highpass' if current_filter == 'lowpass' else 'lowpass'
        
        # Sélection de la fréquence de coupure
        if current_filter == 'lowpass':
            cutoff = random.uniform(*FILTER_SETTINGS['lowpass_range'])
            b, a = butter_lowpass(cutoff, sr)
        else:
            cutoff = random.uniform(*FILTER_SETTINGS['highpass_range'])
            b, a = butter_highpass(cutoff, sr)
        
        # Journalisation des paramètres
        current_track['filter_params'] = {
            'type': current_filter,
            'cutoff': round(cutoff, 1),
            'position': round(i/sr, 1)
        }
        logging.info(f"Applying {current_filter} at {cutoff:.1f}Hz")
        
        # Application du filtre
        segment = y[i:i + segment_samples]
        if len(segment) == 0:
            continue
            
        segment_filtered = apply_filter(segment, b, a)
        
        # Transition progressive
        if i > 0 and transition_samples > 0:
            transition_window = np.linspace(0, 1, transition_samples)
            filtered[i:i+transition_samples] = (
                transition_window * segment_filtered[:transition_samples] + 
                (1-transition_window) * filtered[i:i+transition_samples]
            )
            filtered[i+transition_samples:i+len(segment_filtered)] = segment_filtered[transition_samples:]
        else:
            filtered[i:i+len(segment_filtered)] = segment_filtered
    
    return filtered

# === Gestion Playlist ===
def extract_metadata(filepath):
    """Extrait les métadonnées audio avec mise en cache"""
    try:
        if os.path.basename(filepath) in tempo_cache:
            return tempo_cache[os.path.basename(filepath)]
            
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE)
        tempo_raw, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo_raw) if np.isscalar(tempo_raw) else float(tempo_raw[0])
        duration = librosa.get_duration(y=y, sr=sr)
        
        metadata = {
            "filename": os.path.basename(filepath),
            "tempo": round(tempo, 2),
            "duration": round(duration, 1),
            "path": filepath
        }
        
        tempo_cache[metadata["filename"]] = metadata
        return metadata
    except Exception as e:
        logging.error(f"Erreur extraction métadonnées {filepath}: {str(e)}")
        return None

def filter_with_gpt(metadata_list):
    """Filtre les morceaux avec GPT-4"""
    prompt = """Tu es un expert en musicothérapie pour enfants TSA.
Sélectionne les morceaux les plus adaptés selon ces critères:
- Tempo modéré (70-120 BPM)
- Durée > 45 secondes
- Caractère apaisant
- Pas de changements brusques

Réponds uniquement avec un tableau JSON des noms de fichiers valides.
Exemple: ["musique1.mp3", "musique2.wav"]

Morceaux disponibles:"""
    prompt += json.dumps([m for m in metadata_list if m is not None], indent=2)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return list(result.values())[0]  # Adaptation au format de réponse
    except Exception as e:
        logging.error(f"Erreur GPT: {str(e)}")
        return []

def generate_playlist():
    """Génère une playlist filtrée"""
    files = [f for f in os.listdir(AUDIO_DIR) if allowed_file(f)]
    if not files:
        logging.warning("Aucun fichier audio trouvé")
        return []
    
    metadata_list = [extract_metadata(os.path.join(AUDIO_DIR, f)) for f in files]
    valid_tracks = [m for m in metadata_list if m is not None]
    
    if not valid_tracks:
        logging.error("Aucune métadonnée valide")
        return []
    
    return filter_with_gpt(valid_tracks) or [t["filename"] for t in valid_tracks]

# === Effets Physiques ===
def send_to_arduino(command):
    """Envoie une commande à l'Arduino"""
    if USE_ARDUINO and arduino:
        try:
            arduino.write(f"{command}\n".encode())
            time.sleep(0.1)
            return True
        except Exception as e:
            logging.error(f"Erreur Arduino: {str(e)}")
            return False
    logging.info(f"[SIMULATION] Arduino: {command}")
    return True

def update_effects(filename):
    """Met à jour les effets en fonction du morceau"""
    if filename not in tempo_cache:
        extract_metadata(os.path.join(AUDIO_DIR, filename))
    
    tempo = tempo_cache.get(filename, {}).get("tempo", 80)
    delay = max(0.3, 60.0 / tempo)
    duration = FILTER_SETTINGS['effect_duration']
    end_time = time.time() + duration
    
    logging.info(f"Effets pour {filename} (Tempo: {tempo} BPM)")
    
    while time.time() < end_time:
        if vibration_enabled:
            send_to_arduino("vibrate:200")
        send_to_arduino(f"led_color:{random.randint(100, 255)},{random.randint(100, 255)},{random.randint(100, 255)}")
        time.sleep(delay)

# === Lecture Audio ===
def play_with_filter(filepath):
    """Joue un fichier audio avec filtrage Tomatis"""
    try:
        filename = os.path.basename(filepath)
        current_track["filename"] = filename
        logging.info(f"Début lecture: {filename}")
        
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
        y_filtered = enhanced_filter(y, sr)
        
        # Effets dans un thread séparé
        effects_thread = threading.Thread(target=update_effects, args=(filename,))
        effects_thread.daemon = True
        effects_thread.start()
        
        sd.stop()
        sd.play(y_filtered.astype(np.float32), samplerate=sr)
        sd.wait()
        
        logging.info(f"Fin lecture: {filename}")
    except Exception as e:
        logging.error(f"Erreur lecture {filepath}: {str(e)}")

def playback_loop():
    """Boucle principale de lecture"""
    while True:
        try:
            playlist = generate_playlist()
            if not playlist:
                logging.warning("Playlist vide - attente 10s")
                time.sleep(10)
                continue
                
            random.shuffle(playlist)
            logging.info(f"Nouvelle playlist: {playlist}")
            
            for track in playlist:
                play_with_filter(os.path.join(AUDIO_DIR, track))
                
        except Exception as e:
            logging.error(f"Erreur playback_loop: {str(e)}")
            time.sleep(5)

# === Routes Flask ===
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/audio/<path:filename>")
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename)

@app.route("/now")
def now_playing():
    return jsonify({
        **current_track,
        "vibration": vibration_enabled,
        "arduino": USE_ARDUINO
    })

@app.route("/vibration", methods=["POST"])
def set_vibration():
    global vibration_enabled
    vibration_enabled = request.get_json().get("on", False)
    return jsonify({"status": "ok", "vibration": vibration_enabled})

@app.route("/upload", methods=["POST"])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier"}), 400
        
    file = request.files['file']
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Type non supporté"}), 400
        
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        file.save(filepath)
        extract_metadata(filepath)  # Pré-charge les métadonnées
        return jsonify({"success": True, "filename": filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === Lancement ===
if __name__ == "__main__":
    logging.info("Démarrage du serveur Tomatis Simulation")
    
    # Thread de lecture
    player_thread = threading.Thread(target=playback_loop)
    player_thread.daemon = True
    player_thread.start()
    
    # Serveur Flask
    app.run(host="0.0.0.0", port=3000, threaded=True)