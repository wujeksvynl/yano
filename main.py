import os
import json
import tempfile
import urllib.request
import urllib.parse
import re
import librosa
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Optional
from shazamio import Shazam
import speech_recognition as sr

app = FastAPI(title="YANO API")

# Mount the static directory for serving MIDI files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Allow CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load database of classical motifs (intervals) from the generated JSON
DB_PATH = os.path.join(os.path.dirname(__file__), 'database.json')
CLASSICAL_DATABASE = []

try:
    if os.path.exists(DB_PATH):
        with open(DB_PATH, 'r', encoding='utf-8') as f:
            CLASSICAL_DATABASE = json.load(f)
        print(f"Loaded {len(CLASSICAL_DATABASE)} pieces from {DB_PATH}")
    else:
        print("Warning: database.json not found. Run build_database.py first.")
except Exception as e:
    print(f"Error loading database: {e}")

LEARNED_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'learned_weights.json')
LEARNED_WEIGHTS = {}
if os.path.exists(LEARNED_WEIGHTS_PATH):
    try:
        with open(LEARNED_WEIGHTS_PATH, 'r', encoding='utf-8') as f:
            LEARNED_WEIGHTS = json.load(f)
    except:
        pass

def save_learned_weights():
    with open(LEARNED_WEIGHTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(LEARNED_WEIGHTS, f, indent=2)

def midi_to_abc(midi_notes: List[float]) -> str:
    if not midi_notes:
        return ""
    pitch_classes_upper = ['C', '^C', 'D', '^D', 'E', 'F', '^F', 'G', '^G', 'A', '^A', 'B']
    pitch_classes_lower = ['c', '^c', 'd', '^d', 'e', 'f', '^f', 'g', '^g', 'a', '^a', 'b']
    
    abc_notes = []
    for m in midi_notes:
        m_int = int(round(m))
        pc = m_int % 12
        octave = (m_int // 12) - 1
        
        if octave < 4:
            note_str = pitch_classes_upper[pc] + ("," * (4 - octave))
        elif octave == 4:
            note_str = pitch_classes_upper[pc]
        elif octave == 5:
            note_str = pitch_classes_lower[pc]
        else:
            note_str = pitch_classes_lower[pc] + ("'" * (octave - 5))
            
        abc_notes.append(note_str)
        
def generate_leadsheet_abc(midi_notes: List[float], chords: List[str], lyrics: str, bpm: int = 120, key: str = "C") -> str:
    if not midi_notes:
        return ""
    
    # 1. Normalize octave into comfortable singing / treble clef range (MIDI 60 = C4 to 77 = F5)
    rounded_notes = [int(round(m)) for m in midi_notes]
    if rounded_notes:
        avg_pitch = sum(rounded_notes) / len(rounded_notes)
        octave_shift = int(round((65 - avg_pitch) / 12)) * 12
        normalized_notes = [max(48, min(84, n + octave_shift)) for n in rounded_notes]
    else:
        normalized_notes = rounded_notes

    pitch_classes_upper = ['C', '^C', 'D', '^D', 'E', 'F', '^F', 'G', '^G', 'A', '^A', 'B']
    pitch_classes_lower = ['c', '^c', 'd', '^d', 'e', 'f', '^f', 'g', '^g', 'a', '^a', 'b']
    
    # Convert to clean ABC note tokens
    abc_tokens = []
    for n in normalized_notes:
        pc = n % 12
        octave = (n // 12) - 1 # 4 is C4, 5 is C5
        if octave < 4:
            note_str = pitch_classes_upper[pc] + ("," * max(1, 4 - octave))
        elif octave == 4:
            note_str = pitch_classes_upper[pc]
        elif octave == 5:
            note_str = pitch_classes_lower[pc]
        else:
            note_str = pitch_classes_lower[pc] + ("'" * max(1, octave - 5))
        abc_tokens.append(note_str)

    # 2. Filter distinct main chords
    clean_chords = [c.replace("other", "").replace("diminished", "dim").strip() for c in chords if c.strip()]
    if not clean_chords:
        clean_chords = ["C"]

    # 3. Structure into 4/4 measures (4 quarter notes per measure)
    measures = []
    current_measure = []
    notes_per_measure = 4
    chord_idx = 0
    
    for i, note in enumerate(abc_tokens):
        if len(current_measure) == 0:
            c = clean_chords[chord_idx % len(clean_chords)]
            chord_idx += 1
            current_measure.append(f'"{c}"{note}')
        else:
            current_measure.append(note)
            
        if len(current_measure) == notes_per_measure:
            measures.append(current_measure)
            current_measure = []
            
    if current_measure:
        measures.append(current_measure)
        
    # Clean lyrics words
    raw_words = []
    for line in lyrics.split('\n'):
        l_str = line.strip()
        if l_str and not l_str.startswith('(') and not l_str.startswith('['):
            # Clean punctuation from words for sheet music
            cleaned_line = l_str.replace('|', ' ').replace('-', ' ')
            raw_words.extend(cleaned_line.split())
            
    word_idx = 0
    score_blocks = []
    
    # Group measures into lines of 4 measures each
    for m_i in range(0, len(measures), 4):
        line_measures = measures[m_i:m_i+4]
        # Line of music
        music_line = " | ".join([" ".join(m) for m in line_measures]) + " |"
        
        # Line of lyrics (w:) aligned to each beat
        lyric_measure_parts = []
        has_lyrics_in_line = False
        
        for m in line_measures:
            m_words = []
            for _ in range(len(m)):
                if word_idx < len(raw_words):
                    w = raw_words[word_idx]
                    word_idx += 1
                    m_words.append(w)
                    has_lyrics_in_line = True
                else:
                    m_words.append("*")
            lyric_measure_parts.append(" ".join(m_words))
            
        score_blocks.append(music_line)
        if has_lyrics_in_line:
            score_blocks.append("w: " + " | ".join(lyric_measure_parts) + " |")

    return "\n".join(score_blocks)

def generate_campfire_leadsheet(chords: List[str], lyrics: str, bpm: int = 120, key: str = "C major") -> str:
    """
    Generates a professional text-based leadsheet (Chords directly over Lyrics with tempo & key).
    """
    lines = []
    lines.append("=" * 68)
    lines.append("                       YANO SONG LEADSHEET")
    lines.append(f"  Tempo: ~{bpm} BPM   |   Takt: 4/4   |   Tonart: {key}")
    lines.append("=" * 68)
    lines.append("")
    
    unique_chords = []
    for c in chords:
        clean_c = c.replace("other", "").replace("diminished", "dim").strip()
        if clean_c and clean_c not in unique_chords:
            unique_chords.append(clean_c)
            
    if unique_chords:
        lines.append("[Verwendete Hauptakkorde]")
        lines.append("   " + "   ".join([f"[{c}]" for c in unique_chords]))
        lines.append("")
        
    lines.append("-" * 68)
    lines.append("[Liedtext & Akkorde]")
    lines.append("")
    
    raw_lines = [l.strip() for l in lyrics.split('\n') if l.strip() and not l.startswith('(') and not l.startswith('[')]
    
    if not raw_lines or (len(raw_lines) == 1 and "Kein Text" in raw_lines[0]):
        # Instrumental measure grid
        lines.append("  (Instrumental - Akkordablauf im 4/4 Takt)")
        lines.append("")
        prog = chords if chords else ["C"]
        for i in range(0, len(prog), 4):
            bar_slice = prog[i:i+4]
            bar_strs = [f"| {c:<5} .  .  . " for c in bar_slice]
            lines.append("  " + "".join(bar_strs) + "|")
    else:
        chord_idx = 0
        prog = chords if chords else ["C", "G", "Am", "F"]
        
        for section_idx, text_line in enumerate(raw_lines):
            words = text_line.split()
            if not words:
                continue
                
            # Split into phrase chunks of ~6-8 words for optimal leadsheet reading
            chunk_size = 7
            word_chunks = [words[i:i+chunk_size] for i in range(0, len(words), chunk_size)]
            
            for chunk in word_chunks:
                chunk_str = " ".join(chunk)
                
                # Assign 2 chords per line/chunk
                c1 = prog[chord_idx % len(prog)]
                chord_idx += 1
                c2 = prog[chord_idx % len(prog)]
                chord_idx += 1
                
                half_len = max(len(c1) + 4, len(chunk_str) // 2)
                chord_line = f"  {c1:<{half_len}}{c2}"
                lyric_line = f"  {chunk_str}"
                
                lines.append(chord_line)
                lines.append(lyric_line)
                lines.append("") # Spacing after each phrase
                
    lines.append("=" * 68)
    return "\n".join(lines)

def clean_lyrics_text(text: str) -> str:
    """
    Strips LRC timestamps like [00:12.34], header markers, and formats clean lyrics.
    """
    if not text:
        return ""
    # Remove LRC timestamps like [01:23.45] or [01:23]
    cleaned = re.sub(r'\[\d{2}:\d{2}(\.\d+)?\]', '', text)
    # Filter empty brackets/headers like [Chorus] or [Verse 1]
    lines = [
        line.strip() for line in cleaned.split('\n') 
        if line.strip() and not (line.strip().startswith('[') and line.strip().endswith(']'))
    ]
    return "\n".join(lines)

def fetch_lyrics_online(artist: str, title: str) -> str:
    """
    Fetches official studio lyrics from online databases (LRCLIB, lyrics.ovh).
    """
    if not artist or not title or artist == "Unbekannter Künstler" or title == "Unbekannt":
        return ""
        
    clean_artist = artist.split('feat.')[0].split('Feat.')[0].split('&')[0].strip()
    clean_title = title.split('(')[0].split('-')[0].strip()
    
    # 1. Try LRCLIB (free, fast, high accuracy)
    try:
        query = urllib.parse.urlencode({'artist_name': clean_artist, 'track_name': clean_title})
        url = f"https://lrclib.net/api/get?{query}"
        req = urllib.request.Request(url, headers={'User-Agent': 'YANO-MusicApp/1.0'})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            plain_lyrics = data.get('plainLyrics')
            if plain_lyrics and plain_lyrics.strip():
                return clean_lyrics_text(plain_lyrics.strip())
            synced = data.get('syncedLyrics')
            if synced and synced.strip():
                return clean_lyrics_text(synced.strip())
    except Exception as e:
        print(f"LRCLIB fetch error: {e}")
        
    # 2. Try lyrics.ovh fallback
    try:
        enc_artist = urllib.parse.quote(clean_artist)
        enc_title = urllib.parse.quote(clean_title)
        url = f"https://api.lyrics.ovh/v1/{enc_artist}/{enc_title}"
        req = urllib.request.Request(url, headers={'User-Agent': 'YANO-MusicApp/1.0'})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            lyrics = data.get('lyrics')
            if lyrics and lyrics.strip():
                return clean_lyrics_text(lyrics.strip())
    except Exception as e:
        print(f"Lyrics.ovh fetch error: {e}")
        
    return ""

def extract_lyrics(audio_path: str) -> str:
    """
    Uses SpeechRecognition to extract German/English lyrics for the ENTIRE song via chunking.
    """
    recognizer = sr.Recognizer()
    full_transcript = []
    try:
        with sr.AudioFile(audio_path) as source:
            total_duration = getattr(source, 'DURATION', 0)
            if not total_duration or total_duration <= 0:
                audio_data = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio_data, language="de-DE")
                    return text
                except sr.UnknownValueError:
                    try:
                        return recognizer.recognize_google(audio_data, language="en-US")
                    except:
                        return "(Kein Text erkannt oder nur Instrumental)"
                        
            chunk_len = 40 # 40-second chunks across the full song
            offset = 0
            while offset < total_duration:
                current_chunk = min(chunk_len, total_duration - offset)
                audio_data = recognizer.record(source, duration=current_chunk)
                
                chunk_text = ""
                try:
                    chunk_text = recognizer.recognize_google(audio_data, language="de-DE")
                except sr.UnknownValueError:
                    try:
                        chunk_text = recognizer.recognize_google(audio_data, language="en-US")
                    except sr.UnknownValueError:
                        pass
                except Exception as e:
                    print(f"Speech recognition chunk error: {e}")
                    
                if chunk_text.strip():
                    full_transcript.append(chunk_text.strip())
                offset += current_chunk
                
        return "\n".join(full_transcript) if full_transcript else "(Kein Text erkannt oder nur Instrumental)"
    except Exception as e:
        return f"(Fehler beim Lesen der Audio-Datei für Text: {e})"

def extract_features(audio_path: str) -> Dict:
    """
    Extracts intervals, absolute notes, key, and chords from the full audio.
    """
    # 1. Load full audio with 16kHz for fast and accurate processing
    y, sr = librosa.load(audio_path, sr=16000)
    
    # 2. Pitch tracking (pYIN algorithm on full song)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, 
        fmin=librosa.note_to_hz('C2'), 
        fmax=librosa.note_to_hz('C7'),
        sr=sr,
        hop_length=1024
    )
    
    # Filter out unvoiced frames (where no pitch was detected)
    pitches = f0[voiced_flag]
    
    if len(pitches) == 0:
        return {"intervals": [], "midi_notes": [], "key": "", "chords": []}
        
    # 3. Convert Hz to MIDI note numbers
    raw_midi = librosa.hz_to_midi(pitches)
    
    # 4. Adaptive note segmentation: group consecutive frames within 1 semitone
    notes = []
    current_group = []
    for p in raw_midi:
        r_p = round(p)
        if not current_group:
            current_group.append(r_p)
        else:
            median_p = np.median(current_group)
            if abs(r_p - median_p) <= 1.0:
                current_group.append(r_p)
            else:
                if len(current_group) >= 3:
                    notes.append(int(round(np.median(current_group))))
                current_group = [r_p]
    if len(current_group) >= 3:
        notes.append(int(round(np.median(current_group))))
        
    # Fallback to chunking if adaptive segmentation yields too few notes
    if len(notes) < 3 and len(raw_midi) > 0:
        chunk_size = max(1, len(raw_midi) // 10)
        notes = []
        for i in range(0, len(raw_midi), chunk_size):
            chunk = raw_midi[i:i+chunk_size]
            notes.append(int(round(np.median(chunk))))
        
    # 5. Convert absolute notes to relative intervals
    intervals = []
    for i in range(1, len(notes)):
        interval = notes[i] - notes[i-1]
        intervals.append(int(interval))
        
    # 6. Extract Key and Chords using Chroma
    import scipy.signal
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    
    # Key estimation
    chroma_sum = np.sum(chroma, axis=1)
    maj_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    min_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    maj_profile = maj_profile / np.linalg.norm(maj_profile)
    min_profile = min_profile / np.linalg.norm(min_profile)
    norm = np.linalg.norm(chroma_sum)
    if norm > 0:
        chroma_norm = chroma_sum / norm
    else:
        chroma_norm = chroma_sum
    maj_corrs = [np.dot(np.roll(maj_profile, i), chroma_norm) for i in range(12)]
    min_corrs = [np.dot(np.roll(min_profile, i), chroma_norm) for i in range(12)]
    
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    best_maj_idx = np.argmax(maj_corrs)
    best_min_idx = np.argmax(min_corrs)
    
    if maj_corrs[best_maj_idx] > min_corrs[best_min_idx]:
        estimated_key = f"{note_names[best_maj_idx]} major"
    else:
        estimated_key = f"{note_names[best_min_idx]} minor"
        
    # Chord extraction (Dual Engine: Pro Extended Chords & Simplified Campfire Chords)
    templates = []
    chord_names = []
    for i in range(12):
        root = note_names[i]
        # Major Triad (1, 3, 5)
        t = np.zeros(12); t[i]=1; t[(i+4)%12]=1; t[(i+7)%12]=1
        templates.append(t / np.linalg.norm(t)); chord_names.append(root)
        
        # Minor Triad (1, b3, 5)
        t = np.zeros(12); t[i]=1; t[(i+3)%12]=1; t[(i+7)%12]=1
        templates.append(t / np.linalg.norm(t)); chord_names.append(root + 'm')
        
        # Dominant 7th (1, 3, 5, b7)
        t = np.zeros(12); t[i]=1; t[(i+4)%12]=1; t[(i+7)%12]=1; t[(i+10)%12]=0.85
        templates.append(t / np.linalg.norm(t)); chord_names.append(root + '7')
        
        # Major 7th (1, 3, 5, 7)
        t = np.zeros(12); t[i]=1; t[(i+4)%12]=1; t[(i+7)%12]=1; t[(i+11)%12]=0.85
        templates.append(t / np.linalg.norm(t)); chord_names.append(root + 'maj7')
        
        # Minor 7th (1, b3, 5, b7)
        t = np.zeros(12); t[i]=1; t[(i+3)%12]=1; t[(i+7)%12]=1; t[(i+10)%12]=0.85
        templates.append(t / np.linalg.norm(t)); chord_names.append(root + 'm7')
        
        # Sus4 (1, 4, 5)
        t = np.zeros(12); t[i]=1; t[(i+5)%12]=1; t[(i+7)%12]=1
        templates.append(t / np.linalg.norm(t)); chord_names.append(root + 'sus4')
        
        # Diminished (1, b3, b5)
        t = np.zeros(12); t[i]=1; t[(i+3)%12]=1; t[(i+6)%12]=1
        templates.append(t / np.linalg.norm(t)); chord_names.append(root + 'dim')
        
    templates = np.array(templates)
    
    try:
        # Avoid crash on very short audio
        kernel = min(43, max(3, chroma.shape[1] // 2))
        if kernel % 2 == 0: kernel += 1
        chroma_smoothed = scipy.signal.medfilt(chroma, kernel_size=(1, kernel))
        detected_chords = []
        for i in range(chroma_smoothed.shape[1]):
            frame = chroma_smoothed[:, i]
            norm_frame = frame / (np.linalg.norm(frame) + 1e-6)
            corrs = np.dot(templates, norm_frame)
            best_idx = np.argmax(corrs)
            detected_chords.append(chord_names[best_idx])
            
        pro_chord_sequence = []
        for c in detected_chords:
            if not pro_chord_sequence or pro_chord_sequence[-1] != c:
                pro_chord_sequence.append(c)
    except:
        pro_chord_sequence = []
        
    # Simplify for campfire
    campfire_chord_sequence = []
    enharmonics = {'A#': 'Bb', 'D#': 'Eb', 'G#': 'Ab', 'A#m': 'Bbm', 'D#m': 'Ebm', 'G#m': 'Abm'}
    for c in pro_chord_sequence:
        simple = c
        for ext in ['maj7', 'm7b5', 'm7', '7', 'sus4', 'add9', 'dim7', 'dim', 'aug', '6']:
            if simple.endswith(ext):
                if ext.startswith('m7') or ext == 'm':
                    simple = simple[:-len(ext)] + 'm'
                else:
                    simple = simple[:-len(ext)]
                break
        simple = enharmonics.get(simple, simple)
        if not campfire_chord_sequence or campfire_chord_sequence[-1] != simple:
            campfire_chord_sequence.append(simple)
        
    # 7. Tempo / BPM estimation
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo_val = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            tempo_val = float(tempo)
        bpm = int(round(tempo_val)) if tempo_val > 0 else 120
    except:
        bpm = 120
        
    return {
        "intervals": intervals,
        "midi_notes": notes,
        "key": estimated_key,
        "bpm": bpm,
        "chords": pro_chord_sequence,
        "pro_chords": pro_chord_sequence,
        "campfire_chords": campfire_chord_sequence
    }


def get_origin_description(composer: str, piece: str) -> str:
    composer_lower = composer.lower()
    if "bach" in composer_lower:
        return "Barock-Epoche (18. Jh.) – Kirchenkantaten & Choräle von Johann Sebastian Bach."
    elif "beatles" in composer_lower:
        return "Pop/Rock-Klassiker – Komposition von Lennon/McCartney (The Beatles)."
    elif "adele" in composer_lower:
        return "Modernes Pop/Soul-Werk – Berühmt für gefühlvolle Kadenzen und Melodieführung."
    elif "ed sheeran" in composer_lower:
        return "Zeitgenössischer Akustik-Pop – Geprägt von eingängigen 4-Chord-Kadenzen."
    elif "mozart" in composer_lower:
        return "Wiener Klassik (18. Jh.) – Typische periodische Melodie- und Phrasenbildung."
    elif "beethoven" in composer_lower:
        return "Klassik/Romantik – Kraftvolle rhythmische Motive und harmonische Progressionen."
    else:
        return f"Historisches Werk aus der Musikgeschichte ({composer})."

def compare_two_features(feat_a: Dict, feat_b: Dict) -> Dict:
    a_intervals = feat_a.get("intervals", [])
    b_intervals = feat_b.get("intervals", [])
    a_notes = feat_a.get("midi_notes", [])
    b_notes = feat_b.get("midi_notes", [])
    a_key = feat_a.get("key", "").strip()
    b_key = feat_b.get("key", "").strip()
    a_chords = feat_a.get("chords", [])
    b_chords = feat_b.get("chords", [])
    
    if not a_intervals or not b_intervals:
        return {
            "score": 0, 
            "matched_notes": 0, 
            "insights": [],
            "reasons": [],
            "explanation": "Keine ausreichenden Tonintervalle zur Analyse vorhanden.",
            "shared_chords": [],
            "transposition": "Unbekannt"
        }
        
    TOLERANCE = 2
    MAX_MISSES = 2
    
    best_streak = 0
    total_matched = 0
    
    for i in range(len(a_intervals)):
        for j in range(len(b_intervals)):
            if abs(a_intervals[i] - b_intervals[j]) <= TOLERANCE:
                k = 0
                misses = 0
                matched_in_streak = 0
                
                while (i + k < len(a_intervals) and 
                       j + k < len(b_intervals)):
                    
                    diff = abs(a_intervals[i+k] - b_intervals[j+k])
                    if diff <= TOLERANCE:
                        misses = 0
                        matched_in_streak += 1
                    else:
                        misses += 1
                        if misses > MAX_MISSES:
                            break
                    k += 1
                    
                valid_length = k - misses
                best_streak = max(best_streak, valid_length)
                total_matched = max(total_matched, matched_in_streak)
                
    current_score = 0
    if len(a_intervals) > 0 and best_streak > 0:
        current_score = int((best_streak / len(a_intervals)) * 100)
        if len(a_intervals) < 5:
            current_score -= 20
            
    current_score = min(max(current_score, 0), 100)
    
    # Calculate shared chords
    shared_chords = []
    if a_chords and b_chords:
        clean_a = [c.replace("other", "").replace("diminished", "dim") for c in a_chords if c]
        clean_b = [c.replace("other", "").replace("diminished", "dim") for c in b_chords if c]
        shared_chords = sorted(list(set(clean_a).intersection(set(clean_b))))

    # Calculate Transposition
    semitone_shift = 0
    transposition_text = "Original-Tonhöhe"
    if len(a_notes) > 0 and len(b_notes) > 0:
        avg_a = sum(a_notes) / len(a_notes)
        avg_b = sum(b_notes) / len(b_notes)
        diff_semi = round(avg_a - avg_b)
        semitone_shift = diff_semi
        if abs(diff_semi) <= 1:
            transposition_text = "Exakt gleiche Tonhöhe (Original)"
        elif diff_semi > 0:
            transposition_text = f"Transponiert (+{diff_semi} Halbtöne höher)"
        else:
            transposition_text = f"Transponiert ({diff_semi} Halbtöne tiefer)"

    insights = []
    reasons = []
    
    if current_score >= 20:
        insights.append(f"📈 {current_score}% Melodie-Match")
        reasons.append(f"🎶 **Melodieverlauf:** Eine zusammenhängende Tonfolge von **{best_streak} Intervallen** folgt exakt derselben Auf-/Abwärtsbewegung.")
        
        if abs(semitone_shift) <= 1:
            insights.append("🎯 Exakte Tonhöhe")
            reasons.append("🎯 **Tonhöhe:** Deine Töne treffen exakt die absolute Tonhöhe des Originals.")
        else:
            insights.append("🔄 Transponiert")
            reasons.append(f"🔄 **Transposition:** Die Melodie ist musikalisch verwandt, aber {transposition_text.lower()} angesetzt.")
            
        if a_key and b_key and a_key.lower() == b_key.lower():
            insights.append(f"🔑 Tonart: {a_key}")
            reasons.append(f"🔑 **Tonart-Übereinstimmung:** Beide Stücke basieren auf der Tonart **{a_key}**.")
        elif a_key and b_key:
            reasons.append(f"🔑 **Tonart:** Deine Melodie liegt in **{a_key}**, das Vergleichsstück in **{b_key}**.")
            
        if len(shared_chords) >= 2:
            chords_str = ", ".join(shared_chords[:4])
            insights.append(f"🎸 Akkorde: {chords_str}")
            reasons.append(f"🎸 **Harmonische Basis:** Gemeinsame Akkorde erkannt ({chords_str}).")

    explanation = " | ".join([r.replace("**", "").replace("🎶 ", "").replace("🎯 ", "").replace("🔄 ", "").replace("🔑 ", "").replace("🎸 ", "") for r in reasons])

    return {
        "score": current_score,
        "matched_notes": best_streak,
        "insights": insights,
        "reasons": reasons,
        "explanation": explanation,
        "shared_chords": shared_chords,
        "transposition": transposition_text,
        "user_key": a_key,
        "target_key": b_key
    }

def match_motif(user_features: Dict, top_n: int = 5) -> List[Dict]:
    """
    Compares the user's features against the database.
    Returns the top_n best matches with detailed insights, reasons, and origin info.
    """
    if not user_features.get("intervals"):
        return []
        
    results = []
    
    for entry in CLASSICAL_DATABASE:
        comp_res = compare_two_features(user_features, entry)
        score = comp_res["score"]
        
        if score > 20:
            piece_key = f"{entry['composer']}_{entry['piece']}"
            # Apply learned weights (boost)
            boost = LEARNED_WEIGHTS.get(piece_key, 0)
            final_score = min(score + boost, 99)
            
            origin_desc = get_origin_description(entry['composer'], entry['piece'])
            
            results.append({
                "composer": entry["composer"],
                "piece": entry["piece"],
                "score": final_score,
                "midi_file": entry.get("midi_file"),
                "matched_notes": comp_res["matched_notes"],
                "insights": comp_res["insights"],
                "reasons": comp_res["reasons"],
                "explanation": comp_res["explanation"],
                "shared_chords": comp_res["shared_chords"],
                "transposition": comp_res["transposition"],
                "target_key": entry.get("key", ""),
                "origin_info": origin_desc
            })
    
    # Sort by score descending, take top_n
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Deduplicate by piece name (some pieces appear multiple times as different movements)
    seen_pieces = set()
    unique_results = []
    for r in results:
        key = f"{r['composer']}_{r['piece']}"
        if key not in seen_pieces:
            seen_pieces.add(key)
            unique_results.append(r)
        if len(unique_results) >= top_n:
            break
    
    return unique_results


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...), mode: str = Form("klassik")):
    # Save uploaded file to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name
        
    try:
        shazam_match = None
        
        # If mode is 'pop', run Shazam
        if mode == "pop":
            try:
                shazam = Shazam()
                out = await shazam.recognize(temp_path)
                
                if 'track' in out:
                    track = out['track']
                    title = track.get('title', 'Unbekannt')
                    artist = track.get('subtitle', 'Unbekannter Künstler')
                    cover = track.get('images', {}).get('coverarthq') or track.get('images', {}).get('coverart')
                    
                    db_lyrics = ""
                    sections = track.get('sections', [])
                    for sec in sections:
                        if sec.get('type') == 'LYRICS' and 'text' in sec:
                            db_lyrics = clean_lyrics_text("\n".join(sec['text']))
                            break
                    if not db_lyrics and artist and title:
                        db_lyrics = fetch_lyrics_online(artist, title)
                        
                    shazam_match = {
                        "title": title,
                        "artist": artist,
                        "cover_art": cover,
                        "lyrics": db_lyrics
                    }
            except Exception as e:
                print(f"Shazam error: {e}")
            
        # ALWAYS run classical interval matching (both modes)
        features = extract_features(temp_path)
        matches = match_motif(features, top_n=5)
        
        return {
            "status": "success",
            "extracted_intervals": features.get('intervals', []),
            "shazam_match": shazam_match,
            "matches": matches
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/feedback")
async def receive_feedback(
    composer: str = Form(...),
    piece: str = Form(...),
    is_positive: str = Form(...),
    comment: str = Form("")
):
    import datetime
    feedback_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "composer": composer,
        "piece": piece,
        "is_positive": is_positive.lower() == 'true',
        "comment": comment
    }
    
    FEEDBACK_PATH = os.path.join(os.path.dirname(__file__), 'feedback.json')
    feedbacks = []
    if os.path.exists(FEEDBACK_PATH):
        try:
            with open(FEEDBACK_PATH, 'r', encoding='utf-8') as f:
                feedbacks = json.load(f)
        except: pass
    feedbacks.append(feedback_entry)
    with open(FEEDBACK_PATH, 'w', encoding='utf-8') as f:
        json.dump(feedbacks, f, indent=2)
        
    if feedback_entry["is_positive"]:
        key = f"{composer}_{piece}"
        LEARNED_WEIGHTS[key] = LEARNED_WEIGHTS.get(key, 0) + 5
        save_learned_weights()
        
    return {"status": "success"}

@app.post("/compare")
async def compare_audio(file_a: UploadFile = File(...), file_b: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_a:
        temp_a.write(await file_a.read())
        path_a = temp_a.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_b:
        temp_b.write(await file_b.read())
        path_b = temp_b.name
        
    try:
        feat_a = extract_features(path_a)
        feat_b = extract_features(path_b)
        
        comp_res = compare_two_features(feat_a, feat_b)
        
        matches_a = match_motif(feat_a, top_n=3)
        matches_b = match_motif(feat_b, top_n=3)
        
        return {
            "status": "success",
            "score": comp_res["score"],
            "insights": comp_res["insights"],
            "reasons": comp_res.get("reasons", []),
            "explanation": comp_res.get("explanation", ""),
            "shared_chords": comp_res.get("shared_chords", []),
            "transposition": comp_res.get("transposition", ""),
            "abc_a": midi_to_abc(feat_a.get("midi_notes", [])),
            "abc_b": midi_to_abc(feat_b.get("midi_notes", [])),
            "similar_a": matches_a,
            "similar_b": matches_b
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(path_a): os.remove(path_a)
        if os.path.exists(path_b): os.remove(path_b)

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...), custom_lyrics: Optional[str] = Form(None)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name
        
    wav_path = None
    try:
        # Convert any uploaded file (e.g. MP3) to a standard PCM WAV 
        # so SpeechRecognition can read it, using librosa/soundfile which can decode it.
        import soundfile as sf
        y, sr_rate = librosa.load(temp_path, sr=16000)
        wav_path = temp_path + "_pcm.wav"
        sf.write(wav_path, y, sr_rate, subtype='PCM_16')
        process_path = wav_path
        
        # 1. Melodie und Akkorde
        features = extract_features(process_path)
        abc_notation = midi_to_abc(features.get("midi_notes", []))
        
        raw_pro_chords = features.get("pro_chords", [])
        raw_campfire_chords = features.get("campfire_chords", [])
        
        # Pro chords processing
        pro_unique = []
        for c in raw_pro_chords:
            if c and c not in pro_unique:
                pro_unique.append(c)
        pro_progression = []
        for c in raw_pro_chords:
            if c and (not pro_progression or pro_progression[-1] != c):
                pro_progression.append(c)
                
        # Campfire chords processing (simplified open chords)
        campfire_unique = []
        for c in raw_campfire_chords:
            if c and c not in campfire_unique:
                campfire_unique.append(c)
        campfire_progression = []
        for c in raw_campfire_chords:
            if c and (not campfire_progression or campfire_progression[-1] != c):
                campfire_progression.append(c)
        
        # 2. Text (Lyrics) & Online Music Database Song Recognition
        recognized_song = None
        lyrics_source = "custom" if (custom_lyrics and custom_lyrics.strip()) else "speech_recognition"
        
        if custom_lyrics and custom_lyrics.strip():
            lyrics = custom_lyrics.strip()
        else:
            db_lyrics = ""
            try:
                # Recognize song via Shazam
                shazam = Shazam()
                out = await shazam.recognize(process_path)
                if 'track' in out:
                    track = out['track']
                    title = track.get('title', '')
                    artist = track.get('subtitle', '')
                    cover = track.get('images', {}).get('coverarthq') or track.get('images', {}).get('coverart')
                    
                    # 1. Check if Shazam returned lyrics directly in sections
                    sections = track.get('sections', [])
                    for sec in sections:
                        if sec.get('type') == 'LYRICS' and 'text' in sec:
                            db_lyrics = clean_lyrics_text("\n".join(sec['text']))
                            break
                            
                    # 2. If not in sections, fetch from online lyrics databases (LRCLIB, lyrics.ovh)
                    if not db_lyrics and artist and title:
                        db_lyrics = fetch_lyrics_online(artist, title)
                        
                    if artist or title:
                        recognized_song = {
                            "title": title,
                            "artist": artist,
                            "cover_art": cover,
                            "lyrics_found": bool(db_lyrics)
                        }
            except Exception as e:
                print(f"Song recognition in transcribe error: {e}")
                
            if db_lyrics:
                lyrics = db_lyrics
                lyrics_source = "database"
            else:
                lyrics = extract_lyrics(process_path)
                lyrics_source = "speech_recognition"
        
        bpm = features.get("bpm", 120)
        key = features.get("key", "C major")
        
        # 3. Professional Leadsheet ABC (with rich Pro Jazz/Pop extensions)
        leadsheet_abc = generate_leadsheet_abc(features.get("midi_notes", []), pro_progression if pro_progression else pro_unique, lyrics, bpm=bpm, key=key)
        
        # 4. Simplified Campfire Leadsheet (easy acoustic chords & aligned text)
        campfire_text = generate_campfire_leadsheet(campfire_progression if campfire_progression else campfire_unique, lyrics, bpm=bpm, key=key)
        
        return {
            "status": "success",
            "abc_notation": abc_notation,
            "chords": pro_unique,
            "pro_chords": pro_unique,
            "pro_progression": pro_progression,
            "campfire_chords": campfire_unique,
            "campfire_progression": campfire_progression,
            "unique_chords": campfire_unique,
            "progression": campfire_progression,
            "lyrics": lyrics,
            "lyrics_source": lyrics_source,
            "recognized_song": recognized_song,
            "bpm": bpm,
            "key": key,
            "leadsheet_abc": leadsheet_abc,
            "campfire_text": campfire_text
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

