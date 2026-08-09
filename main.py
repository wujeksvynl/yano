import os
import json
import tempfile
import librosa
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Dict
from shazamio import Shazam

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

def extract_intervals(audio_path: str) -> List[int]:
    """
    Extracts pitch contours from an audio file and converts them to intervals.
    """
    # 1. Load audio
    y, sr = librosa.load(audio_path, sr=None)
    
    # 2. Pitch tracking (pYIN algorithm)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, 
        fmin=librosa.note_to_hz('C2'), 
        fmax=librosa.note_to_hz('C7')
    )
    
    # Filter out unvoiced frames (where no pitch was detected)
    pitches = f0[voiced_flag]
    
    if len(pitches) == 0:
        return []
        
    # 3. Convert Hz to MIDI note numbers
    midi_notes = librosa.hz_to_midi(pitches)
    
    # 4. Smooth/Quantize notes (very simplified approach)
    # Get median note in chunks to simulate "notes" instead of raw frames
    chunk_size = max(1, len(midi_notes) // 10) # arbitrary segmentation for mock
    notes = []
    for i in range(0, len(midi_notes), chunk_size):
        chunk = midi_notes[i:i+chunk_size]
        notes.append(round(np.median(chunk)))
        
    # 5. Convert absolute notes to relative intervals
    intervals = []
    for i in range(1, len(notes)):
        interval = notes[i] - notes[i-1]
        intervals.append(int(interval))
        
    return intervals

def match_motif(user_intervals: List[int], top_n: int = 5) -> List[Dict]:
    """
    Compares the user's interval sequence against the database with tolerance.
    Returns the top_n best matches as a list.
    """
    if not user_intervals:
        return []
        
    results = []
    
    # Matching Parameters
    TOLERANCE = 2    # Allow up to 2 semitones of error per note
    MAX_MISSES = 2   # Allow up to 2 wrong notes in a row without breaking the streak
    
    for entry in CLASSICAL_DATABASE:
        db_intervals = entry["intervals"]
        
        best_streak = 0
        total_matched = 0
        
        # Check all possible alignments between user input and database piece
        for i in range(len(user_intervals)):
            for j in range(len(db_intervals)):
                # If we find a fuzzy starting point
                if abs(user_intervals[i] - db_intervals[j]) <= TOLERANCE:
                    k = 0
                    misses = 0
                    matched_in_streak = 0
                    
                    # Trace forward
                    while (i + k < len(user_intervals) and 
                           j + k < len(db_intervals)):
                        
                        diff = abs(user_intervals[i+k] - db_intervals[j+k])
                        if diff <= TOLERANCE:
                            misses = 0 # reset misses on a good note
                            matched_in_streak += 1
                        else:
                            misses += 1
                            if misses > MAX_MISSES:
                                break
                        k += 1
                        
                    # The valid length is the streak minus any trailing misses
                    valid_length = k - misses
                    best_streak = max(best_streak, valid_length)
                    total_matched = max(total_matched, matched_in_streak)
                    
        # Calculate percentage based on the length of the USER'S motif
        if len(user_intervals) > 0 and best_streak > 0:
            current_score = int((best_streak / len(user_intervals)) * 100)
            
            # Penalty for very short inputs to avoid false positives
            if len(user_intervals) < 5:
                current_score -= 20
            
            # Only include if above minimum threshold    
            if current_score > 20:
                results.append({
                    "composer": entry["composer"],
                    "piece": entry["piece"],
                    "score": min(current_score, 99),
                    "midi_file": entry.get("midi_file"),
                    "matched_notes": best_streak
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
                    shazam_match = {
                        "title": track.get('title', 'Unbekannt'),
                        "artist": track.get('subtitle', 'Unbekannter Künstler'),
                        "cover_art": track.get('images', {}).get('coverarthq') or track.get('images', {}).get('coverart')
                    }
            except Exception as e:
                print(f"Shazam error: {e}")
            
        # ALWAYS run classical interval matching (both modes)
        intervals = extract_intervals(temp_path)
        matches = match_motif(intervals, top_n=5)
        
        return {
            "status": "success",
            "extracted_intervals": intervals,
            "shazam_match": shazam_match,
            "matches": matches
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

