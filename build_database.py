import json
import os
from music21 import corpus, note, chord

def extract_intervals_from_part(part):
    """Extracts relative intervals from a music21 part/stream."""
    notes = []
    # Flatten the stream to just notes and chords
    for element in part.flatten().notes:
        if isinstance(element, note.Note):
            notes.append(element.pitch.midi)
        elif isinstance(element, chord.Chord):
            # Take the highest note of the chord as the melody
            highest_pitch = max(p.midi for p in element.pitches)
            notes.append(highest_pitch)
            
    intervals = []
    for i in range(1, len(notes)):
        intervals.append(notes[i] - notes[i-1])
        
    return intervals

def build_db():
    print("Starte Datenbank-Aufbau (music21)...")
    db = []
    
    # We will pick a representative subset of the corpus for speed.
    # E.g., some Bach chorales, Beethoven, and Mozart.
    composers = {
        'bach': 20,       # take first 20 bach pieces
        'beethoven': 10,  # take 10 beethoven pieces
        'mozart': 10,     # take 10 mozart pieces
        'palestrina': 5
    }
    
    # Create static/midi directory for playback
    midi_dir = os.path.join(os.path.dirname(__file__), 'static', 'midi')
    os.makedirs(midi_dir, exist_ok=True)
    
    for composer, limit in composers.items():
        print(f"Suche nach Werken von {composer.capitalize()}...")
        paths = corpus.getComposer(composer)
        count = 0
        
        for path in paths:
            if count >= limit:
                break
            
            try:
                # Parse the score
                score = corpus.parse(path)
                
                # Get basic metadata
                title = score.metadata.title if score.metadata and score.metadata.title else str(path).split('/')[-1].split('\\')[-1]
                
                # Try to find the melody part (usually the first/highest part)
                parts = score.parts
                if not parts:
                    continue
                    
                melody_part = parts[0]
                intervals = extract_intervals_from_part(melody_part)
                
                # Only add if we found a meaningful melody
                if len(intervals) > 10:
                    item_id = f"{composer}_{count}"
                    midi_filename = f"{item_id}.mid"
                    midi_path = os.path.join(midi_dir, midi_filename)
                    
                    # Write MIDI file
                    score.write('midi', fp=midi_path)
                    
                    db.append({
                        "id": item_id,
                        "composer": composer.capitalize(),
                        "piece": title,
                        "intervals": intervals,
                        "midi_file": f"/static/midi/{midi_filename}"
                    })
                    count += 1
                    print(f"Hinzugefügt: {composer.capitalize()} - {title} ({len(intervals)} Intervalle)")
                    
            except Exception as e:
                print(f"Fehler beim Parsen von {path}: {e}")
                continue

    # Save to JSON
    output_path = os.path.join(os.path.dirname(__file__), 'database.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=2)
        
    print(f"\nFertig! {len(db)} Stücke in {output_path} gespeichert.")

if __name__ == "__main__":
    build_db()
