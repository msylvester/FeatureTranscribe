import whisper
from pathlib import Path
import json
import argparse
import subprocess
import torch
import time
import numpy as np
from datetime import timedelta
from pydub import AudioSegment
import librosa
import soundfile as sf

def format_time(seconds):
    """Convert seconds into human readable time string"""
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"

def check_gpu():
    """Check if CUDA GPU is available and print device info"""
    if torch.cuda.is_available():
        device = torch.cuda.get_device_properties(0)
        print(f"Using GPU: {device.name} with {device.total_memory / 1024**3:.2f} GB memory")
        return "cuda"
    else:
        print("No GPU found, using CPU")
        return "cpu"

def extract_audio_features(audio_segment, start_time, end_time):
    """Extract audio features for a segment including volume and emotional characteristics"""
    # Convert milliseconds to samples
    start_sample = int(start_time * 1000)
    end_sample = int(end_time * 1000)
    
    # Extract the specific segment
    segment = audio_segment[start_sample:end_sample]
    
    # Convert to numpy array for analysis
    samples = np.array(segment.get_array_of_samples())
    
    # Calculate basic audio features
    rms = librosa.feature.rms(y=samples.astype(float))[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(samples.astype(float))[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=samples.astype(float), sr=segment.frame_rate)[0]
    
    # Calculate average values
    avg_volume = float(np.mean(rms))
    avg_zcr = float(np.mean(zero_crossing_rate))
    avg_spectral_centroid = float(np.mean(spectral_centroid))
    
    # Determine volume level
    if avg_volume < 0.1:
        volume_level = "quiet"
    elif avg_volume < 0.3:
        volume_level = "normal"
    else:
        volume_level = "loud"
    
    # Estimate emotional characteristics based on audio features
    # High ZCR and spectral centroid often indicate excitement/intensity
    intensity = "high" if avg_zcr > 0.15 and avg_spectral_centroid > 2000 else "normal"
    
    return {
        "volume": {
            "level": volume_level,
            "value": avg_volume
        },
        "characteristics": {
            "intensity": intensity,
            "zero_crossing_rate": avg_zcr,
            "spectral_centroid": avg_spectral_centroid
        }
    }

def extract_audio(video_path):
    """Extract audio from video using ffmpeg"""
    video_file = Path(video_path)
    audio_path = video_file.with_suffix('.wav')
    
    # Extract audio using ffmpeg with optimal settings
    subprocess.run([
        'ffmpeg', '-i', str(video_file), 
        '-vn', '-acodec', 'pcm_s16le', 
        '-ar', '16000', '-ac', '1', 
        str(audio_path)
    ])
    
    return audio_path

def transcribe_with_features(model, audio_path, device):
    """Get transcription with timestamps and audio features"""
    print("Generating enhanced transcription...")
    enhanced_segments = []
    
    # Load the full audio file for feature extraction
    audio = AudioSegment.from_wav(str(audio_path))
    
    transcribe_start = time.time()
    
    # Run Whisper transcription
    if device == "cuda":
        with torch.cuda.amp.autocast():
            result = model.transcribe(str(audio_path), language='en', fp16=True)
    else:
        result = model.transcribe(str(audio_path), language='en', fp16=False)
    
    # Process segments with audio feature extraction
    for segment in result["segments"]:
        # Extract audio features for this segment
        audio_features = extract_audio_features(
            audio,
            segment["start"],
            segment["end"]
        )
        
        # Combine transcription with audio features
        enhanced_segments.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
            "audio_features": audio_features
        })
    
    transcribe_end = time.time()
    print(f"Enhanced transcription processing took: {format_time(transcribe_end - transcribe_start)}")
    
    return enhanced_segments

def process_video(video_path, model_size="base"):
    """Process video to create enhanced transcription"""
    process_start = time.time()
    device = check_gpu()
    
    video_file = Path(video_path)
    transcription_path = video_file.with_suffix('.enhanced_transcription.json')
    
    print(f"Processing {video_file.name}...")
    
    # Extract audio
    audio_path = extract_audio(video_path)
    
    # Load and optimize Whisper model
    print(f"Loading Whisper {model_size} model...")
    model = whisper.load_model(model_size)
    if device == "cuda":
        model = model.cuda()
    
    # Generate enhanced transcription
    enhanced_transcription = transcribe_with_features(model, audio_path, device)
    
    # Save enhanced transcription
    with open(transcription_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_transcription, f, indent=2, ensure_ascii=False)
    
    # Cleanup
    audio_path.unlink()
    
    process_end = time.time()
    print(f"Total processing time: {format_time(process_end - process_start)}")
    print(f"Enhanced transcription saved to {transcription_path}")

def main():
    parser = argparse.ArgumentParser(description='Create enhanced transcription with audio features')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--model', default='base', 
                      choices=['tiny', 'base', 'small', 'medium', 'large', 'turbo'],
                      help='Whisper model size to use')
    
    args = parser.parse_args()
    process_video(args.video_path, model_size=args.model)

if __name__ == "__main__":
    main()
