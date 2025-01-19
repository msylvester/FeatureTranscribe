import whisper
from pathlib import Path
import json
import argparse
import subprocess
import torch
import time
from datetime import timedelta
from pydub import AudioSegment

def check_gpu():
    """Check if CUDA GPU is available and print device info"""
    if torch.cuda.is_available():
        device = torch.cuda.get_device_properties(0)
        print(f"Using GPU: {device.name} with {device.total_memory / 1024**3:.2f} GB memory")
        return "cuda"
    else:
        print("No GPU found, using CPU")
        return "cpu"

def format_time(seconds):
    """Convert seconds to human readable time"""
    return str(timedelta(seconds=int(seconds)))

def optimize_model(model, device):
    """Optimize the model using torch.compile() if available"""
    if hasattr(torch, 'compile'):
        try:
            print("Optimizing model with torch.compile()...")
            # Using reduced precision for better performance
            if device == "cuda":
                model = model.half()  # Convert to FP16 for GPU
            
            # Compile the model with performance optimizations
            model = torch.compile(
                model,
                mode='reduce-overhead',  # Optimize for inference
                fullgraph=True,  # Enable full graph optimization
                dynamic=True,  # Enable dynamic shape handling
            )
            print("Model optimization complete!")
        except Exception as e:
            print(f"Warning: Model optimization failed: {e}")
            print("Continuing with unoptimized model...")
    else:
        print("torch.compile() not available. Using standard model.")
    
    return model

def extract_audio(video_path):
    """Extract audio from video using ffmpeg"""
    video_file = Path(video_path)
    audio_path = video_file.with_suffix('.wav')
    
    # Extract audio using ffmpeg with optimal settings for Whisper
    subprocess.run([
        'ffmpeg', '-i', str(video_file), 
        '-vn', '-acodec', 'pcm_s16le', 
        '-ar', '16000', '-ac', '1', 
        str(audio_path)
    ])
    
    return audio_path

def transcribe_with_timestamps(model, audio_path, device):
    """Get transcription with timestamps using Whisper"""
    print("Generating timestamped transcription...")
    words_with_timestamps = []
    
    transcribe_start = time.time()
    
    # Run Whisper transcription with appropriate precision
    if device == "cuda":
        with torch.cuda.amp.autocast():  # Enable automatic mixed precision
            result = model.transcribe(str(audio_path), language='en', fp16=True)
    else:
        result = model.transcribe(str(audio_path), language='en', fp16=False)
    
    # Process segments
    for segment in result["segments"]:
        words_with_timestamps.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        })
    
    transcribe_end = time.time()
    transcribe_duration = transcribe_end - transcribe_start
    print(f"Transcription processing took: {format_time(transcribe_duration)}")
    
    return words_with_timestamps

def save_transcription(words_with_timestamps, output_path):
    """Save transcription data to a JSON file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(words_with_timestamps, f, indent=2, ensure_ascii=False)
    print(f"Transcription saved to {output_path}")

def process_video(video_path, model_size="base", optimize=True):
    """Process video to create transcription"""
    process_start = time.time()
    
    # Check GPU availability
    device = check_gpu()
    
    video_file = Path(video_path)
    transcription_path = video_file.with_suffix('.transcription.json')
    
    print(f"Processing {video_file.name}...")
    
    # Extract audio using ffmpeg
    audio_path = extract_audio(video_path)
    
    print(f"Loading Whisper {model_size} model...")
    model = whisper.load_model(model_size)
    
    if device == "cuda":
        model = model.cuda()
    
    # Optimize model if requested
    if optimize:
        model = optimize_model(model, device)
    
    print("Generating transcription...")
    words_with_timestamps = transcribe_with_timestamps(model, audio_path, device)
    save_transcription(words_with_timestamps, transcription_path)
    
    # Clean up temporary audio file
    audio_path.unlink()
    
    process_end = time.time()
    total_duration = process_end - process_start
    print(f"Total processing time: {format_time(total_duration)}")
    print("Done!")

def main():
    parser = argparse.ArgumentParser(description='Transcribe video and save as JSON')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--model', default='base', 
                      choices=['tiny', 'base', 'small', 'medium', 'large'],
                      help='Whisper model size to use')
    parser.add_argument('--no-optimize', action='store_true',
                      help='Disable model optimization with torch.compile()')
    
    args = parser.parse_args()
    process_video(args.video_path, model_size=args.model, optimize=not args.no_optimize)

if __name__ == "__main__":
    main()
