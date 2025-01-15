
# Transcribe Feat OpenAI || Google SR

This repository contains two different implementations for automatically adding subtitles to videos using speech recognition. Both tools will transcribe speech and create beautiful, customizable subtitles!

## 🛠️ Available Scripts
- `feat_transcribe_v2.py` - Uses Google Speech Recognition API  
- `feat_trans_whisper.py` - Uses OpenAI Whisper

## ✨ Features

- 🎯 Choice of speech recognition engines:
  - Google Speech Recognition API (`feat_transcribe_v2.py`)
  - OpenAI Whisper (`feat_trans_whisper.py`)
- 🎨 Customizable subtitle appearance (font, size, color, outline)
- ⚡ Smart text wrapping and positioning
- 🔄 Timestamp synchronization
- 📱 Support for multiple video formats
- 💾 Save and reuse transcriptions

## 🚀 Installation

1. First, make sure you have Python 3.7+ installed on your system.

2. Install the required packages based on which version you want to use:

For Google Speech Recognition version (`feat_transcribe_v2.py`):
```bash
pip install moviepy SpeechRecognition Pillow numpy
```

For OpenAI Whisper version (`feat_trans_whisper.py`):
```bash
pip install moviepy SpeechRecognition Pillow numpy openai-whisper torch
```

3. Additional system requirements:
   - FFmpeg (required by moviepy)
   - A working internet connection (for Google Speech Recognition)

## 💻 Usage

### Initial Transcription
When using either script for the first time with a video, you must use the `--generate-transcription` flag. This will create a JSON file containing the transcription data:

```bash
# For Google Speech Recognition version
python feat_transcribe_v2.py myvideo.mp4 --generate-transcription

# For OpenAI Whisper version
python feat_trans_whisper.py myvideo.mp4 --generate-transcription
```

The script will create a `myvideo.transcription.json` file alongside your video file.

### Reusing Existing Transcription
After generating the transcription file, you can modify subtitle styles without re-transcribing by omitting the `--generate-transcription` flag:

```bash
python feat_transcribe_v2.py myvideo.mp4
# or
python feat_trans_whisper.py myvideo.mp4
```

### 🎮 Available Options

- `video_path`: Path to your input video file
- `--generate-transcription`: Generate new transcription (omit to use existing transcription file)
- `--output`: Custom output path (optional)
- `--font-path`: Path to TTF font file (default: Arial)
- `--font-size`: Font size in pixels (default: 30)
- `--font-color`: Text color (hex code or name) (default: white)
- `--outline-color`: Outline color (hex code or name) (default: black)
- `--outline-width`: Width of text outline in pixels (default: 2)
- `--line-spacing`: Spacing between lines in pixels (default: 4)
- `--bottom-padding`: Padding from bottom of screen in pixels (default: 50)
- `--width-percent`: Width of text box as percentage of video width (0.0-1.0) (default: 0.8)

### 🎨 Supported Colors

You can use either:
- Color names: `red`, `white`, `black`, `yellow`, `blue`, `green`
- Hex codes: `#FF0000`, `#FFFFFF`, etc.

## 🤔 Choosing Between Versions

- `feat_transcribe_v2.py` (Google SR):
  - Requires internet connection
  - Good for general-purpose transcription
  - Faster processing for shorter videos
  - Free to use (with API limits)

- `feat_trans_whisper.py` (OpenAI Whisper):
  - Can work offline
  - Better accuracy for various accents and languages
  - More resource-intensive
  - Better for longer videos
  - Open source and free to use

## 📝 Example Commands

1. Basic usage with default settings:
```bash
python feat_trans_whisper.py myvideo.mp4 --generate-transcription
```

2. Custom font and colors:
```bash
python feat_transcribe_v2.py myvideo.mp4 --font-size 45 --font-color yellow --outline-color black --outline-width 3
```

3. Custom output path:
```bash
python feat_trans_whisper.py input.mp4 --output subtitled_video.mp4
```

4. Adjust subtitle positioning:
```bash
python feat_transcribe_v2.py video.mp4 --bottom-padding 70 --width-percent 0.7
```

## ⚠️ Important Notes

- The Google SR version requires an internet connection for speech recognition
- Processing time depends on video length and system performance
- Font paths may need adjustment based on your operating system
- Make sure you have sufficient disk space for temporary files
- The transcription JSON file can be edited manually if you need to adjust timings or text

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is open source and available under the MIT License.
