# Download youtube video as mp4
# Save to content/videoname.mp4
# !yt-dlp https://youtu.be/H0iHi7iFkiQ --format m4a -o "./content/%(id)s.%(ext)s"

# Use whisper model to transcribe
# !whisper "./content/H0iHi7iFkiQ.m4a" --model small --language English