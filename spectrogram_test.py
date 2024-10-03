import torch
import torchaudio
import matplotlib.pyplot as plt

# Sample rate and signal duration (in seconds)
sample_rate = 16000
duration = 5

# Generate a random signal (for demonstration purposes)
waveform = torch.randn(1, sample_rate * duration)  # 1 channel, 5-second signal

# Define the transformation: Spectrogram
transform = torchaudio.transforms.Spectrogram(
    n_fft=400,           # Number of FFT bins
    win_length=None,     # Window length for FFT (default is n_fft)
    hop_length=200,      # Hop length for the sliding window
    normalized=False     # If true, normalizes the spectrogram
)

# Apply the transformation
spectrogram = transform(waveform)

# Plot the spectrogram
plt.figure(figsize=(10, 5))
plt.imshow(spectrogram.log2()[0].numpy(), cmap='viridis', aspect='auto')
plt.title('Spectrogram')
plt.ylabel('Frequency bins')
plt.xlabel('Time frames')
plt.colorbar(format="%+2.0f dB")
# plt.show()
plt.savefig('spectrogram.png')
print()
