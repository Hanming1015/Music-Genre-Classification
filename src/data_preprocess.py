import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Convert a .wav file to a mel-spec img
def convert_to_mel_spectrogram(audio_base_dir, output_base_dir, n_mels=128, sr=22050):
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    for genre in os.listdir(audio_base_dir):
        genre_dir = os.path.join(audio_base_dir, genre)
        if os.path.isdir(genre_dir):
            output_genre_dir = os.path.join(output_base_dir, genre)
            if not os.path.exists(output_genre_dir):
                os.makedirs(output_genre_dir)

            for filename in os.listdir(genre_dir):
                if filename.endswith('.wav'):
                    file_path = os.path.join(genre_dir, filename)
                    try:
                        y, sr = librosa.load(file_path, sr=sr)
                        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
                        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

                        plt.figure(figsize=(10, 4))
                        librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
                        plt.colorbar(format='%+2.0f dB')
                        plt.title(f'Mel Spectrogram of {filename}')
                        plt.tight_layout()

                        # Save the mel-spec as temp-img
                        temp_image_path = os.path.join(output_genre_dir, f"temp_{os.path.splitext(filename)[0]}.png")
                        plt.savefig(temp_image_path)
                        plt.close()

                        # Crop the temp-img
                        input_image = cv2.imread(temp_image_path)

                        # Save the cropped image to a temporary path
                        temp_cropped_path = os.path.join(output_genre_dir,
                                                         f"cropped_{os.path.splitext(filename)[0]}.png")
                        cv2.imwrite(temp_cropped_path, input_image)

                    except Exception as e:
                        print(f"Error processing {filename}: {e}")

convert_to_mel_spectrogram(audio_base_dir='../data/music_genre_dataset/genres_original',
                           output_base_dir='../data/music_genre_dataset/mel_spectrograms')


