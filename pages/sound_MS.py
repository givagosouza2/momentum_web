import streamlit as st
from scipy import signal
import pandas as pd
import numpy as np
import scipy
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft
import wave


def app():
    def calculate_fundamental_frequency(frequencies, fft_result, frame_rate):
        # Encontre o índice do pico dominante no espectro de frequência
        peak_index = np.argmax(np.abs(fft_result))

        # Calcule a frequência fundamental correspondente ao pico
        fundamental_frequency = frequencies[peak_index]

        return fundamental_frequency

    def read_wave_file(file_path):
        with wave.open(file_path, 'rb') as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            num_frames = wf.getnframes()
            data = wf.readframes(num_frames)

        audio_array = np.frombuffer(data, dtype=np.int16)

        return audio_array, channels, sample_width, frame_rate

    def convert_to_mono(audio_array, channels):
        if channels == 2:
            # Reshape the audio data to have two columns (channels)
            audio_array_reshaped = audio_array.reshape(-1, channels)

            # Take the mean along the second axis to get a single channel
            audio_mono = np.mean(audio_array_reshaped, axis=1).astype(np.int16)

            return audio_mono
        else:
            return audio_array

    def calculate_shimmer(audio_signal):
        # Calculate the absolute differences between consecutive samples
        differences = np.abs(np.diff(audio_signal))

        # Calculate the mean amplitude difference
        mean_difference = np.mean(differences)

        # Calculate the amplitude perturbation quotient (APQ)
        shimmer = (2 * mean_difference) / np.mean(np.abs(audio_signal))

        return shimmer

    def calculate_jitter(audio_signal, sampling_rate):
        # Calculate the differences between consecutive pitch periods
        period_diffs = np.diff(np.where(audio_signal > 0)[0])

        # Calculate the mean pitch period
        mean_period = np.mean(period_diffs) / sampling_rate

        # Calculate the frequency perturbation quotient (FPQ)
        jitter = (2 * np.mean(np.abs(period_diffs - mean_period))) / \
            np.mean(period_diffs)

        return jitter

    def calculate_frequencies(audio_mono, frame_rate):
        # Calculate the FFT of the audio signal
        fft_result = np.fft.fft(audio_mono)

        # Calculate the frequencies corresponding to the FFT result
        frequencies = np.fft.fftfreq(len(fft_result), d=1/frame_rate)

        return frequencies, np.abs(fft_result)

    st.markdown(
        '<h1 style="text-align: center; color: blue;">Análise de voz</h1>', unsafe_allow_html=True)

    # Primeiro parágrafo
    texto_markdown = """
        &nbsp;&nbsp;&nbsp;&nbsp; A avaliação da voz realizada aqui nesta página é feita através dos registros do áudio da letra A dita por 10 s em intensidade confortável pelo participante de acordo com Tai et al. (2021).
        <br>
        <br>
        <br>
        """
    st.markdown(
        f"<div style='text-align: justify;'>{texto_markdown}</div>", unsafe_allow_html=True)

    # Acceleration file upload button
    uploaded_sound = st.file_uploader(
        "Selecione o aquivo de áudio", type=["wav"],)

    # Check if a file has been uploaded
    if uploaded_sound is not None:
        # Read and display the data from the uploaded CSV file
        if uploaded_sound is not None:
            if uploaded_sound is not None:
                # Read and display the data from the uploaded WAV file
                st.audio(uploaded_sound, format='audio/wav')

                audio_bytes, channels, sample_width, frame_rate = read_wave_file(
                    uploaded_sound)
                # Convert to mono if there are multiple channels
                audio_mono = convert_to_mono(audio_bytes, channels)

                audio_array = np.frombuffer(audio_mono, dtype=np.int16)
                audio_array_sq = np.abs(audio_array)
                avg_audio_array = np.mean(audio_array_sq[5000:20000])
                std_audio_array = np.std(audio_array_sq[5000:20000])

                t = 0
                for i in audio_array_sq:
                    if i > avg_audio_array + 5*std_audio_array:
                        break
                    t = t + 1

                data = audio_array[t:-1]

                frequencies, fft_result = calculate_frequencies(
                    audio_mono, frame_rate)

                # Create time axis
                time_axis_total = np.linspace(
                    0, len(audio_array) / frame_rate, num=len(audio_array))
                time_axis = np.linspace(
                    0, len(data) / frame_rate, num=len(data))

                # Plot waveform

                plt.figure(figsize=(5, 3))
                plt.plot(time_axis_total, audio_array)
                plt.plot([time_axis_total[t], time_axis_total[t]],
                         [-30000, 30000], '--r')
                plt.xlabel('Tempo (seconds)')
                plt.ylabel('Amplitude')
                st.pyplot(plt)

                plt.figure(figsize=(5, 3))
                # Plot the frequency spectrum
                plt.plot(frequencies, fft_result)
                plt.xlabel("Frequência (Hz)")
                plt.ylabel("Amplitude")
                plt.xlim(0, 10000)
                st.pyplot(plt)

                # Plot spectrogram
                plt.figure(figsize=(5, 3))
                plt.specgram(data, Fs=frame_rate, cmap='viridis')
                plt.xlabel('Tempo (seconds)')
                plt.ylabel('Frequência (Hz)')

                st.pyplot(plt)

                shimmer_value = calculate_shimmer(audio_array)
                jitter_value = calculate_jitter(audio_array, frame_rate)
                fundamental_frequency_value = calculate_fundamental_frequency(
                    frequencies, fft_result, frame_rate)
                st.write("Shimmer = " + str(round(shimmer_value, 2)))
                st.write("Jitter = " + str(round(jitter_value, 2)))
                st.write("Frequência fundamental (Hz) = " +
                         str(round(fundamental_frequency_value, 2)))
