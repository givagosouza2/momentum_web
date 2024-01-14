import streamlit as st
from scipy import signal
import pandas as pd
import numpy as np
import scipy
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft


def app():
    def rms_amplitude(time_series):
        chunk = time_series
        # Square each value
        squared_values = [x**2 for x in chunk]
        # Calculate the mean of the squared values
        mean_squared = np.mean(squared_values)
        # Take the square root of the mean
        return mean_squared

    def approximate_entropy2(data, n, r):
        correlation = np.zeros(2)
        for m in range(n, n+2):  # Run it twice, with window size differing by 1
            set = 0
            count = 0
            counter = np.zeros(len(data) - m + 1)
            window_correlation = np.zeros(len(data) - m + 1)

            for i in range(0, len(data) - m + 1):
                # Current window stores the sequence to be compared with other sequences
                current_window = data[i:i + m]

                for j in range(0, len(data) - m + 1):
                    # Get a window for comparison with the current_window
                    sliding_window = data[j:j + m]

                    for k in range(m):
                        if (abs(current_window[k] - sliding_window[k]) > r) and set == 0:
                            set = 1  # The difference between the two sequences is greater than the given value

                    if set == 0:
                        count += 1  # Measure how many sliding_windows are similar to the current_window

                        set = 0  # Reset 'set'

                # Number of similar windows for every current_window
                counter[i] = count / (len(data) - m + 1)
                count = 0

            correlation[m - n] = np.sum(counter) / (len(data) - m + 1)

        apen = np.log(correlation[0] / correlation[1])
        return apen

    def tremor_fft(data):
        fs = 100  # Sampling frequency
        fft_results = []
        frequencies = []

        # Perform FFT
        fft_result = np.abs(fft(data))
        fft_result[0] = 0
        N = len(fft_result)
        freq = np.fft.fftfreq(N, 1/fs)

        pos = 0
        for i in freq:
            if i >= 0:
                pos = pos + 1
            else:
                f = pos
                break
        power_spectrum = fft_result[0:f]
        temp_freq = freq[0:f]

        return power_spectrum, temp_freq

    st.markdown(
        '<h1 style="text-align: center; color: blue;">Avaliação do tremor de mão por smartphone</h1>', unsafe_allow_html=True)

    # Primeiro parágrafo
    texto_markdown = """
        &nbsp;&nbsp;&nbsp;&nbsp; A avaliação do tremor de mão realizada aqui nesta página é feita através dos registros do acelerômetro durante uma tarefa de manutenção da mão em posição de repouso ou em imóvel em alguma postura ativa. Os sinais obtidos pelo giroscópio também podem ser usados no lugar dos dados acelerométricos, mas será preciso atentar que as unidades devem deixar de ser m/s^2 para rad/s. As análises aqui apresentadas são baseadas nas análises descritas em Santos et al. (2022). Os arquivos do Momentum Science são compatíveis com a presente rotina.
        <br>
        <br>
        <br>
        """
    st.markdown(
        f"<div style='text-align: justify;'>{texto_markdown}</div>", unsafe_allow_html=True)

    t1, t2, t3 = st.columns([1, 1, 0.7])

    # Create acceleration and gyroscope file upload buttons
    uploaded_acc_tremor = st.file_uploader(
        "Selecione o arquivo de texto do tremor", type=["txt"],)
    if uploaded_acc_tremor is not None:
        # Allocation of the acceleration data to the variables
        if uploaded_acc_tremor is not None:
            custom_separator = ';'
            df = pd.read_csv(uploaded_acc_tremor, sep=custom_separator)
            t = df.iloc[:, 0]
            x = df.iloc[:, 1]
            y = df.iloc[:, 2]
            z = df.iloc[:, 3]
            time = t

            # Pre-processing data: All channels were detrended, and interpolated to 100 Hz
            if np.max(x) > 9 or np.max(y) > 9 or np.max(z) > 9:
                x = signal.detrend(x)
                y = signal.detrend(y)
                z = signal.detrend(z)
            else:
                x = signal.detrend(x)
                y = signal.detrend(y)
                z = signal.detrend(z)
            interpf = scipy.interpolate.interp1d(time, x)
            time_ = np.arange(start=time[0], stop=time[len(time)-1], step=10)
            x_ = interpf(time_)
            t, x = time_/1000, x_
            interpf = scipy.interpolate.interp1d(time, y)
            time_ = np.arange(start=time[0], stop=time[len(time)-1], step=10)
            y_ = interpf(time_)
            t, y = time_/1000, y_
            interpf = scipy.interpolate.interp1d(time, z)
            time_ = np.arange(start=time[0], stop=time[len(time)-1], step=10)
            z_ = interpf(time_)
            t, z = time_/1000, z_

            p_spectrum_x_1, t_freq_x_1 = tremor_fft(x[0:499])
            p_spectrum_y_1, t_freq_y_1 = tremor_fft(y[0:499])
            p_spectrum_z_1, t_freq_z_1 = tremor_fft(z[0:499])
            p_spectrum_1_norm = np.sqrt(
                p_spectrum_x_1**2+p_spectrum_y_1**2+p_spectrum_z_1**2)

            max_sweep_1 = np.max([x[0:499], y[0:499], z[0:499]])
            min_sweep_1 = np.min([x[0:499], y[0:499], z[0:499]])
            max_power_1 = np.max(
                [p_spectrum_x_1, p_spectrum_y_1, p_spectrum_z_1])

            p_spectrum_x_2, t_freq_x_2 = tremor_fft(x[500:999])
            p_spectrum_y_2, t_freq_y_2 = tremor_fft(y[500:999])
            p_spectrum_z_2, t_freq_z_2 = tremor_fft(z[500:999])
            p_spectrum_2_norm = np.sqrt(
                p_spectrum_x_2**2+p_spectrum_y_2**2+p_spectrum_z_2**2)

            max_sweep_2 = np.max([x[500:999], y[500:999], z[500:999]])
            min_sweep_2 = np.min([x[500:999], y[500:999], z[500:999]])
            max_power_2 = np.max(
                [p_spectrum_x_2, p_spectrum_y_2, p_spectrum_z_2])

            p_spectrum_x_3, t_freq_x_3 = tremor_fft(x[1000:1499])
            p_spectrum_y_3, t_freq_y_3 = tremor_fft(y[1000:1499])
            p_spectrum_z_3, t_freq_z_3 = tremor_fft(z[1000:1499])
            p_spectrum_3_norm = np.sqrt(
                p_spectrum_x_3**2+p_spectrum_y_3**2+p_spectrum_z_3**2)

            max_sweep_3 = np.max([x[1000:1499], y[1000:1499], z[1000:1499]])
            min_sweep_3 = np.min([x[1000:1499], y[1000:1499], z[1000:1499]])
            max_power_3 = np.max(
                [p_spectrum_x_3, p_spectrum_y_3, p_spectrum_z_3])

            p_spectrum_x_4, t_freq_x_4 = tremor_fft(x[1500:1999])
            p_spectrum_y_4, t_freq_y_4 = tremor_fft(y[1500:1999])
            p_spectrum_z_4, t_freq_z_4 = tremor_fft(z[1500:1999])
            p_spectrum_4_norm = np.sqrt(
                p_spectrum_x_4**2+p_spectrum_y_4**2+p_spectrum_z_4**2)

            max_sweep_4 = np.max([x[1500:1999], y[1500:1999], z[1500:1999]])
            min_sweep_4 = np.min([x[1500:1999], y[1500:1999], z[1500:1999]])
            max_power_4 = np.max(
                [p_spectrum_x_4, p_spectrum_y_4, p_spectrum_z_4])

            p_spectrum_x_5, t_freq_x_5 = tremor_fft(x[2000:2499])
            p_spectrum_y_5, t_freq_y_5 = tremor_fft(y[2000:2499])
            p_spectrum_z_5, t_freq_z_5 = tremor_fft(z[2000:2499])
            p_spectrum_5_norm = np.sqrt(
                p_spectrum_x_5**2+p_spectrum_y_5**2+p_spectrum_z_5**2)

            max_sweep_5 = np.max([x[2000:2499], y[2000:2499], z[2000:2499]])
            min_sweep_5 = np.min([x[2000:2499], y[2000:2499], z[2000:2499]])
            max_power_5 = np.max(
                [p_spectrum_x_5, p_spectrum_y_5, p_spectrum_z_5])
            p_spectrum = np.mean([p_spectrum_1_norm, p_spectrum_2_norm,
                                 p_spectrum_3_norm, p_spectrum_4_norm, p_spectrum_5_norm], axis=0)

            f = 0
            for i in t_freq_x_1:
                if i > 4:
                    break
                f = f + 1
            h = 0
            for i in t_freq_x_1:
                if i > 14:
                    break
                h = h + 1
            c = 0
            peak_spectrum = np.max(p_spectrum[f:len(p_spectrum)-1])
            for i in p_spectrum:
                if i == peak_spectrum:
                    peak_freq = t_freq_x_1[c]
                    break
                c = c + 1

            total_power = np.sum(p_spectrum[f:h])
            for i in range(h-f):
                if np.sum(p_spectrum[f:f+i]) >= total_power/2:
                    print(i)
                    m_freq = t_freq_x_1[f+i]
                    break

            rms_sweep_1_x = rms_amplitude(x[0:499])
            rms_sweep_2_x = rms_amplitude(x[500:999])
            rms_sweep_3_x = rms_amplitude(x[1000:1499])
            rms_sweep_4_x = rms_amplitude(x[1500:1999])
            rms_sweep_5_x = rms_amplitude(x[2000:2499])
            rms_x = np.mean([rms_sweep_1_x, rms_sweep_2_x,
                            rms_sweep_3_x, rms_sweep_4_x, rms_sweep_5_x])

            rms_sweep_1_y = rms_amplitude(y[0:499])
            rms_sweep_2_y = rms_amplitude(y[500:999])
            rms_sweep_3_y = rms_amplitude(y[1000:1499])
            rms_sweep_4_y = rms_amplitude(y[1500:1999])
            rms_sweep_5_y = rms_amplitude(y[2000:2499])
            rms_y = np.mean([rms_sweep_1_y, rms_sweep_2_y,
                            rms_sweep_3_y, rms_sweep_4_y, rms_sweep_5_y])

            rms_sweep_1_z = rms_amplitude(z[0:499])
            rms_sweep_2_z = rms_amplitude(z[500:999])
            rms_sweep_3_z = rms_amplitude(z[1000:1499])
            rms_sweep_4_z = rms_amplitude(z[1500:1999])
            rms_sweep_5_z = rms_amplitude(z[2000:2499])
            rms_z = np.mean([rms_sweep_1_z, rms_sweep_2_z,
                            rms_sweep_3_z, rms_sweep_4_z, rms_sweep_5_z])

            apEn_sweep_1_x = approximate_entropy2(x[0:499], 2, 0.2)
            apEn_sweep_2_x = approximate_entropy2(x[500:999], 2, 0.2)
            apEn_sweep_3_x = approximate_entropy2(x[1000:1499], 2, 0.2)
            apEn_sweep_4_x = approximate_entropy2(x[1500:1999], 2, 0.2)
            apEn_sweep_5_x = approximate_entropy2(x[2000:2499], 2, 0.2)
            apEn_x = np.mean([apEn_sweep_1_x, apEn_sweep_2_x,
                              apEn_sweep_3_x, apEn_sweep_4_x, apEn_sweep_5_x])

            apEn_sweep_1_y = approximate_entropy2(y[0:499], 2, 0.2)
            apEn_sweep_2_y = approximate_entropy2(y[500:999], 2, 0.2)
            apEn_sweep_3_y = approximate_entropy2(y[1000:1499], 2, 0.2)
            apEn_sweep_4_y = approximate_entropy2(y[1500:1999], 2, 0.2)
            apEn_sweep_5_y = approximate_entropy2(y[2000:2499], 2, 0.2)
            apEn_y = np.mean([apEn_sweep_1_y, apEn_sweep_2_y,
                              apEn_sweep_3_y, apEn_sweep_4_y, apEn_sweep_5_y])

            apEn_sweep_1_z = approximate_entropy2(z[0:499], 2, 0.2)
            apEn_sweep_2_z = approximate_entropy2(z[500:999], 2, 0.2)
            apEn_sweep_3_z = approximate_entropy2(z[1000:1499], 2, 0.2)
            apEn_sweep_4_z = approximate_entropy2(z[1500:1999], 2, 0.2)
            apEn_sweep_5_z = approximate_entropy2(z[2000:2499], 2, 0.2)
            apEn_z = np.mean([apEn_sweep_1_z, apEn_sweep_2_z,
                              apEn_sweep_3_z, apEn_sweep_4_z, apEn_sweep_5_z])

            with t1:
                plt.figure(figsize=(5, 5))
                plt.plot(t, x, 'r')
                plt.plot(t, y, 'g')
                plt.plot(t, z, 'b')
                plt.xlabel("Tempo (s)")
                plt.ylabel("Aceleração (g)")
                plt.ylim(min_sweep_1, max_sweep_1)
                st.pyplot(plt)

            with t2:
                # avg
                plt.figure(figsize=(5, 5))
                plt.plot(t_freq_z_5, p_spectrum, 'k')
                plt.plot(peak_freq, peak_spectrum, marker='o', markersize=12,
                         markerfacecolor='none', markeredgecolor='r')
                plt.plot([m_freq, m_freq], [0, np.max(p_spectrum)*1.5], '--b')
                plt.xlabel("Frequência temporal (Hz)")
                plt.ylabel("Magnitude da aceleração (g)")
                plt.xlim(0, 14)
                plt.ylim(0, max_power_5)
                st.pyplot(plt)
            with t3:
                st.markdown(
                    '<h3 style="text-align: left; color: blue;">Resultados</h3>', unsafe_allow_html=True)
                st.text('Amplitude rms X (g) = ' + str(round(rms_x, 4)))
                st.text('Amplitude rms Y (g) = ' + str(round(rms_y, 4)))
                st.text('Amplitude rms Z (g) = ' + str(round(rms_z, 4)))
                st.text('Entropia aproximada X = ' + str(round(apEn_x, 3)))
                st.text('Entropia aproximada Y = ' + str(round(apEn_y, 3)))
                st.text('Entropia aproximada Z = ' + str(round(apEn_z, 3)))
                st.text('Amplitude de pico (g) = ' +
                        str(round(peak_spectrum, 3)))
                st.text('Frequência de pico (Hz) = ' +
                        str(round(peak_freq, 3)))
                st.text('Frequência mediana (Hz) = ' + str(round(m_freq, 3)))
