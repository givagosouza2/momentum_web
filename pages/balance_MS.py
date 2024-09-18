import streamlit as st
from scipy import signal
import pandas as pd
import numpy as np
import scipy
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


def app():
    def butterworth_filter(data, cutoff, fs, order=4, btype='low'):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype, analog=False)
        y = filtfilt(b, a, data)
        return y

    def balance_fft(data):
        fs = 100  # Sampling frequency
        # Perform FFT
        fft_result = np.fft.fft(data)
        N = len(fft_result)

        frequencies = np.fft.fftfreq(N, 1/fs)
        spectrum_amplitude = []
        spectrum_amplitude = (np.abs(fft_result/N))
        spectrum_amplitude[0] = 0
        freq = []
        spectra = []
        a = 0
        c = 0
        for i in frequencies:
            if i >= 0:
                freq.append(i)
                spectra.append(spectrum_amplitude[c])
                a = a + 1
            c = c + 1

        a = 0
        for i in freq:
            a = a + 1
            if i > 0.5:
                f1 = a
                break
        a = 0
        for i in freq:
            a = a + 1
            if i > 2:
                f2 = a
                break
        a = 0
        for i in freq:
            a = a + 1
            if i > 6:
                f3 = a
                break

        total_spectral_energy = sum(spectra[0:f3])
        energy = 0

        c = 1
        while energy < total_spectral_energy/2:
            energy = np.sum(spectra[0:c])
            c = c + 1
        median_frequency = freq[c]
        LF_energy = sum(spectra[0:f1])
        MF_energy = sum(spectra[f1:f2])
        HF_energy = sum(spectra[f2:f3])

        return freq, spectra, median_frequency, LF_energy, MF_energy, HF_energy

    def set_ellipse(fpML, fpAP):
        points = np.column_stack((fpML, fpAP))
        hull = ConvexHull(points)

        # Get the boundary points of the convex hull
        boundary_points = points[hull.vertices]

        # Calculate the centroid of the boundary points
        centroidx = np.mean(fpML)
        centroidy = np.mean(fpAP)
        centroid = centroidx, centroidy

        # Calculate the covariance matrix of the boundary points
        covariance = np.cov(boundary_points, rowvar=False)

        # Calculate the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(covariance)

        # Calculate the major and minor axis of the ellipse
        major_axis = np.sqrt(eigenvalues[0]) * np.sqrt(-2 * np.log(1 - 0.95))/2
        minor_axis = np.sqrt(eigenvalues[1]) * np.sqrt(-2 * np.log(1 - 0.95))/2

        # Calculate the angle of the ellipse
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        area = np.pi*major_axis*minor_axis
        num_points = 101  # 360/100 + 1
        ellipse_points = np.zeros((num_points, 2))
        a = 0
        for i in np.arange(0, 361, 360 / 100):
            ellipse_points[a, 0] = centroid[0] + \
                major_axis * np.cos(np.radians(i))
            ellipse_points[a, 1] = centroid[1] + \
                minor_axis * np.sin(np.radians(i))
            a += 1
        angle_deg = -angle
        angle_rad = np.radians(angle_deg)

        # Matrix for ellipse rotation
        R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                      [np.sin(angle_rad), np.cos(angle_rad)]])
        ellipse_points = np.dot(ellipse_points, R)
        return ellipse_points, area, angle_deg, major_axis, minor_axis

    st.markdown("<h1 style='text-align: center; color: blue;'>Avaliação do equilíbrio estático</h1>",
                unsafe_allow_html=True)
    # Primeiro parágrafo
    texto_markdown = """
        &nbsp;&nbsp;&nbsp;&nbsp; A avaliação do equilíbrio realizada nesta página é a avaliação do equilíbrio estático na qual são usados os dados acelerométricos dos eixos antero-posterior e médio-lateral. O método usado para a obtenção dos registros são os mesmos usados em Rodrigues et al. (2022) e em Correa et al. (2023). A posição do smartphone pode ser tanto em pé quanto deitado, pois o código indentifica automaticamente o eixo superior-inferior e o exclui da análise. Aqui são realizadas análise no domínio do tempo e das frequências temporais. Os arquivos do Momentum Science são compatíveis com a presente rotina.
        <br>
        <br>
        <br>
        """
    st.markdown(
        f"<div style='text-align: justify;'>{texto_markdown}</div>", unsafe_allow_html=True)

    t1, t2, t3 = st.columns([1, 1, 1])

    # Acceleration file upload button
    uploaded_acc = st.file_uploader(
        "Selecione o aquivo de texto do acelerômetro", type=["txt"],)

    # Check if a file has been uploaded
    if uploaded_acc is not None:
        # Read and display the data from the uploaded CSV file
        if uploaded_acc is not None:
            custom_separator = ';'

            # Allocation of the data to the variables
            df = pd.read_csv(uploaded_acc, sep=custom_separator)
            t = df.iloc[:, 0]
            x = df.iloc[:, 1]
            y = df.iloc[:, 2]
            z = df.iloc[:, 3]
            time = t
            AP = z

            # Selection of ML axis between x and y axis. The one that recorded the gravity acceleration is excluded and the other is the ML axis
            if np.mean(x) > np.mean(y):
                ML = y
            else:
                ML = x
            AP = signal.detrend(AP)
            ML = signal.detrend(ML)

            # Pre-processing data: interpolating to 100 Hz
            interpf = scipy.interpolate.interp1d(time, AP)
            time_ = np.arange(start=time[0], stop=time[len(time)-1], step=10)
            AP_ = interpf(time_)
            xAP, yAP = time_/1000, AP_
            interpf = scipy.interpolate.interp1d(time, ML)
            time_ = np.arange(start=time[0], stop=time[len(time)-1], step=10)
            ML_ = interpf(time_)
            xML, yML = time_/1000, ML_
            # norm calculation
            norm = np.sqrt(yAP**2+yML**2)
            length_balance = len(yML)

            # Creating controls to interacts with the plots
            with t3:
                slider_min = st.number_input(
                    "Selecione o início do registro (número de pontos no registro)", min_value=1, max_value=length_balance-1, step=1, value=1)
                slider_max = st.number_input(
                    "Selecione o final do registro (número de pontos no registro)", min_value=1, max_value=length_balance-1, value=length_balance-1, step=1)
                initial_state = True
                checkbox_1 = st.checkbox(
                    "Mostrar o registro completo", value=initial_state)
                checkbox_2 = st.checkbox(
                    "Mostrar o período de análise do registro", value=initial_state)

                # Ellipse fitting and features extraction from ellipse
            ellipse_fit, area_value, angle_deg_value, major_axis_value, minor_axis_value = set_ellipse(
                yML[slider_min:slider_max], yAP[slider_min:slider_max])

            # Extracting features: total deviation, rmsAP, rmsML
            total_deviation = sum(np.sqrt(norm[slider_min:slider_max]))
            rmsAP = np.sqrt(np.mean(np.square(yAP[slider_min:slider_max])))
            rmsML = np.sqrt(np.mean(np.square(yML[slider_min:slider_max])))

            frequencies, spectrum_amplitude_ML, median_frequency_ML, LF_energy_ML, MF_energy_ML, HF_energy_ML = balance_fft(
                yML[slider_min:slider_max])
            frequencies, spectrum_amplitude_AP, median_frequency_AP, LF_energy_AP, MF_energy_AP, HF_energy_AP = balance_fft(
                yAP[slider_min:slider_max])

            maxX = np.max(ellipse_fit[:, 0])
            maxY = np.max(ellipse_fit[:, 1])
            maxValue = np.max([maxX, maxY])
            if maxValue <= 0.1:
                lim = 0.1
            elif maxValue > 0.1 and maxValue < 0.3:
                lim = 0.3
            elif maxValue > 0.3 and maxValue < 0.5:
                lim = 0.5
            elif maxValue > 0.5 and maxValue < 1:
                lim = 2
            else:
                lim = 5

            # Plotting statokinesiogram
            with t1:
                plt.figure(figsize=(5, 5))
                if checkbox_1 == True:
                    plt.plot(yML, yAP, 'grey')
                if checkbox_2 == True:
                    plt.plot(yML[slider_min:slider_max],
                             yAP[slider_min:slider_max], 'k')
                plt.plot(ellipse_fit[:, 0], ellipse_fit[:, 1], 'r')
                plt.fill(ellipse_fit[:, 0], ellipse_fit[:,
                         1], color='tomato', alpha=0.5)
                plt.xlabel('Aceleração ML (g)')
                plt.ylabel('Aceleração AP (g)')
                plt.ylim(-lim, lim)
                plt.xlim(-lim, lim)
                st.pyplot(plt)

                plt.figure(figsize=(5, 5))
                plt.plot(frequencies, spectrum_amplitude_AP, 'k')
                plt.xlabel('Frequência Temporal (Hz)')
                plt.ylabel('Magnitude de aceleração AP (g)')
                plt.xlim(0, 6)
                plt.ylim(0, lim/10)
                st.pyplot(plt)

                # Stabilograms plot
            with t2:
                plt.figure(figsize=(5, 1.9))
                plt.rcParams.update({'font.size': 12})
                if checkbox_1 == True:
                    plt.plot(xAP, yAP, 'grey')
                if checkbox_2 == True:
                    plt.plot(xAP[slider_min:slider_max],
                             yAP[slider_min:slider_max], 'k')
                plt.xlabel('Tempo (s)')
                plt.ylabel('Aceleração AP (g)')
                plt.ylim(-lim, lim)
                st.pyplot(plt)
                plt.figure(figsize=(5, 1.9))
                if checkbox_1 == True:
                    plt.plot(xML, yML, 'grey')
                if checkbox_2 == True:
                    plt.plot(xML[slider_min:slider_max],
                             yML[slider_min:slider_max], 'k')
                plt.xlabel('Tempo (s)')
                plt.ylabel('Aceleração ML (g)')
                plt.ylim(-lim, lim)
                st.pyplot(plt)
                plt.figure(figsize=(5, 5))
                plt.plot(frequencies, spectrum_amplitude_ML, 'k')
                plt.xlabel('Frequência Temporal (Hz)')
                plt.ylabel('Energia da aceleração ML (g^2)')
                plt.xlim(0, 6)
                plt.ylim(0, lim/10)
                st.pyplot(plt)

                # Printing of the features values
            with t3:
                st.markdown(
                    '<h3 style="text-align: left; color: blue;">Resultados</h3>', unsafe_allow_html=True)
                st.text('RMS AP (g) = ' + str(round(rmsAP, 5)))
                st.text('RMS ML (g) = ' + str(round(rmsML, 5)))
                st.text('Desvio total (g) = ' +
                        str(round(total_deviation, 3)))
                st.text('Área (g^2) = ' + str(round(area_value, 5)))
                st.text('Eixo maior (g) = ' + str(round(major_axis_value, 5)))
                st.text('Eixo menor (g) = ' + str(round(minor_axis_value, 5)))
                st.text('Ângulo de rotação (graus) = ' +
                        str(round(angle_deg_value, 2)))
                st.text('Frequência mediana AP (Hz) = ' +
                        str(round(median_frequency_AP, 2)))
                st.text('Frequência mediana ML (Hz) = ' +
                        str(round(median_frequency_ML, 2)))
                st.text('Energia das frequências baixas AP (g^2) = ' +
                        str(round(LF_energy_AP, 2)))
                st.text('Energia das frequências médias AP (g^2) = ' +
                        str(round(MF_energy_AP, 2)))
                st.text('Energia das frequências altas AP (g^2) = ' +
                        str(round(HF_energy_AP, 2)))
                st.text('Energia das frequências baixas ML (g^2) = ' +
                        str(round(LF_energy_ML, 2)))
                st.text('Energia das frequências médias ML (g^2) = ' +
                        str(round(MF_energy_ML, 2)))
                st.text('Energia das frequências altas ML (g^2) = ' +
                        str(round(HF_energy_ML, 2)))
