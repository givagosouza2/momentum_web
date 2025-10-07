import streamlit as st
from scipy import signal
import pandas as pd
import numpy as np
import scipy
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def create_results_content(file_name, values_dict):
    """Create result content as a string"""
    content = f"{file_name}\n\n"
    content += '\n'.join(f"{key}: {value}" for key, value in values_dict.items())
    return content


def app():
    def butterworth_filter(data, cutoff, fs, order=4, btype='low'):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype, analog=False)
        y = filtfilt(b, a, data)
        return y

    st.markdown(
        '<h1 style="text-align: center; color: blue;">Timed Up and Go test instrumentado</h1>', unsafe_allow_html=True)

    # Primeiro parágrafo
    texto_markdown = """
        &nbsp;&nbsp;&nbsp;&nbsp; A avaliação da mobilidade realizada aqui nesta página é feita através dos registros inerciais obtidos durante o teste Timed Up and Go. Para esta análise são necessárias as séries temporais do acelerômetro e do giroscópio e a forma de onda analisada é a norma das séries temporais dos 3 eixos de cada sensor como feito no estudo de Correa et al. (2023). O uso da norma como forma de onda a ser analisada, dentre outras vantagens, permite que o smartphone seja colocado em qualquer orientação na região lombar baixa do participante a ser testado. Aqui são realizadas análise no domínio do tempo e das frequências temporais. Os arquivos do Momentum Science são compatíveis com a presente rotina.
        <br>
        <br>
        <br>
        """
    st.markdown(
        f"<div style='text-align: justify;'>{texto_markdown}</div>", unsafe_allow_html=True)

    t1, t2, t3 = st.columns([0.75, 1.75, 1])

    # Create acceleration and gyroscope file upload buttons
    uploaded_acc_iTUG = st.file_uploader(
        "Carregue o arquivo de texto do acelerômetro", type=["txt"],)
    if uploaded_acc_iTUG is not None:
        # Allocation of the acceleration data to the variables
        if uploaded_acc_iTUG is not None:
            name_file = uploaded_acc_iTUG.name
            name_file = name_file[:-4]
            custom_separator = ';'
            df = pd.read_csv(uploaded_acc_iTUG, sep=custom_separator)
            t = df.iloc[:, 0]
            x = df.iloc[:, 1]
            y = df.iloc[:, 2]
            z = df.iloc[:, 3]
            time = t

            # Pre-processing data: All channels were detrended, normalized to gravity acceleration, and interpolated to 100 Hz
            if np.max(x) > 9 or np.max(y) > 9 or np.max(z) > 9:
                x = signal.detrend(x/9.81)
                y = signal.detrend(y/9.81)
                z = signal.detrend(z/9.81)
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

            # filtering acceleration data
            x = butterworth_filter(x, 4, 100, order=2, btype='low')
            y = butterworth_filter(y, 4, 100, order=2, btype='low')
            z = butterworth_filter(z, 4, 100, order=2, btype='low')

            # Calculating acceleration data norm
            norm_waveform = np.sqrt(x**2+y**2+z**2)

            uploaded_gyro_iTUG = st.file_uploader(
                "Carregue o arquivo de texto do giroscópio", type=["txt"],)
            # Allocation of the gyroscope data to the variables
        if uploaded_gyro_iTUG is not None:
            custom_separator = ';'
            df_gyro = pd.read_csv(uploaded_gyro_iTUG, sep=custom_separator)
            t_gyro = df_gyro.iloc[:, 0]
            x_gyro = df_gyro.iloc[:, 1]
            y_gyro = df_gyro.iloc[:, 2]
            z_gyro = df_gyro.iloc[:, 3]
            time_gyro = t_gyro

            # Pre-processing data: All channels were detrended, and interpolated to 100 Hz
            x_gyro = signal.detrend(x_gyro)
            y_gyro = signal.detrend(y_gyro)
            z_gyro = signal.detrend(z_gyro)
            interpf = scipy.interpolate.interp1d(time_gyro, x_gyro)
            time_gyro_ = np.arange(
                start=time_gyro[0], stop=time_gyro[len(time_gyro)-1], step=10)
            x_gyro_ = interpf(time_gyro_)
            t_gyro, x_gyro = time_gyro_/1000, x_gyro_
            interpf = scipy.interpolate.interp1d(time_gyro, y_gyro)
            time_gyro_ = np.arange(
                start=time_gyro[0], stop=time_gyro[len(time_gyro)-1], step=10)
            y_gyro_ = interpf(time_gyro_)
            t_gyro, y_gyro = time_gyro_/1000, y_gyro_
            interpf = scipy.interpolate.interp1d(time_gyro, z_gyro)
            time_gyro_ = np.arange(
                start=time_gyro[0], stop=time_gyro[len(time_gyro)-1], step=10)
            z_gyro_ = interpf(time_gyro_)
            t_gyro, z_gyro = time_gyro_/1000, z_gyro_

            # Filtering gyroscope data
            x_gyro = butterworth_filter(x_gyro, 1.5, 100, order=2, btype='low')
            y_gyro = butterworth_filter(y_gyro, 1.5, 100, order=2, btype='low')
            z_gyro = butterworth_filter(z_gyro, 1.5, 100, order=2, btype='low')

            # Calculating norm for angular velocity
            norm_waveform_gyro = np.sqrt(x_gyro**2+y_gyro**2+z_gyro**2)

            # Creating controls to interacts with the plots
            with t1:
                # Create slider widgets to set limits of baselines for the test onset and test offset
                st.markdown(
                    '<h3 style="text-align: left; color: blue;">Ajustar as baselines</h3>', unsafe_allow_html=True)
                length = len(norm_waveform_gyro)
                slider_baseline1 = st.number_input(
                    "Selecione o início do baseline do ONSET", min_value=1, max_value=length, step=1, value=200)
                slider_baseline2 = st.number_input(
                    "Selecione o final do baseline do ONSET", min_value=1, max_value=length, value=350, step=1)
                slider_baseline3 = st.number_input(
                    "Selecione o início do baseline do OFFSET", min_value=1, max_value=length, value=length-150, step=1)
                slider_baseline4 = st.number_input(
                    "Selecione o final do baseline do OFFSET", min_value=1, max_value=length, value=length-100, step=1)

                # We used the data from gyroscope to find the onset of the sitting to standing transition following Van Lummel et al. (2013) recommendation. We calculated the first derivative of the norm because we observed that it has less variability than the norm and it would facilitate to find the moment of the deflection from sitting to standing position. The basic idea to find the onset of the task was to choose a period during the pre-standing as baseline. Then, we calculated the average and standard deviation of the baseline from gyroscope first derivative vector and searched for the moment that the vector value exceeded the mean plus 4*standard deviation
                firstDerivative = np.diff(norm_waveform_gyro)

                # Setting the limits of the baseline to find the task onset. It is selected a range in the beginning of the recording
                if slider_baseline1 < slider_baseline2:
                    avg_firstderivative = np.mean(
                        firstDerivative[slider_baseline1:slider_baseline2])
                    std_firstderivative = np.std(
                        firstDerivative[slider_baseline1:slider_baseline2])
                    loc_onset = slider_baseline2
                    # Finding the task onset
                for i in firstDerivative[slider_baseline2:length-slider_baseline2]:
                    if i < avg_firstderivative + 4 * std_firstderivative:
                        loc_onset = loc_onset + 1
                    else:
                        break

                # Setting the limits of the baseline to find the task onset. It is selected a range in the end of the recording.
                if slider_baseline3 < slider_baseline4:
                    avg_firstderivative_offset = np.mean(
                        firstDerivative[slider_baseline3:slider_baseline4])
                    std_firstderivative_offset = np.std(
                        firstDerivative[slider_baseline3:slider_baseline4])
                    loc_offset = slider_baseline3

                # Finding the task offset
                for i in reversed(firstDerivative[1:slider_baseline3]):
                    if i < avg_firstderivative_offset + 4*std_firstderivative_offset:
                        loc_offset = loc_offset - 1
                    else:
                        break

                # Setting the sliders with onset and offset positions
                st.markdown(
                    '<h3 style="text-align: left; color: blue;">Ajustes manuais</h3>', unsafe_allow_html=True)
                slider_onset = st.number_input(
                    "Ajuste o momento do ONSET", min_value=1, max_value=length, value=loc_onset, step=1)
                slider_offset = st.number_input(
                    "Ajuste o momento do OFFSET", min_value=1, max_value=length-1, value=loc_offset, step=1)

                # Next step is to find the angular velocity peak during the return turn and pre-sitting turn. For that, we found the amplitude peaks and the position of the two largest amplitudes in the gyroscope norm. To indicate which component is each amplitude we compared the location in the vector. The earlier is from the return turn and the later is from the pre-sitting turn
                peaks, _ = find_peaks(norm_waveform_gyro, height=0.5)
                amplitude = norm_waveform_gyro[peaks]
                amplitude = sorted(amplitude, reverse=True)
                a = 0
                for i in norm_waveform_gyro:
                    a = a + 1
                    if i == amplitude[0]:
                        loc1 = a
                        latency1 = t_gyro[a]
                        amplitude1 = norm_waveform_gyro[a]
                        break
                a = 0
                for i in norm_waveform_gyro:
                    a = a + 1
                    if i == amplitude[1]:
                        loc2 = a
                        latency2 = t_gyro[a]
                        amplitude2 = norm_waveform_gyro[a]
                        break
                if latency1 > latency2:
                    g1_latency = latency2
                    g1_amplitude = amplitude2
                    loc_g1 = loc2
                    g2_latency = latency1
                    g2_amplitude = amplitude1
                    loc_g2 = loc1
                else:
                    g1_latency = latency1
                    g1_amplitude = amplitude1
                    loc_g1 = loc1
                    g2_latency = latency2
                    g2_amplitude = amplitude2
                    loc_g2 = loc2

                # Setting the sliders with angular velocity peak positions
                slider_G1 = st.number_input(
                    "Selecionar o pico de G1", min_value=1, max_value=length-1, value=loc_g1, step=1)
                slider_G2 = st.number_input(
                    "Selecione o pico de G2", min_value=1, max_value=length-1, value=loc_g2, step=1)

                # Now, we search the peak in the acceleration norm between the task onset and 200 ms later. The value of 200 ms was arbitrary and we observed that was suitable to the peak detection. This peak is the acceleration peak during the sit-to-standing transition.
                standing_peak_acc = np.max(
                    norm_waveform[slider_onset:slider_onset+200])
                standing_peak_loc = 0
                for i in norm_waveform:
                    if i != standing_peak_acc:
                        standing_peak_loc = standing_peak_loc + 1
                        standing_peak_latency = t[standing_peak_loc]
                    else:
                        break

                # Now, we search the peak in the acceleration norm between moment of the angular velocity peak of the pre-sitting turn and the task offset. This peak is the acceleration peak during the standing-to-sit transition.
                if loc_g2 < loc_offset:
                    sitting_peak_acc = np.max(norm_waveform[loc_g2:loc_offset])
                else:
                    sitting_peak_acc = np.max(norm_waveform[loc_g2:])
                sitting_peak_loc = 0
                for i in norm_waveform:
                    if i != sitting_peak_acc:
                        sitting_peak_loc = sitting_peak_loc + 1
                        sitting_peak_latency = t[sitting_peak_loc]
                    else:
                        break

                # Setting the sliders with acceleration peak positions
                slider_A1 = st.number_input(
                    "Secione o pico de A1", min_value=1, max_value=length-1, value=standing_peak_loc, step=1)
                slider_A2 = st.number_input(
                    "Selecione o pico de A2", min_value=1, max_value=length-1, value=sitting_peak_loc, step=1)

                # Extracting the features from iTUG
                sit_to_standing_duration = t[slider_A1] - t_gyro[slider_onset]
                walking_to_go_duration = t_gyro[slider_G1] - t[slider_A1]
                walking_to_return_duration = t_gyro[slider_G2] - \
                    t_gyro[slider_G1]
                return_to_sit_duration = t[slider_A2] - t_gyro[slider_G2]
                standing_to_sit_duration = t_gyro[slider_offset] - t[slider_A2]
                total_duration = sit_to_standing_duration + walking_to_go_duration + \
                    walking_to_return_duration + return_to_sit_duration + standing_to_sit_duration

                # Creating arrays to plot the baselines for onset and offset detection
                shade_baseline1_x = [
                    t[slider_baseline1], t[slider_baseline1], t[slider_baseline2], t[slider_baseline2]]
                shade_baseline1_y = [0, 0.5, 0.5, 0]
                shade_baseline2_x = [
                    t[slider_baseline3], t[slider_baseline3], t[slider_baseline4], t[slider_baseline4]]
                shade_baseline2_y = [0, 0.5, 0.5, 0]
            with t2:
                # Plotting the gyroscope norm with iTUG stages in color shades
                plt.figure(figsize=(5, 3))
                plt.plot(t_gyro, norm_waveform_gyro, 'k')
                plt.fill(shade_baseline1_x, shade_baseline1_y, 'b', alpha=0.2)
                plt.fill(shade_baseline2_x, shade_baseline2_y, 'b', alpha=0.2)
                lim_y = np.max(norm_waveform_gyro)
                shade_sitting_2_standing_x = [
                    t_gyro[slider_onset], t_gyro[slider_onset], t[slider_A1], t[slider_A1]]
                shade_y = [0, lim_y, lim_y, 0]
                plt.fill(shade_sitting_2_standing_x, shade_y,
                         color=(1, 0.5, 0.5), alpha=0.65)
                shade_go_x = [t[slider_A1], t[slider_A1],
                              t_gyro[slider_G1], t_gyro[slider_G1]]
                plt.fill(shade_go_x, shade_y, color=(0.6, 1, 0.5), alpha=0.65)
                shade_return_x = [t_gyro[slider_G1], t_gyro[slider_G1],
                                  t_gyro[slider_G2], t_gyro[slider_G2]]
                plt.fill(shade_return_x, shade_y,
                         color=(1, 1, 0.4), alpha=0.65)
                shade_pre_sitting_x = [t_gyro[slider_G2],
                                       t_gyro[slider_G2], t[slider_A2], t[slider_A2]]
                plt.fill(shade_pre_sitting_x, shade_y,
                         color=(0.5, 0.6, 1), alpha=0.65)
                shade_sitting_x = [t[slider_A2], t[slider_A2],
                                   t_gyro[slider_offset], t_gyro[slider_offset]]
                plt.fill(shade_sitting_x, shade_y,
                         color=(0.4, 0.4, 0.4), alpha=0.65)
                baseline_duration = str(
                    round(t[slider_baseline2] - t[slider_baseline1], 2)) + " s"
                plt.text(t[slider_baseline1], 0.6, baseline_duration)
                baseline_duration_offset = str(
                    round(t[slider_baseline4] - t[slider_baseline3], 2)) + " s"
                plt.text(t[slider_baseline3], 0.6, baseline_duration_offset)
                plt.plot([t_gyro[slider_onset], t_gyro[slider_onset]],
                         [0, lim_y], '--r')
                plt.plot([t_gyro[slider_offset], t_gyro[slider_offset]],
                         [0, lim_y], '--b')
                plt.plot(t_gyro[slider_G1], norm_waveform_gyro[slider_G1], marker='o',
                         markerfacecolor='none', markeredgecolor='k', markersize=14)
                plt.text(t_gyro[slider_G1-20],
                         norm_waveform_gyro[slider_G1]*1.05, 'G1')
                plt.plot(t_gyro[slider_G2], norm_waveform_gyro[slider_G2], marker='o',
                         markerfacecolor='none', markeredgecolor='k', markersize=14)
                plt.text(t_gyro[slider_G2-20],
                         norm_waveform_gyro[slider_G2]*1.05, 'G2')
                plt.ylim(0, np.max(norm_waveform_gyro)*1.2)
                plt.xlabel('Tempo (s)')
                plt.ylabel('Velocidade angular (rad/s)')
                st.pyplot(plt)

            # Plotting the accelerometer norm with iTUG stages in color shades
                fig = plt.figure(figsize=(5, 3))
                plt.plot(t, norm_waveform, 'k')
                lim_y = np.max(norm_waveform)
                shade_y = [0, lim_y, lim_y, 0]
                plt.plot(t[slider_A1], norm_waveform[slider_A1], marker='o',
                         markerfacecolor='none', markeredgecolor='k', markersize=14)
                plt.text(t[slider_A1-20], norm_waveform[slider_A1]*1.05, 'A1')
                plt.plot(t[slider_A2], norm_waveform[slider_A2], marker='o',
                         markerfacecolor='none', markeredgecolor='k', markersize=14)
                plt.text(t[slider_A2-20], norm_waveform[slider_A2]*1.05, 'A2')
                plt.fill(shade_sitting_2_standing_x, shade_y,
                         color=(1, 0.5, 0.5), alpha=0.65)
                plt.fill(shade_go_x, shade_y, color=(0.6, 1, 0.5), alpha=0.65)
                plt.fill(shade_return_x, shade_y,
                         color=(1, 1, 0.4), alpha=0.65)
                plt.fill(shade_pre_sitting_x, shade_y,
                         color=(0.5, 0.6, 1), alpha=0.65)
                plt.fill(shade_sitting_x, shade_y,
                         color=(0.4, 0.4, 0.4), alpha=0.65)
                plt.plot([t_gyro[slider_onset], t_gyro[slider_onset]],
                         [0, lim_y], '--r')
                plt.plot([t_gyro[slider_offset], t_gyro[slider_offset]],
                         [0, lim_y], '--b')
                plt.xlabel('Tempo (s)')
                plt.ylabel('Aceleração (g)')

                st.pyplot(plt)
            # Priting the feature values
            with t3:
                st.markdown(
                    '<h3 style="text-align: left; color: blue;">Resultados</h3>', unsafe_allow_html=True)
                st.text('Duração total (s) = ' + str(round(total_duration, 2)))
                st.text('Duração de sentar para levantar (s) = ' +
                        str(round(sit_to_standing_duration, 2)))
                st.text('Duração da caminhada de ida (s) = ' +
                        str(round(walking_to_go_duration, 2)))
                st.text('Duração da caminhada de retorno (s) = ' +
                        str(round(walking_to_return_duration, 2)))
                st.text('Duração de em pé para sentar (s) = ' +
                        str(round(standing_to_sit_duration, 2)))
                st.text('Pico de A1 (g) = ' +
                        str(round(norm_waveform[slider_A1], 2)))
                st.text('Pico de A2 (g) = ' +
                        str(round(norm_waveform[slider_A2], 2)))
                st.text('Pico de G1 (rad/s) = ' +
                        str(round(norm_waveform_gyro[slider_G1], 2)))
                st.text('Pico de G2 (rad/s) = ' +
                        str(round(norm_waveform_gyro[slider_G2], 2)))

                # Define the values to be saved
                results_dict = {
                    'Duração total (s) = ': str(round(total_duration, 5)),
                    'Duração de sentar para levantar (s) ': str(round(sit_to_standing_duration, 5)),
                    'Duração da caminhada de ida (s) ': str(round(walking_to_go_duration, 5)),
                    'Duração da caminhada de retorno (s) ': str(round(walking_to_return_duration, 5)),
                    'Duração de em pé para sentar (s) ': str(round(standing_to_sit_duration, 5)),
                    'Pico de subida (g) ': str(round(norm_waveform[slider_A1], 5)),
                    'Pico de descida (g) ': str(round(norm_waveform[slider_A2], 5)),
                    'Pico do primeiro giro (rad/s) ': str(round(norm_waveform_gyro[slider_G1], 5)),
                    'Pico do segundo giro (rad/s) ': str(round(norm_waveform_gyro[slider_G2], 5)),
                    'Impulso de subida (g/s) ': str(round(norm_waveform[slider_A1]/sit_to_standing_duration, 5)),
                    'Impulso de descida (g/s) ': str(round(norm_waveform[slider_A2]/standing_to_sit_duration, 5))
                }

                # Generate the results content
                content = create_results_content(name_file, results_dict)

                # Create download button for the text file directly
                st.download_button(
                    label="Download results",
                    data=content,
                    file_name=f"{name_file}.txt",
                    mime="text/plain"
                )