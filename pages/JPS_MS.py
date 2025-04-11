import pandas as pd
import streamlit as st
import numpy as np
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.signal import butter, filtfilt
import math

def app():
    def butter_lowpass_filter(data, cutoff_freq, sample_rate, order=4):
        nyquist_freq = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist_freq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, data)
        return filtered_data
    
    
    uploaded_acc = st.file_uploader(
        "Selecione o aquivo do smartphone", type=["txt"],)
    if uploaded_acc is not None:
        if uploaded_acc is not None:
            df = pd.read_csv(uploaded_acc, sep=';')
            tempo = df.iloc[:,0]
            x = df.iloc[:, 1]
            y = df.iloc[:,2]
            z = df.iloc[:,3]
            st.warning('1', icon="⚠️")
    
            # Pre-processing data: interpolating to 100 Hz
            #interpf = scipy.interpolate.interp1d(tempo, x)
            #t_ = np.arange(
                #start=tempo[0], stop=tempo[len(tempo)-1], step=10)
            #x_ = interpf(t_)
            #t_vf, x_vf = t_/1000, x_
    
            #interpf = scipy.interpolate.interp1d(tempo, y)
            #t_ = np.arange(
                #start=tempo[0], stop=tempo[len(tempo)-1], step=10)
            #y_ = interpf(t_)
            #t_vf, y_vf = t_/1000, y_
    
            #interpf = scipy.interpolate.interp1d(tempo, z)
            #t_ = np.arange(
                #start=tempo[0], stop=tempo[len(tempo)-1], step=10)
            #z_ = interpf(t_)
            #t_vf, z_vf = t_/1000, z_
    
            # Set filter parameters
            cutoff_frequency = 40  # Cutoff frequency in Hz
            sample_rate = 100.0  # Sample rate in Hz
            st.warning('2', icon="⚠️")
            
            x_vf = butter_lowpass_filter(
                x, cutoff_frequency, sample_rate, order=4)
            y_vf = butter_lowpass_filter(
                y, cutoff_frequency, sample_rate, order=4)
            z_vf = butter_lowpass_filter(
                z, cutoff_frequency, sample_rate, order=4)
            
            st.warning('3', icon="⚠️")
            accelAngleX = np.arctan(y_vf/np.sqrt(x_vf**2+z_vf**2)) * 180/math.pi
            accelAngleY = np.arctan(x_vf/np.sqrt(y_vf**2+z_vf**2)) * 180/math.pi
            accelAngleZ = np.arctan(z_vf/np.sqrt(x_vf**2+y_vf**2)) * 180/math.pi
    
            angulo = accelAngleX+90
    
            flexao_90 = []
            extensao_90 = []
            for index, i in enumerate(angulo[500:len(angulo)]):
                if i < 40:
                    t1 = index+500
                    flexao_90.append(np.mean(angulo[t1-500:t1]))
                    break
            for index, i in enumerate(angulo[t1:len(angulo)]):
                if i > 40:
                    t2 = t1 + index
                    extensao_90.append(np.mean(angulo[t2-500:t2]))
                    break
            for index, i in enumerate(angulo[t2:len(angulo)]):
                if i < 40:
                    t3 = t2 + index
                    flexao_90.append(np.mean(angulo[t3-500:t3]))
                    break
            for index, i in enumerate(angulo[t3:len(angulo)]):
                if i > 40:
                    t4 = t3 + index
                    extensao_90.append(np.mean(angulo[t4-500:t4]))
                    break
            for index, i in enumerate(angulo[t4:len(angulo)]):
                if i < 40:
                    t5 = t4 + index
                    flexao_90.append(np.mean(angulo[t5-500:t5]))
                    break
            for index, i in enumerate(angulo[t5:len(angulo)]):
                if i > 40:
                    t6 = t5 + index
                    extensao_90.append(np.mean(angulo[t6-500:t6]))
                    break
            for index, i in enumerate(angulo[t6:len(angulo)]):
                if i < 40:
                    t7 = t6 + index
                    flexao_90.append(np.mean(angulo[t7-500:t7]))
                    break
            for index, i in enumerate(angulo[t7:len(angulo)]):
                if i > 40:
                    t8 = t7 + index
                    extensao_90.append(np.mean(angulo[t8-500:t8]))
                    break
            for index, i in enumerate(angulo[t8:len(angulo)]):
                if i < 40:
                    t9 = t8 + index
                    flexao_90.append(np.mean(angulo[t9-500:t9]))
                    break
            marcadores = st.checkbox('Mostrar marcadores temporais', value=True)
            fig, axs = plt.subplots(1, 1, figsize=(7, 3.5))
            t_vf = tempo
            axs.plot(t_vf, angulo, 'k')
            axs.set_ylabel('Joint angle (degrees)')
            axs.set_xlabel('Time (seconds)')
            if marcadores == True:
                axs.plot([t_vf[t1], t_vf[t1]], [0, 180], '--g')
                axs.plot([t_vf[t2], t_vf[t2]], [0, 180], '--g')
                axs.plot([t_vf[t3], t_vf[t3]], [0, 180], '--g')
                axs.plot([t_vf[t4], t_vf[t4]], [0, 180], '--g')
                axs.plot([t_vf[t5], t_vf[t5]], [0, 180], '--g')
                axs.plot([t_vf[t6], t_vf[t6]], [0, 180], '--g')
                axs.plot([t_vf[t7], t_vf[t7]], [0, 180], '--g')
                axs.plot([t_vf[t8], t_vf[t8]], [0, 180], '--g')
                axs.plot([t_vf[t9], t_vf[t9]], [0, 180], '--g')
            axs.set_ylim(0, 180)
            st.pyplot(fig)
    
            st.markdown('baseline = ' + str(flexao_90[0]))
            st.markdown('Média da flexão de 90 graus = ' +
                        str(np.mean(flexao_90[1:4])))
            st.markdown('Média da extensão de 0 graus = ' +
                        str(np.mean(extensao_90[0:3])))
    
            output_file = "output.txt"
    
            # Abrir o arquivo para escrita
            with open(output_file, "w") as file:
                file.write(str(flexao_90[1]) + "\t" + str(extensao_90[0]) + "\n")
                file.write(str(flexao_90[2]) + "\t" + str(extensao_90[1]) + "\n")
                file.write(str(flexao_90[3]) + "\t" + str(extensao_90[2]) + "\n")
                file.write(str(flexao_90[4]) + "\t" + str(extensao_90[3]))
    
            with open(output_file, "r") as file:
                file_contents = file.read()
    
            # Usar st.download_button para baixar o arquivo
            st.download_button("Baixar resultados - Equilíbrio",
                               data=file_contents, key='download_results')

    
