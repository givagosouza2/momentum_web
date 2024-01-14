import streamlit as st
from scipy import signal
import pandas as pd
import numpy as np
import scipy
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point


def app():
    def ellipse_model(x, y):
        centro_u = np.mean(x)
        centro_v = np.mean(y)

        for j in range(1):
            P = np.column_stack((x, y))
            K = ConvexHull(P).vertices
            K = np.unique(K)
            PK = P[K].T
            d, N = PK.shape
            Q = np.zeros((d + 1, N))
            Q[:d, :] = PK[:d, :N]
            Q[d, :] = np.ones(N)

            count = 1
            err = 1
            u = (1 / N) * np.ones(N)
            tolerance = 0.01

            while err > tolerance:
                X = Q @ np.diag(u) @ Q.T
                M = np.diag(Q.T @ np.linalg.inv(X) @ Q)
                j = np.argmax(M)
                maximum = M[j]
                step_size = (maximum - d - 1) / ((d + 1) * (maximum - 1))
                new_u = (1 - step_size) * u
                new_u[j] = new_u[j] + step_size
                count = count + 1
                err = np.linalg.norm(new_u - u)
                u = new_u

            U = np.diag(u)
            A = (1 / d) * np.linalg.inv(PK @ U @ PK.T - (PK @ u) @ (PK @ u).T)
            c = PK @ u
            U, Q, V = np.linalg.svd(A)
            r1 = 1 / np.sqrt(Q[0])
            r2 = 1 / np.sqrt(Q[1])
            v = np.array([r1, r2, c[0], c[1], V[0, 0]])

            D = v[1]
            d = v[0]
            tan = centro_v / centro_u
            arco = np.arctan(tan)
            rot = arco
            angle = np.arange(0, 2 * np.pi, 2 * np.pi / 360)
            u = centro_u + (D * np.cos(angle) * np.cos(rot)) - \
                (d * np.sin(angle) * np.sin(rot))
            v = centro_v + (D * np.cos(angle) * np.sin(rot)) + \
                (d * np.sin(angle) * np.cos(rot))
            phi = rot * 180 / np.pi
            DM = D
            dm = d
            q = len(x)
            positive = 0
            number_of_true = len(angle)

            while number_of_true/q > 0.95:
                u = centro_u + (DM * np.cos(angle) * np.cos(rot)) - \
                    (dm * np.sin(angle) * np.sin(rot))
                v = centro_v + (DM * np.cos(angle) * np.sin(rot)) + \
                    (dm * np.sin(angle) * np.cos(rot))
                vertices = zip(u, v)
                polygon = Polygon(vertices)
                for xi, yi in zip(x, y):
                    points = Point(xi, yi)
                    if polygon.contains(points) == True:
                        positive = positive + 1
                number_of_true = positive
                positive = 0
                DM = DM - DM*0.2
                dm = dm - dm*0.2
        return u, v, DM, dm, phi

    st.markdown(
        '<h1 style="text-align: center; color: blue;">Finger tapping test com smartphone</h1>', unsafe_allow_html=True)

    # Primeiro parágrafo
    texto_markdown = """
        &nbsp;&nbsp;&nbsp;&nbsp; A avaliação da coordenação motora e mobilidade da mão realizada aqui nesta página é baseada no desempenho durante o Finger Tapping test. As informações de coordenadas espaciais e tempo de cada um dos toques são convertidas em características globais, temporais e espaciais como feito em Brito et al. (2023). Os arquivos exportados do Momentum Touch não são compatíveis com a presente rotina de análise.
        <br>
        <br>
        <br>
        """
    st.markdown(
        f"<div style='text-align: justify;'>{texto_markdown}</div>", unsafe_allow_html=True)

    t1, t2, t3 = st.columns([0.75, 1.75, 1])

    # Create acceleration and gyroscope file upload buttons
    uploaded_ftt = st.file_uploader(
        "Carregue o arquivo de texto do finger tapping test", type=["txt"],)
    if uploaded_ftt is not None:
        custom_separator = ';'
        df = pd.read_csv(uploaded_ftt, sep=custom_separator)
        t = df.iloc[:, 0]/1000
        x = df.iloc[:, 1]
        y = df.iloc[:, 2]

        dados = x, y

        ellipse_x, ellipse_y, D, d, angle = ellipse_model(x, y)
        n_touches = len(x)
        intervals = np.diff(t)

        mean_interval = np.mean(intervals)
        max_interval = np.max(intervals)
        min_interval = np.min(intervals)
        std_interval = np.std(intervals)
        ellipse_area = np.pi*D*d
        ellipse_major_axis = D
        ellipse_minor_axis = d
        ellipse_rotate_angle = angle
        total_deviation_ftt = np.sum(np.sqrt(x**2+y**2))

        with t1:
            plt.figure(figsize=(5, 5))
            plt.plot(x, y, '+', markersize=4, markeredgecolor='k')
            plt.plot(ellipse_x, ellipse_y, 'r')
            plt.fill(ellipse_x, ellipse_y, 'r', alpha=0.3)
            plt.xlim(np.min(ellipse_x), np.max(ellipse_x))
            plt.ylim(np.min(ellipse_y), np.max(ellipse_y))
            plt.axis('off')
            plt.gca().set_aspect('equal')
            st.pyplot(plt)
        with t2:
            plt.figure(figsize=(5, 5))
            plt.plot(t[0:len(intervals)], intervals, 'g')
            plt.xlabel('Tempo (s)')
            plt.ylabel('Intervalo entre os toques (s)')
            st.pyplot(plt)
        with t3:
            st.markdown(
                '<h3 style="text-align: left; color: blue;">Resultados</h3>', unsafe_allow_html=True)
            st.markdown("*Parâmetros globais*")
            st.text('Número de toques = ' + str(n_touches))
            st.markdown("*Parâmetros temporais*")
            st.text('Intervalo médio (s) = ' + str(round(mean_interval, 3)))
            st.text('Desvio-padrão dos intervalos (s) = ' +
                    str(round(std_interval, 3)))
            st.text('Intervalo máximo (s) = ' + str(round(max_interval, 2)))
            st.text('Intervalo mínimo (s) = ' + str(round(min_interval, 2)))
            st.markdown("*Parâmetros espaciais*")
            st.text('Desvio total (px) = ' +
                    str(round(total_deviation_ftt, 2)))
            st.text('Área da elipse (px)= ' + str(round(ellipse_area, 2)))
            st.text('Eixo maior (px) = ' +
                    str(round(ellipse_major_axis, 2)))
            st.text('Eixo menor (px) = ' +
                    str(round(ellipse_minor_axis, 2)))
            st.text('Ângulo de rotação (graus) = ' +
                    str(round(ellipse_rotate_angle, 2)))
