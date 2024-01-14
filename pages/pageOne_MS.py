import streamlit as st


def app():
    st.markdown("<h1 style='text-align: center;color: blue;'>Momentum Web</h1>",
                unsafe_allow_html=True)
    # Primeiro parágrafo
    texto_markdown = """
        &nbsp;&nbsp;&nbsp;&nbsp;O Momentum Web é uma aplicação para análise de dados obtidos usando o aplicativo Momentum Sensors. Nesta aplicação há rotinas escritas em linguagem Python para a extração de característica de diferentes testes motores que podem ser acessados nas opção da barra lateral desta aplicação. A aplicação Web e o aplicativo Momentum Sensors são ferramentas idealizadas para pesquisadores interessados em análise do movimento, ofertando ferramentas de obtenção e análise de dados baseadas em conhecimento prévio, especialmente àqueles referentes às produções científicas relacionadas ao projeto Momentum que está descrito abaixo.  O Momentum Web terá atualizações constantes seja de novas análises quanto da inclusão de novos protocolos de avaliação do movimento usando o Momentum Sensors. Se você é usuário dos outros aplicativos do projeto Momentum (Momentum Science e Momentum Touch) verifique em cada janela de análise se os seus dados são compatíveis com as análises propostas.
        """
    st.markdown(
        f"<div style='text-align: justify;'>{texto_markdown}</div>", unsafe_allow_html=True)

    # Create two columns
    col1, col2 = st.columns(2)
    with col1:
        # Cabeçalho da primeira coluna
        st.markdown("<h2 style='color: blue;'>Sobre o projeto Momentum</h2>",
                    unsafe_allow_html=True)

        # Primeiro parágrafo
        texto_markdown = """
        &nbsp;&nbsp;&nbsp;&nbsp;O projeto Momentum é projeto colaborativo entre pesquisadores da Universidade Federal do Pará, da Universidade do Estado do Pará e do Instituto Federal de São Paulo. A ideia principal do projeto é o desenvolvimento de protocolos de avaliação sensório-motora que possam usar dos sensores presentes nos smartphones para poder extrair informações relacionados ao movimento humano.
        """
        st.markdown(
            f"<div style='text-align: justify;'>{texto_markdown}</div>", unsafe_allow_html=True)

        # Segundo parágrafo
        texto_markdown = """
        &nbsp;&nbsp;&nbsp;&nbsp;O projeto teve início em 2019 e estende-se até hoje. O primeiro aplicativo desenvolvido no projeto foi o Momentum Science, o qual tinha por objetivo salvar as leituras de sensores inerciais do smartphone durante a realização do movimento. Com o Momentum Science já foram publicados estudos que avaliaram o equilíbrio, mobilidade, ajustes posturais antecipatórios e tremor de mão.
        """
        st.markdown(
            f"<div style='text-align: justify;'>{texto_markdown}</div>", unsafe_allow_html=True)

        # Terceiro parágrafo
        texto_markdown = """
        &nbsp;&nbsp;&nbsp;&nbsp;O segundo aplicativo desenvolvido dentro do projeto foi o Momentum Touch, o qual captura os toques na tela do smartphone durante a realização do Finger Tapping test. Com este aplicativo foi publicado um artigo científico mostrando os efeitos da sexo e da dominância manual sobre o desempenho em 3 diferentes protocolos do teste.
        """
        st.markdown(
            f"<div style='text-align: justify;'>{texto_markdown}</div>", unsafe_allow_html=True)

        # Inserção do logo do projeto
        image_path = "logo.svg"
        st.image(image_path, caption='Logo do projeto Momentum',
                 use_column_width=True)

        # Quarto parágrafo
        texto_markdown = """
        &nbsp;&nbsp;&nbsp;&nbsp;O terceiro aplicativo desenvolvido no projeto é Momentum Sensors. Este aplicativo busca colocar em um único aplicativo acesso aos sensores presentes no smartphone. Além dos sensores inerciais e tela sensível ao toque que já eram acessados nos aplicativos anteriores do projeto, o Momentum Sensors adiciona o acesso ao microfone do smartphone e oferece uma aplicação Web para análise dos dados obtidos no aplicativo.
        """
        st.markdown(
            f"<div style='text-align: justify;'>{texto_markdown}</div>", unsafe_allow_html=True)

        # Quinto parágrafo
        texto_markdown = """
        &nbsp;&nbsp;&nbsp;&nbsp;Os aplicativos do projeto Momentum são aplicativos experimentais para pesquisadores ou profissionais interessados em análise do movimento. Eles estão em constante modificação devido a natureza do desenvolvimento tecnológico que é dinâmica e exige atualizações a cada vez que um novo conhecimento é gerado. Os aplicativos do projeto Momentum têm sido aplicados em trabalhos de conclusão de curso, dissertações de mestrado, teses de doutorado e a cada novo conhecimento gerado é nosso compromisso atualizar esta página e trazer as novas informações e aprimoramentos que puderem ser feitos.
        <br>
        <br>
        <br>
        """
        st.markdown(
            f"<div style='text-align: justify;'>{texto_markdown}</div>", unsafe_allow_html=True)

        # Inserção do logo do projeto
        image_path = "image1.jpeg"
        st.image(image_path, caption='Imagem gerada por IA',
                 use_column_width=True)

        st.markdown(
            f"<div style='text-align: justify;'>{texto_markdown}</div>", unsafe_allow_html=True)

    with col2:
        # Cabeçalho da primeira coluna
        st.markdown("<h2 style='color: blue;'>Princípios do projeto Momentum</h2>",
                    unsafe_allow_html=True)
        # Primeiro parágrafo
        texto_markdown = """
        &nbsp;&nbsp;&nbsp;&nbsp;Todos os aplicativos desenvolvidos dentro do projeto Momentum estão disponíveis na Play Store de forma gratuita. Qualquer um que queira utilizá-los poderá baixar em seu smartphone com ambiente operacional Android. A aplicação Web desta página para a realização de diferentes análises é voltada para os arquivos de saída do aplicativo Momentum Sensors. Qualquer dúvida ou sugestão, o usuário pode entrar em contato o Prof. Givago da Silva Souza da Universidade Federal do Pará que é um dos coordenadores do projeto pelo email givagosouza@ufpa.br. 
        """
        st.markdown(
            f"<div style='text-align: justify;'>{texto_markdown}</div>", unsafe_allow_html=True)

        # Segundo parágrafo
        texto_markdown = """
        &nbsp;&nbsp;&nbsp;&nbsp;O projeto Momentum tem como objetivo principal desenvolver protocolos de avaliação sensorio-motora para uso em smartphones e aplicá-la especialmente com pacientes do Sistema Único de Saúde do Brasil, já que em sua maioria apresentam baixo poder aquisitivo e têm pouco acesso à atendimentos com tecnologias padrão-ouro para análise do movimento. O uso de smartphones para a análise do movimento não deve substituir as ferramentas padrão-ouro para aavaliação do movimento (eletromiografia, plataforma de força ou captura por vídeo), mas servir de alternativa de baixo custo e fácil acesso que permita um monitoramento da saúde das pessoas com qualidade.
        <br>
        <br>
        <br>
        """
        st.markdown(
            f"<div style='text-align: justify;'>{texto_markdown}</div>", unsafe_allow_html=True)

        # Inserção do logo do projeto
        image_path = "image2.jpeg"
        st.image(image_path, caption='Imagem gerada por IA',
                 use_column_width=True)

        # Cabeçalho da primeira coluna
        st.markdown("<h2 style='color: blue;'>Produção bibliográfica com os aplicativos Momentum</h2>",
                    unsafe_allow_html=True)

        # Primeiro referência
        texto_markdown = """
        DUARTE MB, MORAES AAC, FERREIRA EV, ALMEIDA GCS, SANTOS EGR, PINTO GHL, OLIVEIRA PR, AMORIM CF, CABRAL AS, SAUNIER GJA, COSTA E SILVA AA, SOUZA GS, CALLEGARI B. Validity and reliability of a smartphone-based assessment for anticipatory and compensatory postural adjustments during predictable perturbations. GAIT & POSTURE, v. 96, p. 9-17, 2022.
        """
        st.markdown(
            f"<div style='text-align: justify;'>{texto_markdown}</div>", unsafe_allow_html=True)
        texto_com_link = "[link](https://www.sciencedirect.com/science/article/abs/pii/S0966636222001278)"
        st.markdown(texto_com_link, unsafe_allow_html=True)

        # Segunda referência
        texto_markdown = """
        MORAES AAC, DUARTE MB, FERREIRA EV, ALMEIDA GCS, SANTOS EGR, PINTO GHL, OLIVEIRA PR, AMORIM CF, CABRAL AS, COSTA E SILVA AA, SOUZA GS, CALLEGARI B. Validity and reliability of smartphone app for evaluating postural adjustments during step initiation. SENSORS, v. 1, p. 1, 2022.
        """
        st.markdown(
            f"<div style='text-align: justify;'>{texto_markdown}</div>", unsafe_allow_html=True)
        texto_com_link = "[link](https://www.mdpi.com/1424-8220/22/8/2935)"
        st.markdown(texto_com_link, unsafe_allow_html=True)

        # Terceira referência
        texto_markdown = """
        RODRIGUES LA, SANTOS EGR, SANTOS PSA, IGARASHI Y, OLIVEIRA LKR, PINTO GHL, SANTOS-LOBATO BL, CABRAL AS, BELGAMO A, COSTA E SILVA AA, CALLEGARI B, SOUZA GS. Wearable Devices and Smartphone Inertial Sensors for Static Balance Assessment: A Concurrent Validity Study in Young Adult Population. Journal Of Personalized Medicine, v. 1, p. 1-1, 2022.
        """
        st.markdown(
            f"<div style='text-align: justify;'>{texto_markdown}</div>", unsafe_allow_html=True)
        texto_com_link = "[link](https://www.mdpi.com/2075-4426/12/7/1019)"
        st.markdown(texto_com_link, unsafe_allow_html=True)

        # Quarta referência
        texto_markdown = """
        SANTOS PSA, SANTOS EGR, MONTEIRO LCP, SANTOS-LOBATO BL, PINTO GHL, BELGAMO A, CABRAL AS, COSTA E SILVA AA, CALLEGARI B, SOUZA GS. The hand tremor spectrum is modified by the inertial sensor mass during lightweight wearable and smartphone-based assessment in healthy young subjects. Scientific Reports, v. 12, p. 01, 2022.
        """
        st.markdown(
            f"<div style='text-align: justify;'>{texto_markdown}</div>", unsafe_allow_html=True)
        texto_com_link = "[link](https://www.nature.com/articles/s41598-022-21310-4)"
        st.markdown(texto_com_link, unsafe_allow_html=True)

        # Quinta referência
        texto_markdown = """
        MORAES AAC, DUARTE MB, SANTOS EJM, ALMEIDA GCS, CABRAL AS, COSTA E SILVA AA, GARCEZ DR, SOUZA GS, CALLEGARI B. Comparison of inertial records during anticipatory postural adjustments obtained with devices of different masses. PeerJ, v. 11, p. e15627, 2023.
        """
        st.markdown(
            f"<div style='text-align: justify;'>{texto_markdown}</div>", unsafe_allow_html=True)
        texto_com_link = "[link](https://peerj.com/articles/15627/)"
        st.markdown(texto_com_link, unsafe_allow_html=True)

        # Sexta referência
        texto_markdown = """
        BRITO FAC, MONTEIRO LCP, SANTOS EGR, LIMA RC; SANTOS-LOBATO BL, CABRAL AS, CALLEGARI B, COSTA E SILVA AAC; SOUZA GS. The role of sex and handedness in the performance of the smartphone-based Finger-Tapping Test. PLOS Digital Health, v. 2, p. e0000304, 2023.
        """
        st.markdown(
            f"<div style='text-align: justify;'>{texto_markdown}</div>", unsafe_allow_html=True)
        texto_com_link = "[link](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000304zx)"
        st.markdown(texto_com_link, unsafe_allow_html=True)

        # Sétima referência
        texto_markdown = """
        Correa BDC, Santos E, Belgamo A, Pinto G, Xavier SS, Silva CS, Dias AN, Paranhos A, Cabral A, Callegari B, Costa e Silva AA, Quaresma JAS, Falcao LFM, Souza GS. SMARTPHONE-BASED EVALUATION OF STATIC BALANCE AND MOBILITY IN LONG LASTING COVID-19 PATIENTS. Frontiers in Medicine, aceito para publicação
        """
        st.markdown(
            f"<div style='text-align: justify;'>{texto_markdown}</div>", unsafe_allow_html=True)
        texto_com_link = "[link](https://www.frontiersin.org/articles/10.3389/fneur.2023.1277408)"
        st.markdown(texto_com_link, unsafe_allow_html=True)
