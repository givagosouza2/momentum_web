import streamlit as st
from streamlit_option_menu import option_menu
import pageOne_MS
import balance_MS
import iTUG_MS
import hand_tremor_MS
import ftt_MS
import sound_MS
import info_MS


st.set_page_config(layout="wide", initial_sidebar_state="expanded",
                   page_title="Momentum Web")


class MultiApp:
    def _init_(self):
        self.apps = []

    def add_app(self, title, function):
        self.append.apps({
            "title": title,
            "function": function
        })

    def run():
        with st.sidebar:
            app = option_menu(
                menu_title="Momentum Web",
                options=["Home", "Equilíbrio", "iTUG", "Tremor de mão",
                         "Finger tapping test", "Posição articular","Análise da voz", "Atualizações"],
                icons=["house-fill", "arrows-fullscreen", "arrow-bar-right",
                       "hand-index-thumb", "hand-index", "hand-index","soundwave", "info-circle-fill"],
                default_index=0,
                menu_icon='stars',
                styles={
                    "container": {"padding": "0!important", "background-color": "#fafafa"},
                    "icon": {"color": "orange", "font-size": "14px"},
                    "nav-link": {"font-size": "14px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
                    "nav-link-selected": {"background-color": "green"},
                }
            )

        if app == "Home":
            pageOne_MS.app()
        if app == "Equilíbrio":
            balance_MS.app()
        if app == "iTUG":
            iTUG_MS.app()
        if app == "Tremor de mão":
            hand_tremor_MS.app()
        if app == "Finger tapping test":
            ftt_MS.app()
        if app == "Posição articular":
            JPS_MS.app()
        if app == "Análise da voz":
            sound_MS.app()
        if app == "Atualizações":
            info_MS.app()

    run()
