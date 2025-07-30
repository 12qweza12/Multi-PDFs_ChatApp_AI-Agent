import streamlit as st

def footer():
    st.markdown(
            """
            <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #f0f2f6; padding: 15px; text-align: center;">
                © <a href="https://intranet.hospital.tu.ac.th/" target="_blank">Thammasat Hospital University</a>
            </div>
            """,
            unsafe_allow_html=True
        )