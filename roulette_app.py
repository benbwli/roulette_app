import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Session state for spins
if 'spins' not in st.session_state:
    st.session_state.spins = []
    st.session_state.encoded_values = []
    st.session_state.cumulative_scores = []

def encode_roulette(n):
    if 0 <= n <= 25:
        return -n
    elif 26 <= n <= 36:
        return n + 10
    else:
        raise ValueError("Invalid roulette number")

st.title("Roulette Street Encoder")

spin = st.number_input("Enter spin (0-36):", min_value=0, max_value=36, step=1)
if st.button("Add Spin"):
    st.session_state.spins.append(spin)
    st.session_state.encoded_values.append(encode_roulette(spin))
    if len(st.session_state.cumulative_scores) == 0:
        st.session_state.cumulative_scores.append(st.session_state.encoded_values[-1])
    else:
        st.session_state.cumulative_scores.append(
            st.session_state.cumulative_scores[-1] + st.session_state.encoded_values[-1]
        )

if st.session_state.cumulative_scores:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(st.session_state.cumulative_scores, label='Cumulative Score')
    ax.scatter(range(len(st.session_state.cumulative_scores)), st.session_state.cumulative_scores, color='red', s=20)
    ax.set_xlabel("Spin Number")
    ax.set_ylabel("Cumulative Score")
    ax.set_title("Cumulative Roulette Score (Encoded)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)