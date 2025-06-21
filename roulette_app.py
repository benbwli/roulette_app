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

col1, col2 = st.columns(2)
with col1:
    spin_str = st.text_input("Enter spin (0-36):", value="")
    add = st.button("Add Spin")
with col2:
    reset = st.button("Reset All")

if reset:
    st.session_state.spins = []
    st.session_state.encoded_values = []
    st.session_state.cumulative_scores = []

if add:
    try:
        spin = int(spin_str)
        if 0 <= spin <= 36:
            st.session_state.spins.append(spin)
            st.session_state.encoded_values.append(encode_roulette(spin))
            if len(st.session_state.cumulative_scores) == 0:
                st.session_state.cumulative_scores.append(st.session_state.encoded_values[-1])
            else:
                st.session_state.cumulative_scores.append(
                    st.session_state.cumulative_scores[-1] + st.session_state.encoded_values[-1]
                )
        else:
            st.warning("Spin value must be between 0 and 36.")
    except ValueError:
        st.warning("Please enter a valid integer between 0 and 36.")

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