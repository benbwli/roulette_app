import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.title("Customizable Roulette Encoder & Multi-Plot App")

# --- Session state for multiple plots ---
if 'plots' not in st.session_state:
    st.session_state.plots = []

# --- Encoding configuration ---
with st.expander("Encoding Settings", expanded=True):
    st.markdown("**Define your encoding rules below.**")
    zero_val = st.number_input("Value for 0", value=0, key="zero_val")
    range1_min = st.number_input("Range 1 Min", value=1, key="range1_min")
    range1_max = st.number_input("Range 1 Max", value=25, key="range1_max")
    range1_val = st.number_input("Value for Range 1 (inclusive)", value=-1, key="range1_val")
    range2_min = st.number_input("Range 2 Min", value=26, key="range2_min")
    range2_max = st.number_input("Range 2 Max", value=36, key="range2_max")
    range2_val = st.number_input("Value for Range 2 (inclusive)", value=10, key="range2_val")
    custom_formula = st.text_input(
        "Or enter a custom formula using 'n' (leave blank to use above ranges):",
        value="", key="custom_formula"
    )

def custom_encode(n):
    if custom_formula.strip():
        try:
            return eval(custom_formula, {"n": n, "np": np})
        except Exception:
            st.warning("Invalid custom formula. Using range encoding.")
    if n == 0:
        return zero_val
    elif range1_min <= n <= range1_max:
        return range1_val if not callable(range1_val) else range1_val(n)
    elif range2_min <= n <= range2_max:
        return range2_val if not callable(range2_val) else range2_val(n)
    else:
        raise ValueError("Invalid roulette number")

# --- UI for adding a new plot ---
st.header("Add Spins to a New Plot")
spin_input = st.text_input("Enter spins (comma separated, e.g. 5,12,0,36):", value="", key="spin_input")
add_plot = st.button("Add Plot")

if add_plot:
    try:
        spins = [int(s.strip()) for s in spin_input.split(",") if s.strip() != ""]
        if not all(0 <= n <= 36 for n in spins):
            st.warning("All spins must be between 0 and 36.")
        else:
            encoded = [custom_encode(n) for n in spins]
            cumulative = np.cumsum(encoded).tolist()
            st.session_state.plots.append({
                "spins": spins,
                "encoded": encoded,
                "cumulative": cumulative,
                "encoding": custom_formula if custom_formula.strip() else f"Ranges: 0={zero_val}, {range1_min}-{range1_max}={range1_val}, {range2_min}-{range2_max}={range2_val}"
            })
    except Exception as e:
        st.warning(f"Error: {e}")

if st.button("Reset All Plots"):
    st.session_state.plots = []

# --- Display all plots ---
for idx, plot in enumerate(st.session_state.plots):
    st.subheader(f"Plot {idx+1} - Encoding: {plot['encoding']}")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(plot["cumulative"], label='Cumulative Score')
    ax.scatter(range(len(plot["cumulative"])), plot["cumulative"], color='red', s=20)
    ax.set_xlabel("Spin Number")
    ax.set_ylabel("Cumulative Score")
    ax.set_title(f"Cumulative Roulette Score (Plot {idx+1})")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
    st.markdown(f"**Spins:** {plot['spins']}")
    st.markdown(f"**Encoded:** {plot['encoded']}")