import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.title("Multi-Plot Roulette Encoder App")

# --- Session state for plots ---
if "plots" not in st.session_state:
    st.session_state.plots = []

def default_settings():
    return {
        "zero_val": 0,
        "range1_min": 1,
        "range1_max": 25,
        "range1_val": -1,
        "range2_min": 26,
        "range2_max": 36,
        "range2_val": 10,
        "custom_formula": "",
        "spin_input": "",
        "spins": [],
        "encoded": [],
        "cumulative": [],
    }

# --- Add new plot/settings ---
if st.button("Add New Settings and Plot"):
    st.session_state.plots.append(default_settings())

# --- Display and manage all plots ---
remove_indices = []
for idx, plot in enumerate(st.session_state.plots):
    with st.expander(f"Plot {idx+1} Settings & Data", expanded=True):
        st.markdown("**Custom Encoding Settings**")
        plot["zero_val"] = st.number_input(f"Value for 0 (Plot {idx+1})", value=plot["zero_val"], key=f"zero_val_{idx}")
        plot["range1_min"] = st.number_input(f"Range 1 Min (Plot {idx+1})", value=plot["range1_min"], key=f"range1_min_{idx}")
        plot["range1_max"] = st.number_input(f"Range 1 Max (Plot {idx+1})", value=plot["range1_max"], key=f"range1_max_{idx}")
        plot["range1_val"] = st.number_input(f"Value for Range 1 (Plot {idx+1})", value=plot["range1_val"], key=f"range1_val_{idx}")
        plot["range2_min"] = st.number_input(f"Range 2 Min (Plot {idx+1})", value=plot["range2_min"], key=f"range2_min_{idx}")
        plot["range2_max"] = st.number_input(f"Range 2 Max (Plot {idx+1})", value=plot["range2_max"], key=f"range2_max_{idx}")
        plot["range2_val"] = st.number_input(f"Value for Range 2 (Plot {idx+1})", value=plot["range2_val"], key=f"range2_val_{idx}")
        plot["custom_formula"] = st.text_input(
            f"Or enter a custom formula using 'n' (Plot {idx+1})", value=plot["custom_formula"], key=f"custom_formula_{idx}"
        )

        def custom_encode(n, plot):
            if plot["custom_formula"].strip():
                try:
                    return eval(plot["custom_formula"], {"n": n, "np": np})
                except Exception:
                    st.warning(f"Invalid custom formula for Plot {idx+1}. Using range encoding.")
            if n == 0:
                return plot["zero_val"]
            elif plot["range1_min"] <= n <= plot["range1_max"]:
                return plot["range1_val"]
            elif plot["range2_min"] <= n <= plot["range2_max"]:
                return plot["range2_val"]
            else:
                raise ValueError("Invalid roulette number")

        st.markdown("**Enter spins for this plot**")
        plot["spin_input"] = st.text_input(
            f"Enter spins (comma separated, e.g. 5,12,0,36) (Plot {idx+1}):",
            value=plot["spin_input"], key=f"spin_input_{idx}"
        )
        if st.button(f"Update Plot {idx+1}", key=f"update_plot_{idx}"):
            try:
                spins = [int(s.strip()) for s in plot["spin_input"].split(",") if s.strip() != ""]
                if not all(0 <= n <= 36 for n in spins):
                    st.warning(f"All spins must be between 0 and 36 for Plot {idx+1}.")
                else:
                    plot["spins"] = spins
                    plot["encoded"] = [custom_encode(n, plot) for n in spins]
                    plot["cumulative"] = np.cumsum(plot["encoded"]).tolist()
            except Exception as e:
                st.warning(f"Error in Plot {idx+1}: {e}")

        # Remove button for this plot
        if st.button(f"Remove Plot {idx+1}", key=f"remove_plot_{idx}"):
            remove_indices.append(idx)

        # Show plot if data exists
        if plot.get("cumulative"):
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

# Actually remove plots after iteration (to avoid index errors)
for idx in sorted(remove_indices, reverse=True):
    st.session_state.plots.pop(idx)