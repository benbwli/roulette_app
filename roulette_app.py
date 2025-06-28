import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.signal import find_peaks
import json
import os # For file path operations
import hashlib # For secure password hashing (good practice)

st.set_page_config(layout="wide")

# --- Password Configuration ---
CORRECT_PASSWORD_HASH = hashlib.sha256("roulette_god".encode()).hexdigest()
AUTH_KEY = "authenticated"

# --- File path for storing spins ---
SPINS_FILE = "roulette_spins.json"
# We'll also save chart configurations for full persistence now
CHART_CONFIG_FILE = "roulette_chart_configs.json"

# --- Function to load spins from file ---
def load_spins():
    if os.path.exists(SPINS_FILE):
        try:
            with open(SPINS_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            st.warning("Corrupted spins file, starting with empty spins.")
            return []
    return [] # Return empty list if file doesn't exist

# --- Function to save spins to file ---
def save_spins(spins_list):
    with open(SPINS_FILE, "w") as f:
        json.dump(spins_list, f)

# --- Function to load chart configurations from file ---
def load_chart_configs():
    if os.path.exists(CHART_CONFIG_FILE):
        try:
            with open(CHART_CONFIG_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            st.warning("Corrupted chart configurations file, starting with default charts.")
            return []
    return []

# --- Function to save chart configurations to file ---
def save_chart_configs(chart_configs_list):
    with open(CHART_CONFIG_FILE, "w") as f:
        json.dump(chart_configs_list, f)

# --- Initial Default Chart Configurations ---
def get_default_chart_configs():
    return [
        {'id': 1, 'bb_default': True, 'ranges': [{'min': 0, 'max': 26, 'formula': '-n'}, {'min': 27, 'max': 36, 'formula': 'n+40'}], 'ema_toggle': True, 'rsi_toggle': True, 'peaks_toggle': True, 'dots_toggle': True},
        {'id': 2, 'bb_default': False, 'ranges': [{'min': 0, 'max': 26, 'formula': '-n'}, {'min': 27, 'max': 36, 'formula': 'n+40'}], 'ema_toggle': True, 'rsi_toggle': True, 'peaks_toggle': True, 'dots_toggle': True}
    ]

# --- Authentication Logic ---
if AUTH_KEY not in st.session_state:
    st.session_state[AUTH_KEY] = False

def check_password():
    if st.session_state[AUTH_KEY]:
        return True

    st.sidebar.title("Login")
    password_attempt = st.sidebar.text_input("Enter Password", type="password")
    
    if password_attempt:
        # Hash the attempted password and compare
        hashed_attempt = hashlib.sha256(password_attempt.encode()).hexdigest()
        if hashed_attempt == CORRECT_PASSWORD_HASH:
            st.session_state[AUTH_KEY] = True
            st.experimental_rerun() # Rerun to show the app
            return True
        else:
            st.sidebar.error("Incorrect password")
    return False

# --- App entry point: Check authentication ---
if not check_password():
    st.stop() # Stop execution if not authenticated

# --- Continue with the app if authenticated ---
st.title("Interactive Roulette Spin Analyzer")

# --- Initialize Session State for Spins and Charts ---
if 'spins' not in st.session_state:
    st.session_state.spins = load_spins()

if 'charts' not in st.session_state:
    loaded_charts = load_chart_configs()
    if loaded_charts:
        st.session_state.charts = loaded_charts
        # Find the max ID to set next_chart_id correctly
        st.session_state.next_chart_id = max([c['id'] for c in loaded_charts]) + 1 if loaded_charts else 1
    else:
        st.session_state.charts = get_default_chart_configs()
        st.session_state.next_chart_id = len(st.session_state.charts) + 1 # 3 for default 2 charts


# --- User Input for New Spin ---
st.sidebar.header("Add New Spin")
new_spin_value = st.sidebar.number_input("Enter roulette number (0-36):", min_value=0, max_value=36, value=0, step=1, key="new_spin_input")

def add_spin_and_save():
    st.session_state.spins.append(new_spin_value)
    save_spins(st.session_state.spins) # Save to file after adding
    # To clear the input field after adding, you can explicitly set its value in session state
    # st.session_state["new_spin_input"] = 0 # This line would clear the input, but might feel less intuitive for continuous entry

st.sidebar.button("Record Spin", on_click=add_spin_and_save)

st.sidebar.markdown("---")
st.sidebar.write(f"Total Spins Recorded: **{len(st.session_state.spins)}**")
if st.sidebar.button("Clear All Spins", help="This will reset all recorded spins."):
    st.session_state.spins = []
    save_spins([]) # Clear the file as well
    st.experimental_rerun()

# --- Full Reset Function ---
def full_reset():
    st.session_state.spins = []
    save_spins([]) # Clear spins file

    st.session_state.charts = get_default_chart_configs()
    save_chart_configs(st.session_state.charts) # Save default charts to file

    st.session_state.next_chart_id = len(st.session_state.charts) + 1
    
    # Optionally, log out too
    st.session_state[AUTH_KEY] = False

    st.success("App has been fully reset. Please re-enter password.")
    st.experimental_rerun() # Rerun to reflect changes and potentially re-authenticate

st.sidebar.button("Full Reset App", help="Resets all spins and chart configurations to default.", on_click=full_reset)
st.sidebar.markdown("---")

# --- Functions for Chart Logic ---
def encode_roulette(n, ranges):
    for r in ranges:
        if r['min'] <= n <= r['max']:
            try:
                return eval(r['formula'], {}, {'n': n})
            except Exception as e:
                st.error(f"Error evaluating formula '{r['formula']}': {e}")
                return 0
    # st.warning(f"No matching range for roulette number: {n}. Returning 0.") # Suppress this for cleaner output
    return 0

def calculate_rsi(data, period=14):
    deltas = np.diff(data)
    if len(deltas) < period - 1:
        return np.zeros(len(data))

    seed_data = deltas[:period]
    up = seed_data[seed_data >= 0].sum() / period
    down = -seed_data[seed_data < 0].sum() / period

    rs = up / down if down != 0 else 0
    rsi_vals = np.zeros(len(data))

    if period > 0 and len(data) >= period:
        rsi_vals[:period] = 100 - 100 / (1 + rs) if (1 + rs) != 0 else 0
        up_avg = up
        down_avg = down

        for i in range(period, len(data)):
            delta = deltas[i - 1]
            if delta > 0:
                up_val = delta
                down_val = 0
            else:
                up_val = 0
                down_val = -delta

            up_avg = (up_avg * (period - 1) + up_val) / period
            down_avg = (down_avg * (period - 1) + down_val) / period

            rs = up_avg / down_avg if down_avg != 0 else 0
            rsi_vals[i] = 100 - 100 / (1 + rs) if (1 + rs) != 0 else 0

    return rsi_vals

def make_encoding_chart(chart_config):
    chart_id = chart_config['id']
    
    # Use a container to group chart controls and output for better organization
    chart_container = st.container()
    with chart_container:
        st.subheader(f"Roulette Chart #{chart_id}")
        
        col1, col2 = st.columns([2, 1])

        with col1:
            st.write("### Encoding Ranges")
            if 'ranges' not in chart_config:
                chart_config['ranges'] = []

            # Create a dynamic list to hold the range inputs
            current_ranges_input = []
            for i, r in enumerate(chart_config['ranges']):
                st.markdown(f"**Range {i+1}**")
                cols = st.columns([1, 1, 2, 0.5])
                
                # Use callbacks for range inputs to immediately update chart_config
                min_val = cols[0].number_input(f'Min {chart_id}-{i}', min_value=0, max_value=36, value=r['min'], key=f'min_{chart_id}_{i}')
                max_val = cols[1].number_input(f'Max {chart_id}-{i}', min_value=0, max_value=36, value=r['max'], key=f'max_{chart_id}_{i}')
                formula_val = cols[2].text_input(f'Encode Formula {chart_id}-{i}', value=r['formula'], key=f'formula_{chart_id}_{i}')
                
                chart_config['ranges'][i]['min'] = min_val
                chart_config['ranges'][i]['max'] = max_val
                chart_config['ranges'][i]['formula'] = formula_val

                if cols[3].button("Remove", key=f'remove_range_{chart_id}_{i}'):
                    chart_config['ranges'].pop(i)
                    save_chart_configs(st.session_state.charts) # Save changes
                    st.experimental_rerun()

            if st.button("Add Range", key=f'add_range_{chart_id}'):
                chart_config['ranges'].append({'min': 0, 'max': 36, 'formula': '-n'})
                save_chart_configs(st.session_state.charts) # Save changes
                st.experimental_rerun()

        with col2:
            st.write("### Display Options")
            # Update chart_config directly from checkbox values
            chart_config['bb_toggle'] = st.checkbox('Bollinger Bands', value=chart_config.get('bb_toggle', chart_config['bb_default']), key=f'bb_{chart_id}')
            chart_config['ema_toggle'] = st.checkbox('EMA', value=chart_config.get('ema_toggle', True), key=f'ema_{chart_id}')
            chart_config['rsi_toggle'] = st.checkbox('RSI', value=chart_config.get('rsi_toggle', True), key=f'rsi_{chart_id}')
            chart_config['peaks_toggle'] = st.checkbox('Peaks/Troughs', value=chart_config.get('peaks_toggle', True), key=f'peaks_{chart_id}')
            chart_config['dots_toggle'] = st.checkbox('Dots', value=chart_config.get('dots_toggle', True), key=f'dots_{chart_id}')

            # Save chart configs whenever a display option changes
            def save_chart_config_on_change():
                save_chart_configs(st.session_state.charts)

            # You can't directly assign on_change to checkboxes, but a simple rerender does the trick as values are read fresh
            # Or, for more granular control if needed:
            # st.checkbox('...', on_change=save_chart_config_on_change, args=(), key=...)

            if st.button("Remove Chart", key=f'remove_chart_{chart_id}', help="Removes this chart"):
                st.session_state.charts = [c for c in st.session_state.charts if c['id'] != chart_id]
                save_chart_configs(st.session_state.charts) # Save changes
                st.experimental_rerun()

        st.markdown("---")

        if st.session_state.spins:
            encoded_values = [encode_roulette(n, chart_config['ranges']) for n in st.session_state.spins]
            cumulative_scores = np.cumsum(encoded_values)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(cumulative_scores, label='Cumulative Score')

            if chart_config['ema_toggle']:
                window = 16
                if len(cumulative_scores) >= window:
                    ema_vals = np.convolve(cumulative_scores, np.ones(window)/window, mode='valid')
                    ax.plot(range(window-1, len(cumulative_scores)), ema_vals, color='orange', label=f'EMA {window}')

            if chart_config['bb_toggle']:
                window = 16
                if len(cumulative_scores) >= window:
                    rolling_mean = np.convolve(cumulative_scores, np.ones(window)/window, mode='valid')
                    rolling_std = np.array([np.std(cumulative_scores[max(0, i-window+1):i+1]) for i in range(window-1, len(cumulative_scores))])

                    upper_band = rolling_mean + rolling_std
                    lower_band = rolling_mean - rolling_std
                    ax.plot(range(window-1, len(cumulative_scores)), upper_band, color='purple', linestyle='--', label='BB +1σ')
                    ax.plot(range(window-1, len(cumulative_scores)), lower_band, color='purple', linestyle='--', label='BB -1σ')

            if chart_config['peaks_toggle']:
                if len(cumulative_scores) > 1:
                    peaks, _ = find_peaks(cumulative_scores, distance=5, prominence=10)
                    troughs, _ = find_peaks(-np.array(cumulative_scores), distance=5, prominence=10)
                    ax.scatter(peaks, np.array(cumulative_scores)[peaks], color='green', marker='^', s=80, label='Local Highs')
                    ax.scatter(troughs, np.array(cumulative_scores)[troughs], color='blue', marker='v', s=80, label='Local Lows')

            if chart_config['dots_toggle']:
                ax.scatter(range(len(cumulative_scores)), cumulative_scores, color='red', s=20, alpha=0.7, label='Spin Dots')

            ax.set_title(f"Roulette Chart #{chart_id} (Current Spins: {len(st.session_state.spins)}, Custom Encoding)")
            ax.set_xlabel("Spin Number")
            ax.set_ylabel("Cumulative Score")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

            if chart_config['rsi_toggle']:
                rsi_fig, rsi_ax = plt.subplots(figsize=(12, 2))
                rsi_vals = calculate_rsi(cumulative_scores)
                rsi_ax.plot(rsi_vals, color='magenta', label='RSI')
                rsi_ax.axhline(70, color='red', linestyle='--', linewidth=1)
                rsi_ax.axhline(30, color='green', linestyle='--', linewidth=1)
                rsi_ax.set_ylim(0, 100)
                rsi_ax.set_ylabel("RSI")
                rsi_ax.set_xlabel("Spin Number")
                rsi_ax.grid(True)
                rsi_ax.legend()
                st.pyplot(rsi_fig)
                plt.close(rsi_fig)
        else:
            st.info("No spins recorded yet. Add a spin using the sidebar input.")


# --- Main App Layout ---
st.markdown("---")

if st.button("Add New Chart"):
    new_chart_config = {
        'id': st.session_state.next_chart_id,
        'bb_default': False,
        'ranges': [{'min': 0, 'max': 26, 'formula': '-n'}, {'min': 27, 'max': 36, 'formula': 'n+40'}],
        'ema_toggle': True,
        'rsi_toggle': True,
        'peaks_toggle': True,
        'dots_toggle': True
    }
    st.session_state.charts.append(new_chart_config)
    st.session_state.next_chart_id += 1
    save_chart_configs(st.session_state.charts) # Save new chart config to file
    st.experimental_rerun()

# Display existing charts
# Ensure we save chart configs whenever an individual chart's display options change.
# The `on_change` callback approach is typically for specific widgets.
# For multiple settings within a chart, rerunning the app (which happens on any widget interaction)
# and then saving the entire `st.session_state.charts` list after the chart loop is often simplest.
# We'll call save_chart_configs after the loop.
for chart_conf in st.session_state.charts:
    make_encoding_chart(chart_conf)
    st.markdown("---")

# Save chart configurations after all charts have been processed
# This ensures any changes to display toggles or range inputs are persisted.
save_chart_configs(st.session_state.charts)