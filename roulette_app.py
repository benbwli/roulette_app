import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.signal import find_peaks
import json
import os
import hashlib

st.set_page_config(layout="wide")

# --- Password Configuration ---
CORRECT_PASSWORD_HASH = hashlib.sha256("roulette_god".encode()).hexdigest()
AUTH_KEY = "authenticated"

# --- File path for storing data ---
SPINS_FILE = "roulette_spins.json"
CHART_CONFIG_FILE = "roulette_chart_configs.json"

# --- Function to load data from file ---
def load_json_file(file_path, default_value, warning_message):
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            st.warning(warning_message)
            return default_value
    return default_value

# --- Function to save data to file ---
def save_json_file(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f)

# --- Initial Default Chart Configurations ---
def get_default_chart_configs():
    return [
        {
            'id': 1,
            'ranges': [{'min': 0, 'max': 26, 'formula': '-n'}, {'min': 27, 'max': 36, 'formula': 'n+40'}],
            'bb_toggle': True,
            'bb_settings': {'window': 16, 'std_devs': 1},
            'ema_toggles': [{'id': 1, 'enabled': True, 'window': 16}], # List of EMA configs
            'rsi_toggle': True,
            'peaks_toggle': True,
            'dots_toggle': True
        },
        {
            'id': 2,
            'ranges': [{'min': 0, 'max': 26, 'formula': '-n'}, {'min': 27, 'max': 36, 'formula': 'n+40'}],
            'bb_toggle': False, # Default BB off for second chart
            'bb_settings': {'window': 16, 'std_devs': 1},
            'ema_toggles': [{'id': 1, 'enabled': True, 'window': 16}],
            'rsi_toggle': True,
            'peaks_toggle': True,
            'dots_toggle': True
        }
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
        hashed_attempt = hashlib.sha256(password_attempt.encode()).hexdigest()
        if hashed_attempt == CORRECT_PASSWORD_HASH:
            st.session_state[AUTH_KEY] = True
            st.rerun() # Use st.rerun()
            return True
        else:
            st.sidebar.error("Incorrect password")
    return False

# --- App entry point: Check authentication ---
if not check_password():
    st.stop()

# --- Continue with the app if authenticated ---
st.title("Interactive Roulette Spin Analyzer")

# --- Initialize Session State for Spins and Charts ---
if 'spins' not in st.session_state:
    st.session_state.spins = load_json_file(SPINS_FILE, [], "Corrupted spins file, starting with empty spins.")

if 'charts' not in st.session_state:
    loaded_charts = load_json_file(CHART_CONFIG_FILE, [], "Corrupted chart configurations file, starting with default charts.")
    if loaded_charts:
        st.session_state.charts = loaded_charts
        # Find the maximum existing chart ID to ensure unique new IDs
        st.session_state.next_chart_id = max([c['id'] for c in loaded_charts]) + 1 if loaded_charts else 1
        # Also ensure EMA IDs are correctly handled on load
        for chart in st.session_state.charts:
            if 'ema_toggles' not in chart or not chart['ema_toggles']:
                chart['ema_toggles'] = [{'id': 1, 'enabled': True, 'window': 16}]
            else:
                # Ensure next_ema_id is correctly set if EMAs exist
                chart['next_ema_id'] = max([e['id'] for e in chart['ema_toggles']]) + 1 if chart['ema_toggles'] else 1
            # Ensure bb_settings exist if BB is toggled on during load
            if chart.get('bb_toggle', False) and 'bb_settings' not in chart:
                chart['bb_settings'] = {'window': 16, 'std_devs': 1}
    else:
        st.session_state.charts = get_default_chart_configs()
        st.session_state.next_chart_id = len(st.session_state.charts) + 1
        for chart in st.session_state.charts:
            chart['next_ema_id'] = len(chart['ema_toggles']) + 1 # Initialize next EMA ID for default charts


# --- User Input for New Spin (Single) ---
st.sidebar.header("Add New Spin")
new_spin_value = st.sidebar.number_input("Enter single roulette number (0-36):", min_value=0, max_value=36, value=0, step=1, key="new_spin_input")

# Callbacks for Add/Delete spins - removed st.rerun()
def add_spin_and_save_callback():
    st.session_state.spins.append(new_spin_value)
    save_json_file(st.session_state.spins, SPINS_FILE)

st.sidebar.button("Record Single Spin", on_click=add_spin_and_save_callback)

st.sidebar.markdown("---")

# --- User Input for Multiple Spins ---
st.sidebar.header("Add Multiple Spins")
multiple_spins_text = st.sidebar.text_area(
    "Enter multiple roulette numbers (0-36), separated by commas, spaces, or newlines:",
    value="",
    height=100,
    key="multiple_spins_input"
)

def add_multiple_spins_and_save_callback():
    if multiple_spins_text:
        raw_numbers = multiple_spins_text.replace(',', ' ').replace('\n', ' ').split(' ')
        parsed_numbers = []
        errors = []
        for num_str in raw_numbers:
            num_str = num_str.strip()
            if num_str:
                try:
                    num = int(num_str)
                    if 0 <= num <= 36:
                        parsed_numbers.append(num)
                    else:
                        errors.append(f"Number out of range (0-36): '{num_str}'")
                except ValueError:
                    errors.append(f"Invalid number format: '{num_str}'")
        
        if parsed_numbers:
            st.session_state.spins.extend(parsed_numbers)
            save_json_file(st.session_state.spins, SPINS_FILE)
            st.sidebar.success(f"Added {len(parsed_numbers)} spins.")
            if errors:
                st.sidebar.warning("Some numbers were not added due to errors:\n" + "\n".join(errors))
            # No st.rerun() here
        elif errors:
            st.sidebar.error("No valid spins to add. Errors found:\n" + "\n".join(errors))
        else:
            st.sidebar.info("No numbers entered.")

st.sidebar.button("Record Multiple Spins", on_click=add_multiple_spins_and_save_callback)


st.sidebar.markdown("---")
st.sidebar.write(f"Total Spins Recorded: **{len(st.session_state.spins)}**")

# --- Delete Last Spin Button ---
def delete_last_spin_callback():
    if st.session_state.spins:
        st.session_state.spins.pop()
        save_json_file(st.session_state.spins, SPINS_FILE)
    # No st.rerun() here

st.sidebar.button("Delete Last Spin", on_click=delete_last_spin_callback, disabled=(not st.session_state.spins), help="Removes the most recent spin entry.")

if st.sidebar.button("Clear All Spins", help="This will reset all recorded spins."):
    st.session_state.spins = []
    save_json_file(st.session_state.spins, SPINS_FILE)
    # No st.rerun() here, button click already triggers a rerun

# --- Full Reset Function ---
def full_reset():
    st.session_state.spins = []
    save_json_file(st.session_state.spins, SPINS_FILE)

    st.session_state.charts = get_default_chart_configs()
    save_json_file(st.session_state.charts, CHART_CONFIG_FILE)

    st.session_state.next_chart_id = len(st.session_state.charts) + 1
    for chart in st.session_state.charts:
        chart['next_ema_id'] = len(chart['ema_toggles']) + 1
    
    st.session_state[AUTH_KEY] = False

    st.success("App has been fully reset. Please re-enter password.")
    st.rerun() # Use st.rerun()

st.sidebar.button("Full Reset App", help="Resets all spins and chart configurations to default.", on_click=full_reset)
st.sidebar.markdown("---")


# --- Functions for Chart Logic and Indicator Calculations ---
def encode_roulette(n, ranges):
    for r in ranges:
        if r['min'] <= n <= r['max']:
            try:
                return eval(r['formula'], {}, {'n': n})
            except Exception as e:
                st.error(f"Error evaluating formula '{r['formula']}': {e}")
                return 0
    return 0

def calculate_rsi(data, period=14):
    # Ensure data is a NumPy array for consistent behavior
    data = np.asarray(data)
    
    if data.size < 2: # Need at least two data points for deltas
        return np.zeros(data.size) # Return array of zeros matching original data size
    
    deltas = np.diff(data)
    # Ensure deltas has enough length for the period
    if deltas.size < period - 1:
        return np.zeros(data.size)

    seed_data = deltas[:period]
    # Check if seed_data is empty before summing
    up = seed_data[seed_data >= 0].sum() / period if seed_data[seed_data >= 0].size > 0 else 0
    down = -seed_data[seed_data < 0].sum() / period if seed_data[seed_data < 0].size > 0 else 0


    rs = up / down if down != 0 else 0
    rsi_vals = np.zeros(data.size) # Initialize with size of original data

    if period > 0 and data.size >= period:
        rsi_vals[:period] = 100 - 100 / (1 + rs) if (1 + rs) != 0 else 0
        up_avg = up
        down_avg = down

        for i in range(period, data.size):
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

# Function to calculate expanding EMA
def calculate_expanding_ema(data, window):
    # Ensure data is a NumPy array
    data = np.asarray(data)
    ema_vals = []
    
    if data.size == 0: # Check if the numpy array is empty
        return np.array(ema_vals) 
    
    for i in range(data.size):
        # Ensure slice is not empty; np.mean on empty raises RuntimeWarning
        current_slice = data[:i+1] if i < window else data[i-window+1:i+1]
        
        if current_slice.size > 0: # Check if the slice itself is empty
            ema_vals.append(np.mean(current_slice))
        else:
            ema_vals.append(np.nan) # Append NaN if slice is unexpectedly empty
    return np.array(ema_vals)

# Function to calculate expanding Bollinger Bands
def calculate_expanding_bollinger_bands(data, window, std_devs):
    # Ensure data is a NumPy array
    data = np.asarray(data)
    rolling_mean = []
    rolling_std = []
    
    if data.size == 0: # Check if the numpy array is empty
        return np.array([]), np.array([]), np.array([])
    
    for i in range(data.size):
        current_slice = data[:i+1] if i < window else data[i-window+1:i+1]
        
        if current_slice.size > 0: # Check if the slice itself is empty
            mean_val = np.mean(current_slice)
            # Std dev of a single number is 0; np.std on < 2 elements is 0, not NaN
            std_val = np.std(current_slice)
        else:
            mean_val = np.nan
            std_val = np.nan 

        rolling_mean.append(mean_val)
        rolling_std.append(std_val)
    
    rolling_mean = np.array(rolling_mean)
    rolling_std = np.array(rolling_std)
    upper_band = rolling_mean + (std_devs * rolling_std)
    lower_band = rolling_mean - (std_devs * rolling_std)
    
    return rolling_mean, upper_band, lower_band


def make_encoding_chart(chart_config):
    chart_id = chart_config['id']
    
    chart_container = st.container()
    with chart_container:
        st.subheader(f"Roulette Chart #{chart_id}")
        
        col1, col2 = st.columns([2, 1])

        with col1:
            st.write("### Encoding Ranges")
            if 'ranges' not in chart_config:
                chart_config['ranges'] = []

            for i, r in enumerate(chart_config['ranges']):
                st.markdown(f"**Range {i+1}**")
                cols = st.columns([1, 1, 2, 0.5])
                
                min_val = cols[0].number_input(f'Min {chart_id}-{i}', min_value=0, max_value=36, value=r['min'], key=f'min_{chart_id}_{i}')
                max_val = cols[1].number_input(f'Max {chart_id}-{i}', min_value=0, max_value=36, value=r['max'], key=f'max_{chart_id}_{i}')
                formula_val = cols[2].text_input(f'Encode Formula {chart_id}-{i}', value=r['formula'], key=f'formula_{chart_id}_{i}')
                
                chart_config['ranges'][i]['min'] = min_val
                chart_config['ranges'][i]['max'] = max_val
                chart_config['ranges'][i]['formula'] = formula_val

                if cols[3].button("Remove", key=f'remove_range_{chart_id}_{i}'):
                    chart_config['ranges'].pop(i)
                    save_json_file(st.session_state.charts, CHART_CONFIG_FILE) # Save changes
                    st.rerun()

            if st.button("Add Range", key=f'add_range_{chart_id}'):
                chart_config['ranges'].append({'min': 0, 'max': 36, 'formula': '-n'})
                save_json_file(st.session_state.charts, CHART_CONFIG_FILE) # Save changes
                st.rerun()

        with col2:
            st.write("### Display Options")
            
            # Bollinger Bands Toggle and Settings
            bb_cols = st.columns([0.6, 0.4])
            chart_config['bb_toggle'] = bb_cols[0].checkbox('Bollinger Bands', value=chart_config.get('bb_toggle', True), key=f'bb_{chart_id}')
            if chart_config['bb_toggle']:
                with bb_cols[1].expander("Edit BB"):
                    if 'bb_settings' not in chart_config:
                        chart_config['bb_settings'] = {'window': 16, 'std_devs': 1}
                    
                    chart_config['bb_settings']['window'] = st.number_input(
                        'BB Length', min_value=2, max_value=100, value=chart_config['bb_settings']['window'], step=1, key=f'bb_len_{chart_id}'
                    )
                    chart_config['bb_settings']['std_devs'] = st.number_input(
                        'BB Std Devs', min_value=0.5, max_value=3.0, value=chart_config['bb_settings']['std_devs'], step=0.1, format="%.1f", key=f'bb_std_{chart_id}'
                    )
            
            # EMA Toggles and Settings (Multiple EMAs)
            ema_general_cols = st.columns([0.6, 0.4])
            # Check any(e['enabled'] for e in chart_config.get('ema_toggles',[])) handles if ema_toggles is missing or empty
            ema_general_toggle = ema_general_cols[0].checkbox('Show EMAs', value=any(e['enabled'] for e in chart_config.get('ema_toggles',[])), key=f'ema_general_{chart_id}')
            
            if 'ema_toggles' not in chart_config: # Ensure list exists before trying to append
                chart_config['ema_toggles'] = []

            if ema_general_toggle:
                with ema_general_cols[1].expander("Edit EMAs"):
                    if not chart_config['ema_toggles']:
                        st.info("No EMAs added yet. Click 'Add New EMA'.")
                    
                    for i, ema_conf in enumerate(chart_config['ema_toggles']):
                        ema_row_cols = st.columns([0.6, 0.3, 0.1])
                        ema_conf['enabled'] = ema_row_cols[0].checkbox(f'EMA {ema_conf["id"]}', value=ema_conf['enabled'], key=f'ema_toggle_{chart_id}_{ema_conf["id"]}')
                        ema_conf['window'] = ema_row_cols[1].number_input(f'Length {ema_conf["id"]}', min_value=2, max_value=100, value=ema_conf['window'], step=1, key=f'ema_len_{chart_id}_{ema_conf["id"]}')
                        if ema_row_cols[2].button('X', key=f'remove_ema_{chart_id}_{ema_conf["id"]}'):
                            chart_config['ema_toggles'].pop(i)
                            save_json_file(st.session_state.charts, CHART_CONFIG_FILE)
                            st.rerun()

                    if st.button("Add New EMA", key=f'add_ema_{chart_id}'):
                        # Ensure next_ema_id is correctly set
                        if 'next_ema_id' not in chart_config or chart_config['next_ema_id'] is None:
                            chart_config['next_ema_id'] = (max([e['id'] for e in chart_config['ema_toggles']]) + 1) if chart_config['ema_toggles'] else 1
                        chart_config['ema_toggles'].append({'id': chart_config['next_ema_id'], 'enabled': True, 'window': 16})
                        chart_config['next_ema_id'] += 1
                        save_json_file(st.session_state.charts, CHART_CONFIG_FILE)
                        st.rerun()


            chart_config['rsi_toggle'] = st.checkbox('RSI', value=chart_config.get('rsi_toggle', True), key=f'rsi_{chart_id}')
            chart_config['peaks_toggle'] = st.checkbox('Peaks/Troughs', value=chart_config.get('peaks_toggle', True), key=f'peaks_{chart_id}')
            chart_config['dots_toggle'] = st.checkbox('Dots', value=chart_config.get('dots_toggle', True), key=f'dots_{chart_id}')
            
            if st.button("Remove Chart", key=f'remove_chart_{chart_id}', help="Removes this chart"):
                st.session_state.charts = [c for c in st.session_state.charts if c['id'] != chart_id]
                save_json_file(st.session_state.charts, CHART_CONFIG_FILE)
                st.rerun()

        st.markdown("---")

        if st.session_state.spins:
            encoded_values = [encode_roulette(n, chart_config['ranges']) for n in st.session_state.spins]
            cumulative_scores = np.cumsum(encoded_values)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(cumulative_scores, label='Cumulative Score')

            # EMA Plots (Loop through multiple EMAs)
            if ema_general_toggle:
                for ema_conf in chart_config['ema_toggles']:
                    if ema_conf['enabled']:
                        ema_vals = calculate_expanding_ema(cumulative_scores, ema_conf['window'])
                        if ema_vals.size > 0 and not np.all(np.isnan(ema_vals)):
                             ax.plot(ema_vals, label=f'EMA {ema_conf["window"]}')

            # Bollinger Bands Plot
            if chart_config['bb_toggle']:
                bb_window = chart_config['bb_settings']['window']
                bb_std_devs = chart_config['bb_settings']['std_devs']
                _, upper_band, lower_band = calculate_expanding_bollinger_bands(cumulative_scores, bb_window, bb_std_devs)
                if upper_band.size > 0 and not np.all(np.isnan(upper_band)):
                    ax.plot(upper_band, color='purple', linestyle='--', label=f'BB +{bb_std_devs}σ ({bb_window})')
                    ax.plot(lower_band, color='purple', linestyle='--', label=f'BB -{bb_std_devs}σ ({bb_window})')


            # Peaks/Troughs
            if chart_config['peaks_toggle']:
                if len(cumulative_scores) > 1: # find_peaks requires at least 2 points
                    peaks, _ = find_peaks(cumulative_scores, distance=5, prominence=10)
                    troughs, _ = find_peaks(-np.array(cumulative_scores), distance=5, prominence=10)
                    ax.scatter(peaks, np.array(cumulative_scores)[peaks], color='green', marker='^', s=80, label='Local Highs')
                    ax.scatter(troughs, np.array(cumulative_scores)[troughs], color='blue', marker='v', s=80, label='Local Lows')

            # Dots
            if chart_config['dots_toggle']:
                ax.scatter(range(len(cumulative_scores)), cumulative_scores, color='red', s=20, alpha=0.7, label='Spin Dots')

            ax.set_title(f"Roulette Chart #{chart_id} (Current Spins: {len(st.session_state.spins)}, Custom Encoding)")
            ax.set_xlabel("Spin Number")
            ax.set_ylabel("Cumulative Score")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

            # RSI Plot (separate figure as it has its own Y-axis)
            if chart_config['rsi_toggle']:
                rsi_fig, rsi_ax = plt.subplots(figsize=(12, 2))
                rsi_vals = calculate_rsi(cumulative_scores)
                if rsi_vals.size > 0 and not np.all(np.isnan(rsi_vals)):
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
                    st.info(f"RSI for Chart #{chart_id} cannot be computed with current data.")
        else:
            st.info("No spins recorded yet. Add a spin using the sidebar input.")


# --- Main App Layout ---
st.markdown("---")

if st.button("Add New Chart"):
    new_chart_config = {
        'id': st.session_state.next_chart_id,
        'ranges': [{'min': 0, 'max': 26, 'formula': '-n'}, {'min': 27, 'max': 36, 'formula': 'n+40'}],
        'bb_toggle': True,
        'bb_settings': {'window': 16, 'std_devs': 1},
        'ema_toggles': [{'id': 1, 'enabled': True, 'window': 16}],
        'next_ema_id': 2, # Initialize next EMA ID for the new chart
        'rsi_toggle': True,
        'peaks_toggle': True,
        'dots_toggle': True
    }
    st.session_state.charts.append(new_chart_config)
    st.session_state.next_chart_id += 1
    save_json_file(st.session_state.charts, CHART_CONFIG_FILE)
    st.rerun()

for chart_conf in st.session_state.charts:
    make_encoding_chart(chart_conf)
    st.markdown("---")

# Save chart configurations after all charts have been processed
# This ensures any changes to display toggles or range inputs are persisted.
save_json_file(st.session_state.charts, CHART_CONFIG_FILE)
