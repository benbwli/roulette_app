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
# Add a file for the fixed Odd/Even chart config too
ODDEVEN_CHART_CONFIG_FILE = "roulette_oddeven_config.json"


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
            'bb_settings': {'window': 16, 'std_devs': 1.0}, # Changed 1 to 1.0
            'ema_toggles': [{'id': 1, 'enabled': True, 'window': 16}],
            'rsi_toggle': True,
            'peaks_toggle': True,
            'dots_toggle': True
        },
        {
            'id': 2,
            'ranges': [{'min': 0, 'max': 26, 'formula': '-n'}, {'min': 27, 'max': 36, 'formula': 'n+40'}],
            'bb_toggle': False,
            'bb_settings': {'window': 16, 'std_devs': 1.0}, # Changed 1 to 1.0
            'ema_toggles': [{'id': 1, 'enabled': True, 'window': 16}],
            'rsi_toggle': True,
            'peaks_toggle': True,
            'dots_toggle': True
        }
    ]

# --- Initial Default Odd/Even Chart Configuration ---
def get_default_oddeven_config():
    return {
        'bb_toggle': True,
        'bb_settings': {'window': 16, 'std_devs': 2.0}, # Changed 2 to 2.0
        'ema_toggles': [{'id': 1, 'enabled': True, 'window': 16}],
        'rsi_toggle': True,
        'peaks_toggle': True,
        'dots_toggle': True,
        'next_ema_id': 2,
        'last_parity': None,
        'current_streak': 0,
        'streak_type': None
    }
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
        st.session_state.next_chart_id = max([c['id'] for c in loaded_charts]) + 1 if loaded_charts else 1
        for chart in st.session_state.charts:
            if 'ema_toggles' not in chart or not chart['ema_toggles']:
                chart['ema_toggles'] = [{'id': 1, 'enabled': True, 'window': 16}]
            if 'next_ema_id' not in chart: # Ensure this exists for older saved configs
                chart['next_ema_id'] = (max([e['id'] for e in chart['ema_toggles']]) + 1) if chart['ema_toggles'] else 1
            if chart.get('bb_toggle', False) and 'bb_settings' not in chart:
                chart['bb_settings'] = {'window': 16, 'std_devs': 1}
    else:
        st.session_state.charts = get_default_chart_configs()
        st.session_state.next_chart_id = len(st.session_state.charts) + 1
        for chart in st.session_state.charts:
            chart['next_ema_id'] = len(chart['ema_toggles']) + 1

# Initialize Odd/Even Chart Configuration
if 'oddeven_config' not in st.session_state:
    loaded_oddeven_config = load_json_file(ODDEVEN_CHART_CONFIG_FILE, None, "Corrupted Odd/Even chart configuration, resetting to default.")
    if loaded_oddeven_config:
        st.session_state.oddeven_config = loaded_oddeven_config
        # Ensure 'next_ema_id' exists for loaded config
        if 'ema_toggles' not in st.session_state.oddeven_config or not st.session_state.oddeven_config['ema_toggles']:
            st.session_state.oddeven_config['ema_toggles'] = [{'id': 1, 'enabled': True, 'window': 16}]
        if 'next_ema_id' not in st.session_state.oddeven_config:
            st.session_state.oddeven_config['next_ema_id'] = (max([e['id'] for e in st.session_state.oddeven_config['ema_toggles']]) + 1) if st.session_state.oddeven_config['ema_toggles'] else 1
    else:
        st.session_state.oddeven_config = get_default_oddeven_config()


# --- User Input for New Spin (Single) ---
st.sidebar.header("Add New Spin")
new_spin_value = st.sidebar.number_input("Enter single roulette number (0-36):", min_value=0, max_value=36, value=0, step=1, key="new_spin_input")

def add_spin_and_save_callback():
    st.session_state.spins.append(new_spin_value)
    save_json_file(st.session_state.spins, SPINS_FILE)
    # The oddeven_config's streak state also needs to be reset/recalculated
    # on spin addition, so a full rerun is needed for it to update correctly.
    # No explicit rerun needed here, button click causes one.

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
    # Also reset the Odd/Even streak state when clearing all spins
    st.session_state.oddeven_config['last_parity'] = None
    st.session_state.oddeven_config['current_streak'] = 0
    st.session_state.oddeven_config['streak_type'] = None
    save_json_file(st.session_state.oddeven_config, ODDEVEN_CHART_CONFIG_FILE) # Save the reset state
    st.rerun() # Forces the Odd/Even chart to recalculate from scratch

# --- Full Reset Function ---
def full_reset():
    st.session_state.spins = []
    save_json_file(st.session_state.spins, SPINS_FILE)

    st.session_state.charts = get_default_chart_configs()
    save_json_file(st.session_state.charts, CHART_CONFIG_FILE)

    st.session_state.next_chart_id = len(st.session_state.charts) + 1
    for chart in st.session_state.charts:
        chart['next_ema_id'] = len(chart['ema_toggles']) + 1
    
    st.session_state.oddeven_config = get_default_oddeven_config() # Reset Odd/Even chart config
    save_json_file(st.session_state.oddeven_config, ODDEVEN_CHART_CONFIG_FILE)

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
    data = np.asarray(data)
    
    if data.size < 2:
        return np.zeros(data.size)
    
    deltas = np.diff(data)
    if deltas.size < period - 1:
        return np.zeros(data.size)

    seed_data = deltas[:period]
    up = seed_data[seed_data >= 0].sum() / period if seed_data[seed_data >= 0].size > 0 else 0
    down = -seed_data[seed_data < 0].sum() / period if seed_data[seed_data < 0].size > 0 else 0

    rs = up / down if down != 0 else 0
    rsi_vals = np.zeros(data.size)

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

def calculate_expanding_ema(data, window):
    data = np.asarray(data)
    ema_vals = []
    
    if data.size == 0:
        return np.array(ema_vals) 
    
    for i in range(data.size):
        current_slice = data[:i+1] if i < window else data[i-window+1:i+1]
        
        if current_slice.size > 0:
            ema_vals.append(np.mean(current_slice))
        else:
            ema_vals.append(np.nan)
    return np.array(ema_vals)

def calculate_expanding_bollinger_bands(data, window, std_devs):
    data = np.asarray(data)
    rolling_mean = []
    rolling_std = []
    
    if data.size == 0:
        return np.array([]), np.array([]), np.array([])
    
    for i in range(data.size):
        current_slice = data[:i+1] if i < window else data[i-window+1:i+1]
        
        if current_slice.size > 0:
            mean_val = np.mean(current_slice)
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

# --- New Scoring Logic for Odd/Even Streaks ---
def calculate_odd_even_scores(spins_list, oddeven_state):
    scores = []
    
    # Initialize state for this calculation pass
    last_parity = oddeven_state['last_parity']
    current_streak = oddeven_state['current_streak']
    streak_type = oddeven_state['streak_type'] # 'odd', 'even', or None

    for n in spins_list:
        if n == 0:
            score = 0
            # Reset streak on zero
            last_parity = None
            current_streak = 0
            streak_type = None
        else:
            current_parity = n % 2 # 0 for even, 1 for odd

            if last_parity is None: # First non-zero spin or after a 0
                score = 1 if current_parity == 0 else -1 # +1 for even, -1 for odd (start of new score system)
                current_streak = 1
                streak_type = 'even' if current_parity == 0 else 'odd'
            elif current_parity == last_parity: # Streak continues
                current_streak += 1
                score = current_streak if current_parity == 0 else -current_streak
            else: # Streak broken, new streak starts
                score = 1 if current_parity == 0 else -1
                current_streak = 1
                streak_type = 'even' if current_parity == 0 else 'odd'
            
            last_parity = current_parity
        
        scores.append(score)

    # Return calculated scores AND the updated state for next time
    return np.array(scores), {
        'last_parity': last_parity,
        'current_streak': current_streak,
        'streak_type': streak_type
    }


# --- Function to Render a Generic Indicator Chart ---
def render_indicator_chart(title, spins_data, chart_config, chart_type_key="custom"):
    st.subheader(title)
    
    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("### Chart Data & Calculations")
        # Display relevant info for this chart type
        if chart_type_key == "custom":
            st.write("Current custom encoding ranges are applied to spins.")
        elif chart_type_key == "oddeven":
            st.write("Spins are encoded based on odd/even streaks (0 is neutral).")
            # Display current streak info if applicable
            current_oddeven_state = chart_config.get('oddeven_state_info', {})
            if current_oddeven_state.get('current_streak', 0) > 0:
                st.markdown(f"**Current Streak:** {current_oddeven_state['current_streak']} {current_oddeven_state['streak_type']}s")
            else:
                st.markdown("**Current Streak:** None")

    with col2:
        st.write("### Display Options")
        
        # Bollinger Bands Toggle and Settings
        bb_cols = st.columns([0.6, 0.4])
        # Use chart_type_key in key to ensure uniqueness across chart types
        chart_config['bb_toggle'] = bb_cols[0].checkbox('Bollinger Bands', value=chart_config.get('bb_toggle', True), key=f'bb_{chart_type_key}_{chart_config.get("id", "")}')
        if chart_config['bb_toggle']:
            with bb_cols[1].expander(f"Edit BB ({chart_type_key})"):
                if 'bb_settings' not in chart_config:
                    chart_config['bb_settings'] = {'window': 16, 'std_devs': 1}
                
                chart_config['bb_settings']['window'] = st.number_input(
                    'BB Length', min_value=2, max_value=100, value=chart_config['bb_settings']['window'], step=1, key=f'bb_len_{chart_type_key}_{chart_config.get("id", "")}'
                )
                chart_config['bb_settings']['std_devs'] = st.number_input(
                    'BB Std Devs', min_value=0.5, max_value=3.0, value=chart_config['bb_settings']['std_devs'], step=0.1, format="%.1f", key=f'bb_std_{chart_type_key}_{chart_config.get("id", "")}'
                )
        
        # EMA Toggles and Settings (Multiple EMAs)
        ema_general_cols = st.columns([0.6, 0.4])
        ema_general_toggle = ema_general_cols[0].checkbox('Show EMAs', value=any(e['enabled'] for e in chart_config.get('ema_toggles',[])), key=f'ema_general_{chart_type_key}_{chart_config.get("id", "")}')
        
        if 'ema_toggles' not in chart_config:
            chart_config['ema_toggles'] = []

        if ema_general_toggle:
            with ema_general_cols[1].expander(f"Edit EMAs ({chart_type_key})"):
                if not chart_config['ema_toggles']:
                    st.info("No EMAs added yet. Click 'Add New EMA'.")
                
                for i, ema_conf in enumerate(chart_config['ema_toggles']):
                    ema_row_cols = st.columns([0.6, 0.3, 0.1])
                    ema_conf['enabled'] = ema_row_cols[0].checkbox(f'EMA {ema_conf["id"]}', value=ema_conf['enabled'], key=f'ema_toggle_{chart_type_key}_{chart_config.get("id", "")}_{ema_conf["id"]}')
                    ema_conf['window'] = ema_row_cols[1].number_input(f'Length {ema_conf["id"]}', min_value=2, max_value=100, value=ema_conf['window'], step=1, key=f'ema_len_{chart_type_key}_{chart_config.get("id", "")}_{ema_conf["id"]}')
                    if ema_row_cols[2].button('X', key=f'remove_ema_{chart_type_key}_{chart_config.get("id", "")}_{ema_conf["id"]}'):
                        chart_config['ema_toggles'].pop(i)
                        if chart_type_key == "custom":
                            save_json_file(st.session_state.charts, CHART_CONFIG_FILE)
                        elif chart_type_key == "oddeven":
                            save_json_file(st.session_state.oddeven_config, ODDEVEN_CHART_CONFIG_FILE)
                        st.rerun()

                if st.button(f"Add New EMA ({chart_type_key})", key=f'add_ema_{chart_type_key}_{chart_config.get("id", "")}'):
                    if 'next_ema_id' not in chart_config or chart_config['next_ema_id'] is None:
                        chart_config['next_ema_id'] = (max([e['id'] for e in chart_config['ema_toggles']]) + 1) if chart_config['ema_toggles'] else 1
                    chart_config['ema_toggles'].append({'id': chart_config['next_ema_id'], 'enabled': True, 'window': 16})
                    chart_config['next_ema_id'] += 1
                    if chart_type_key == "custom":
                        save_json_file(st.session_state.charts, CHART_CONFIG_FILE)
                    elif chart_type_key == "oddeven":
                        save_json_file(st.session_state.oddeven_config, ODDEVEN_CHART_CONFIG_FILE)
                    st.rerun()

        chart_config['rsi_toggle'] = st.checkbox('RSI', value=chart_config.get('rsi_toggle', True), key=f'rsi_{chart_type_key}_{chart_config.get("id", "")}')
        chart_config['peaks_toggle'] = st.checkbox('Peaks/Troughs', value=chart_config.get('peaks_toggle', True), key=f'peaks_{chart_type_key}_{chart_config.get("id", "")}')
        chart_config['dots_toggle'] = st.checkbox('Dots', value=chart_config.get('dots_toggle', True), key=f'dots_{chart_type_key}_{chart_config.get("id", "")}')

        # Only custom charts have a 'Remove Chart' button
        if chart_type_key == "custom" and st.button("Remove Chart", key=f'remove_chart_{chart_config["id"]}', help="Removes this chart"):
            st.session_state.charts = [c for c in st.session_state.charts if c['id'] != chart_config['id']]
            save_json_file(st.session_state.charts, CHART_CONFIG_FILE)
            st.rerun()

    st.markdown("---")

    if spins_data.size > 0: # Check if there are spins to plot
        # Determine how to get cumulative scores based on chart type
        if chart_type_key == "custom":
            encoded_values = [encode_roulette(n, chart_config['ranges']) for n in spins_data]
            cumulative_scores = np.cumsum(encoded_values)
        elif chart_type_key == "oddeven":
            # Pass the current state and get updated state back
            scores, updated_oddeven_state = calculate_odd_even_scores(spins_data, st.session_state.oddeven_config)
            cumulative_scores = np.cumsum(scores)
            # Store the updated state back for next rerun.
            # This is critical for the 'streak' logic to persist across spins.
            st.session_state.oddeven_config = updated_oddeven_state
            save_json_file(st.session_state.oddeven_config, ODDEVEN_CHART_CONFIG_FILE)
            # Pass a copy of the current state to the chart_config for display in col1
            chart_config['oddeven_state_info'] = updated_oddeven_state.copy()


        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(cumulative_scores, label='Cumulative Score')

        # EMA Plots
        if chart_config.get('ema_toggles') and any(e['enabled'] for e in chart_config['ema_toggles']):
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
            if len(cumulative_scores) > 1:
                peaks, _ = find_peaks(cumulative_scores, distance=5, prominence=10)
                troughs, _ = find_peaks(-np.array(cumulative_scores), distance=5, prominence=10)
                ax.scatter(peaks, np.array(cumulative_scores)[peaks], color='green', marker='^', s=80, label='Local Highs')
                ax.scatter(troughs, np.array(cumulative_scores)[troughs], color='blue', marker='v', s=80, label='Local Lows')

        # Dots
        if chart_config['dots_toggle']:
            ax.scatter(range(len(cumulative_scores)), cumulative_scores, color='red', s=20, alpha=0.7, label='Spin Dots')

        ax.set_title(title)
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
                st.info(f"RSI for {title} cannot be computed with current data.")
    else:
        st.info(f"No spins recorded yet for {title}. Add a spin using the sidebar input.")


# --- Main App Layout ---
st.markdown("---")

# Render Custom Encoding Charts
st.header("Custom Encoded Charts")
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
    # We pass 'spins' as np.asarray to ensure consistency
    render_indicator_chart(
        f"Roulette Chart #{chart_conf['id']} (Custom Encoding)",
        np.asarray(st.session_state.spins),
        chart_conf,
        chart_type_key="custom" # Indicate it's a custom chart
    )
    st.markdown("---")

# Save all custom chart configurations at the end of the loop
save_json_file(st.session_state.charts, CHART_CONFIG_FILE)

# --- Render Fixed Odd/Even Chart ---
st.header("Odd/Even Streak Chart")

# Calculate Odd/Even counts (excluding 0)
num_odds = sum(1 for n in st.session_state.spins if n != 0 and n % 2 == 1)
num_evens = sum(1 for n in st.session_state.spins if n != 0 and n % 2 == 0)
num_zeros = st.session_state.spins.count(0)

st.write(f"**Odd numbers:** {num_odds} | **Even numbers:** {num_evens} | **Zeros:** {num_zeros}")

# We pass st.session_state.spins directly, and oddeven_config will manage its internal state
render_indicator_chart(
    "Roulette Chart (Odd/Even Streak Scoring)",
    np.asarray(st.session_state.spins), # Pass spins as numpy array
    st.session_state.oddeven_config, # Pass its dedicated config
    chart_type_key="oddeven" # Indicate it's the odd/even chart
)

# Save the odd/even chart configuration at the end
save_json_file(st.session_state.oddeven_config, ODDEVEN_CHART_CONFIG_FILE)
