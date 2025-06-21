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
    # Check if spin_str is a valid integer between 0 and 36
    try:
        spin_val = int(spin_str)
        valid_spin = 0 <= spin_val <= 36
    except ValueError:
        valid_spin = False
    add = st.button("Add Spin", disabled=not valid_spin)
with col2:
    reset = st.button("Reset All")

if reset:
    st.session_state.spins = []
    st.session_state.encoded_values = []
    st.session_state.cumulative_scores = []

if add and valid_spin:
    spin = int(spin_str)
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


# import streamlit as st
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# # Session state initialization
# if 'spins' not in st.session_state:
#     st.session_state.spins = []

# if 'encoders' not in st.session_state:
#     st.session_state.encoders = {}

# if 'next_encoder_id' not in st.session_state:
#     st.session_state.next_encoder_id = 1

# def encode_roulette(n, encoding_rules):
#     """Apply encoding rules to a roulette number"""
#     for rule in encoding_rules:
#         if rule['min'] <= n <= rule['max']:
#             if rule['type'] == 'multiply':
#                 return n * rule['value']
#             elif rule['type'] == 'add':
#                 return n + rule['value']
#             elif rule['type'] == 'subtract':
#                 return n - rule['value']
#             elif rule['type'] == 'negate':
#                 return -n
#             elif rule['type'] == 'fixed':
#                 return rule['value']
#     return n  # Default if no rule matches

# def calculate_cumulative_scores(spins, encoding_rules):
#     """Calculate cumulative scores for a list of spins"""
#     if not spins:
#         return []
    
#     encoded_values = [encode_roulette(spin, encoding_rules) for spin in spins]
#     cumulative_scores = []
#     cumulative_sum = 0
    
#     for encoded_val in encoded_values:
#         cumulative_sum += encoded_val
#         cumulative_scores.append(cumulative_sum)
    
#     return cumulative_scores, encoded_values

# st.title("Multi-Encoder Roulette Tracker")

# # Sidebar for encoder management
# st.sidebar.header("Encoder Management")

# # Add new encoder
# with st.sidebar.expander("➕ Add New Encoder", expanded=False):
#     encoder_name = st.text_input("Encoder Name:", key="new_encoder_name")
    
#     if st.button("Create Encoder"):
#         if encoder_name and encoder_name not in st.session_state.encoders:
#             st.session_state.encoders[encoder_name] = {
#                 'id': st.session_state.next_encoder_id,
#                 'rules': [],
#                 'color': f'C{len(st.session_state.encoders) % 10}'
#             }
#             st.session_state.next_encoder_id += 1
#             st.success(f"Encoder '{encoder_name}' created!")
#             st.rerun()
#         elif encoder_name in st.session_state.encoders:
#             st.error("Encoder name already exists!")

# # Manage existing encoders
# if st.session_state.encoders:
#     st.sidebar.subheader("Existing Encoders")
    
#     for encoder_name in list(st.session_state.encoders.keys()):
#         with st.sidebar.expander(f"⚙️ {encoder_name}", expanded=False):
            
#             # Display current rules
#             if st.session_state.encoders[encoder_name]['rules']:
#                 st.write("**Current Rules:**")
#                 for i, rule in enumerate(st.session_state.encoders[encoder_name]['rules']):
#                     rule_text = f"{rule['min']}-{rule['max']}: "
#                     if rule['type'] == 'multiply':
#                         rule_text += f"× {rule['value']}"
#                     elif rule['type'] == 'add':
#                         rule_text += f"+ {rule['value']}"
#                     elif rule['type'] == 'subtract':
#                         rule_text += f"- {rule['value']}"
#                     elif rule['type'] == 'negate':
#                         rule_text += "negate"
#                     elif rule['type'] == 'fixed':
#                         rule_text += f"= {rule['value']}"
                    
#                     col1, col2 = st.columns([3, 1])
#                     col1.write(rule_text)
#                     if col2.button("🗑️", key=f"del_rule_{encoder_name}_{i}"):
#                         st.session_state.encoders[encoder_name]['rules'].pop(i)
#                         st.rerun()
            
#             # Add new rule
#             st.write("**Add Rule:**")
#             col1, col2 = st.columns(2)
#             with col1:
#                 min_val = st.number_input("Min:", 0, 36, key=f"min_{encoder_name}")
#                 max_val = st.number_input("Max:", min_val, 36, min_val, key=f"max_{encoder_name}")
            
#             with col2:
#                 rule_type = st.selectbox(
#                     "Type:", 
#                     ['multiply', 'add', 'subtract', 'negate', 'fixed'],
#                     key=f"type_{encoder_name}"
#                 )
                
#                 if rule_type != 'negate':
#                     rule_value = st.number_input(
#                         "Value:", 
#                         value=1.0 if rule_type == 'multiply' else 0.0,
#                         key=f"value_{encoder_name}"
#                     )
#                 else:
#                     rule_value = 0
            
#             if st.button("Add Rule", key=f"add_rule_{encoder_name}"):
#                 new_rule = {
#                     'min': int(min_val),
#                     'max': int(max_val),
#                     'type': rule_type,
#                     'value': rule_value
#                 }
#                 st.session_state.encoders[encoder_name]['rules'].append(new_rule)
#                 st.rerun()
            
#             # Color picker
#             color = st.color_picker(
#                 "Chart Color:", 
#                 value=st.session_state.encoders[encoder_name].get('color', '#1f77b4'),
#                 key=f"color_{encoder_name}"
#             )
#             st.session_state.encoders[encoder_name]['color'] = color
            
#             # Delete encoder
#             if st.button(f"🗑️ Delete {encoder_name}", key=f"del_encoder_{encoder_name}"):
#                 del st.session_state.encoders[encoder_name]
#                 st.rerun()

# # Main content area
# col1, col2, col3 = st.columns([2, 1, 1])

# with col1:
#     spin_str = st.text_input("Enter spin (0-36):", value="")
#     try:
#         spin_val = int(spin_str)
#         valid_spin = 0 <= spin_val <= 36
#     except ValueError:
#         valid_spin = False

# with col2:
#     add = st.button("Add Spin", disabled=not valid_spin)

# with col3:
#     reset = st.button("Reset Spins")

# if reset:
#     st.session_state.spins = []

# if add and valid_spin:
#     spin = int(spin_str)
#     st.session_state.spins.append(spin)

# # Display current spins
# if st.session_state.spins:
#     st.subheader(f"Spins ({len(st.session_state.spins)})")
    
#     # Show recent spins
#     recent_spins = st.session_state.spins[-20:]  # Show last 20
#     st.write("Recent spins: " + " → ".join(map(str, recent_spins)))
    
#     if len(st.session_state.spins) > 20:
#         st.write(f"... and {len(st.session_state.spins) - 20} more")

# # Charts section
# if st.session_state.spins and st.session_state.encoders:
#     st.subheader("Encoder Charts")
    
#     # Chart display options
#     chart_cols = st.columns([1, 1, 1])
#     with chart_cols[0]:
#         show_individual = st.checkbox("Individual Charts", value=True)
#     with chart_cols[1]:
#         show_combined = st.checkbox("Combined Chart", value=True)
#     with chart_cols[2]:
#         show_stats = st.checkbox("Show Statistics", value=True)
    
#     # Individual charts
#     if show_individual:
#         for encoder_name, encoder_data in st.session_state.encoders.items():
#             if encoder_data['rules']:  # Only show if rules exist
#                 cumulative_scores, encoded_values = calculate_cumulative_scores(
#                     st.session_state.spins, encoder_data['rules']
#                 )
                
#                 if cumulative_scores:
#                     fig, ax = plt.subplots(figsize=(10, 4))
#                     ax.plot(cumulative_scores, 
#                            color=encoder_data['color'], 
#                            label=f'{encoder_name}',
#                            linewidth=2)
#                     ax.scatter(range(len(cumulative_scores)), 
#                              cumulative_scores, 
#                              color=encoder_data['color'], 
#                              s=20, alpha=0.7)
#                     ax.set_xlabel("Spin Number")
#                     ax.set_ylabel("Cumulative Score")
#                     ax.set_title(f"Cumulative Score - {encoder_name}")
#                     ax.grid(True, alpha=0.3)
#                     ax.legend()
                    
#                     # Add current score annotation
#                     current_score = cumulative_scores[-1]
#                     ax.annotate(f'Current: {current_score}', 
#                                xy=(len(cumulative_scores)-1, current_score),
#                                xytext=(10, 10), textcoords='offset points',
#                                bbox=dict(boxstyle='round,pad=0.3', 
#                                        facecolor=encoder_data['color'], 
#                                        alpha=0.3))
                    
#                     st.pyplot(fig)
#                     plt.close()
    
#     # Combined chart
#     if show_combined and len(st.session_state.encoders) > 1:
#         fig, ax = plt.subplots(figsize=(12, 6))
        
#         for encoder_name, encoder_data in st.session_state.encoders.items():
#             if encoder_data['rules']:
#                 cumulative_scores, _ = calculate_cumulative_scores(
#                     st.session_state.spins, encoder_data['rules']
#                 )
                
#                 if cumulative_scores:
#                     ax.plot(cumulative_scores, 
#                            color=encoder_data['color'], 
#                            label=encoder_name,
#                            linewidth=2)
#                     ax.scatter(range(len(cumulative_scores)), 
#                              cumulative_scores, 
#                              color=encoder_data['color'], 
#                              s=15, alpha=0.6)
        
#         ax.set_xlabel("Spin Number")
#         ax.set_ylabel("Cumulative Score")
#         ax.set_title("Combined Cumulative Scores")
#         ax.grid(True, alpha=0.3)
#         ax.legend()
#         st.pyplot(fig)
#         plt.close()
    
#     # Statistics table
#     if show_stats:
#         st.subheader("Statistics")
#         stats_data = []
        
#         for encoder_name, encoder_data in st.session_state.encoders.items():
#             if encoder_data['rules']:
#                 cumulative_scores, encoded_values = calculate_cumulative_scores(
#                     st.session_state.spins, encoder_data['rules']
#                 )
                
#                 if cumulative_scores:
#                     stats_data.append({
#                         'Encoder': encoder_name,
#                         'Current Score': cumulative_scores[-1],
#                         'Max Score': max(cumulative_scores),
#                         'Min Score': min(cumulative_scores),
#                         'Average per Spin': np.mean(encoded_values),
#                         'Volatility': np.std(encoded_values)
#                     })
        
#         if stats_data:
#             df = pd.DataFrame(stats_data)
#             st.dataframe(df, use_container_width=True)

# elif not st.session_state.encoders:
#     st.info("👈 Create an encoder in the sidebar to get started!")
# elif not st.session_state.spins:
#     st.info("Add some spins to see the charts!")

# # Quick start templates
# with st.expander("📋 Quick Start Templates"):
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         if st.button("Classic Encoder"):
#             st.session_state.encoders["Classic"] = {
#                 'id': st.session_state.next_encoder_id,
#                 'rules': [
#                     {'min': 0, 'max': 25, 'type': 'negate', 'value': 0},
#                     {'min': 26, 'max': 36, 'type': 'add', 'value': 10}
#                 ],
#                 'color': '#1f77b4'
#             }
#             st.session_state.next_encoder_id += 1
#             st.rerun()
    
#     with col2:
#         if st.button("Simple Positive"):
#             st.session_state.encoders["Simple Positive"] = {
#                 'id': st.session_state.next_encoder_id,
#                 'rules': [
#                     {'min': 0, 'max': 36, 'type': 'multiply', 'value': 1}
#                 ],
#                 'color': '#ff7f0e'
#             }
#             st.session_state.next_encoder_id += 1
#             st.rerun()
    
#     with col3:
#         if st.button("High/Low Split"):
#             st.session_state.encoders["High/Low"] = {
#                 'id': st.session_state.next_encoder_id,
#                 'rules': [
#                     {'min': 0, 'max': 18, 'type': 'fixed', 'value': -1},
#                     {'min': 19, 'max': 36, 'type': 'fixed', 'value': 1}
#                 ],
#                 'color': '#2ca02c'
#             }
#             st.session_state.next_encoder_id += 1
#             st.rerun()