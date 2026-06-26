# src/config.py

RAW_NUMERIC_COLUMNS = [
    'hospital_load_pct',
    'cooling_load_tr',
    'power_kw',
    'kw_per_tr',
    'suction_pressure_bar',
    'discharge_pressure_bar',
    'evap_temp_c',
    'condensing_temp_c',
    'superheat_c',
    'subcooling_c',
    'refrigerant_charge_pct',
    'chw_supply_temp_c',
    'chw_return_temp_c',
    'chw_flow_m3h',
    'chw_dp_bar',
    'cw_inlet_temp_c',
    'cw_outlet_temp_c',
    'cw_flow_m3h',
    'current_a',
    'current_b',
    'current_c',
    'power_factor',
    'compressor_speed_pct',
    'guide_vane_pct',
    'oil_pressure_bar',
    'oil_temp_c',
    'bearing_temp_c',
    'motor_temp_c',
    'vibration_x',
    'vibration_y',
    'vibration_z',
    'pump_speed_pct',
    'pump_vibration_mm_s',
    'pump_discharge_bar',
    'tower_fan_speed_pct',
    'tower_approach_c',
    'sequence_efficiency_pct',
    'health_index',
    'delta_t_chw',
    'delta_t_cw',
    'compression_ratio',
    'power_per_flow',
    'vibration_total',
]

PRIMARY_FEATURES = [
    'power_kw',
    'cooling_load_tr',
    'kw_per_tr',
    'suction_pressure_bar',
    'discharge_pressure_bar',
    'compressor_speed_pct',
    'chw_supply_temp_c',
    'health_index',
]

PRIMARY_WEIGHT = 3
SECONDARY_WEIGHT = 1

MAX_LAG = 60

WINDOW_SIZE = 6
OVERLAP = 2
STEP = WINDOW_SIZE - OVERLAP

Z_POINT = 5.0
Z_ROLLING = 4.0
WINDOW_60 = 6

SIMILARITY_THRESHOLD = 0.90

POINT_RATIO_CRITICAL = 0.05
POINT_RATIO_WARNING = 0.01

HIST_ANOM_RATIO = 0.05
HIST_POINT_RATIO = 0.03

IF_SCORE_THRESHOLD = -0.05

CORRELATION_THRESHOLD = 0.40
LAG_THRESHOLD = 0.40
