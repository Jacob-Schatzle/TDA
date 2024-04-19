import json
import numpy as np
from ripser import ripser
import matplotlib.pyplot as plt
import persim as sim

# Set up parameters
sample_rate = 1000
dim = 2
use_higher_sample_rate_for_inputs = 0
tda_window = 1
number_of_sequence_inputs = 1

# Define the 
def read_and_clean_dataset(filename, sample_rate, use_higher_sample_rate_for_inputs):
    with open(filename, 'r') as file:
        data = json.load(file)
    
    for i in range(len(data['measured_pin_location'])):
        if np.isnan(data['measured_pin_location'][i]):
            data['measured_pin_location'][i] = (data['measured_pin_location'][i-1] + data['measured_pin_location'][i+1]) / 2

    # Determine overlapping time span for both signals
    latest_start_time = max(data['time_acceleration_data'][0], data['measured_pin_location_tt'][0])
    print(latest_start_time)
    earliest_end_time = min(data['time_acceleration_data'][-1], data['measured_pin_location_tt'][-1])
    print(earliest_end_time)

    # Trim signals
    clip_start = np.where(np.array(data['time_acceleration_data']) >= latest_start_time)[0][0]
    clip_end = np.where(np.array(data['time_acceleration_data']) >= earliest_end_time)[0][0]
    data['time_acceleration_data'] = data['time_acceleration_data'][clip_start:clip_end]
    data['acceleration_data'] = data['acceleration_data'][clip_start:clip_end]

    clip_start = np.where(np.array(data['measured_pin_location_tt']) >= latest_start_time)[0][0]
    clip_end = np.where(np.array(data['measured_pin_location_tt']) >= earliest_end_time)[0][0]
    data['measured_pin_location_tt'] = data['measured_pin_location_tt'][clip_start:clip_end]
    data['measured_pin_location'] = data['measured_pin_location'][clip_start:clip_end]

    # Create new time axes
    if use_higher_sample_rate_for_inputs:
        sample_rate_vib = sample_rate * number_of_sequence_inputs
    else:
        sample_rate_vib = sample_rate

    time_vibration = np.arange(data['time_acceleration_data'][0], data['time_acceleration_data'][-1], 1/sample_rate_vib)
    time_pin = np.arange(data['measured_pin_location_tt'][0], data['measured_pin_location_tt'][-1], 1/sample_rate)

    # Interpolate signals
    vibration_signal = np.interp(time_vibration, data['time_acceleration_data'], data['acceleration_data'])
    pin_position = np.interp(time_pin, data['measured_pin_location_tt'], data['measured_pin_location'])

    return time_vibration, vibration_signal, time_pin, pin_position

time_vibration, vibration_signal, time_pin, pin_position = read_and_clean_dataset('data_6_with_FFT.json', sample_rate, use_higher_sample_rate_for_inputs)

# Convert input data into moving window datapoints
vibration_signal_samples = np.zeros((len(vibration_signal) - 1, dim))
for i in range(len(vibration_signal_samples) - 1):
    vibration_signal_samples[i, :] = vibration_signal[i:i+dim]
    
persistence = ripser(vibration_signal_samples)
