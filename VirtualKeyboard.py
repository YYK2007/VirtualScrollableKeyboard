import tensorflow as tf
import pylsl
import numpy as np
from pylsl import StreamInlet, resolve_stream                  
from nltk import flatten
import dsp

import pygame
from pygame.locals import *
import pygame.font
import pygame.draw

from pynput.keyboard import Key, Controller
from timeit import default_timer as timer

confidence_threshold = 0.15   
letterSelector = 0 #to keep track of the letters
selector_x = 0  
letter_width = 40  # Adjust as needed
letters = "abcdefghijklmnopqrstuvwxyz"
selector_width = 5


def select_key(dir):
    global selector_x, letterSelector
    
    if dir == "L":
        selector_x = max(0, selector_x - letter_width)  # Move left, but stay within bounds
        if letterSelector > 0:
            letterSelector -= 1
    elif dir == "R":
        selector_x = min(len(letters) * letter_width - selector_width, selector_x + letter_width)  # Move right, but stay within bounds
        if letterSelector < 27:
            letterSelector += 1
    elif dir == "b":
       print(letters[letterSelector])
    elif dir == "-":
        pass


def features(raw_data):

    implementation_version = 4 # 4 is latest versions

    raw_data = np.array(raw_data)

    axes = ['TP9', 'AF7', 'AF8', 'TP10']                        # Axes names.
    sampling_freq = 250                                         # Sampling frequency of the data.

    #Parameters specific to the spectral analysis DSP block [Default Values].
    scale_axes = 1                                             
    input_decimation_ratio = 1                                  
    filter_type = 'none'                                        
    filter_cutoff = 0                                           
    filter_order = 0                                            
    analysis_type = 'FFT'    
    draw_graphs = False                                  

    # The following parameters only apply to FFT analysis type.  Even if you choose wavelet analysis, these parameters still need dummy values
    fft_length = 64                                             

    # Deprecated parameters. Only for backwards compatibility.  
    spectral_peaks_count = 0                                    
    spectral_peaks_threshold = 0                                
    spectral_power_edges = "0"                                 

    # Current FFT parameters
    do_log = True                                               # Log of the spectral powers from the FFT frames
    do_fft_overlap = True                                       # Overlap FFT frames by 50%.  If false, no overlap
    extra_low_freq = False                                      #Decimate the input window by 10 and perform another FFT on the decimated window.
                                                                # This is useful to extract low frequency data.  The features will be appended to the normal FFT features

    # These parameters only apply to Wavelet analysis type.  Even if you choose FFT analysis, these parameters still need dummy values
    wavelet_level = 2                                           # Level of wavelet decomposition
    wavelet = "rbio3.1"                                         # Wavelet kernel to use

    output = dsp.generate_features(implementation_version, draw_graphs, raw_data, axes, sampling_freq, scale_axes, input_decimation_ratio,
                        filter_type, filter_cutoff, filter_order, analysis_type, fft_length, spectral_peaks_count,
                        spectral_peaks_threshold, spectral_power_edges, do_log, do_fft_overlap,
                        wavelet_level, wavelet, extra_low_freq)


    return output["features"]

def draw_keyboard():
    # Key dimensions and spacing
    letter_height = 40
    margin = 5
    selector_color = (255, 0, 0)  # Red

    # Draw background rectangle
    pygame.draw.rect(screen, (255, 255, 255), (0, 0, letter_width * len(letters) + margin * 2, letter_height + margin * 2))

    # Draw letters
    for i, letter in enumerate(letters):
        font = pygame.font.Font(None, 32)
        text_surface = font.render(letter, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=(margin + letter_width * i + letter_width / 2, margin + letter_height / 2))
        screen.blit(text_surface, text_rect)

    # Draw selector at the current selector_x position
    pygame.draw.rect(screen, selector_color, (margin + selector_x, margin, selector_width, letter_height))
    

# Load the TensorFlow Lite model
model_path = "C:\\Users\\youss\\Desktop\\TKS - The Knowledge Society\\Focus\\Virtual Keyboard Replicate #2\\ei-virtualkeyboard-classifier-tensorflow-lite-float32-model.lite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print()
print(input_details)
print()
print(output_details)

# Connect to the LSL stream
streams = resolve_stream('type', 'EEG')                         # create a new inlet to read # from the stream
inlet = pylsl.stream_inlet(streams[0])

nr_samples = 1

#Creating the game 
pygame.init()
screen_width = 800  # Adjust as needed
screen_height = 400  # Adjust as needed
white = (255, 255, 255)
green = (0, 255, 0)
blue = (0, 0, 128)
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Virtual Keyboard")



while True:

    back_nr = left_nr = right_nr = blink_nr = uncertain_nr = 0
    
    screen.fill(white)

    for iter in range (nr_samples):
        all_samples = []
        for i in range (2000 // 4):                                                 # 2000 ms = 2 secs, 4 EEG-electrodes (channels)
            sample, timestamp = inlet.pull_sample()
            sample.pop()
            all_samples.append(sample)

        all_samples = flatten(all_samples)                                          
        all_samples = features(all_samples)

        input_samples = np.array(all_samples[:65], dtype=np.float32)
        input_samples = np.expand_dims(input_samples, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_samples)            # input_details[0]['index'] = the index which accepts the input
        interpreter.invoke()                                                        # run the inference

        output_data = interpreter.get_tensor(output_details[0]['index'])            # output_details[0]['index'] = the index which provides the input

        background  = output_data[0][0]
        right       = output_data[0][1]
        left        = output_data[0][2]
        blink       = output_data[0][3]
        
        if left >= confidence_threshold:
            predicted_key = "L"  # Adjust based on your model's output mapping
            select_key(predicted_key)
        elif right >= confidence_threshold:
            predicted_key = "R"  # Adjust based on your model's output mapping
            select_key(predicted_key)
        elif blink >= confidence_threshold:
            predicted_key = "b"  # Adjust based on your model's output mapping
            select_key(predicted_key)

    draw_keyboard()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
             pygame.quit()
             quit()
             

    pygame.display.flip()

    #print(f"Left: {left:.8f}  Background: {background:.8f}  Right: {right:.8f} Blink: {blink:.8f}")       # this is used to show the confidence level of each brain activity.
 