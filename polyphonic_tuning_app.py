import pyaudio
import numpy as np
import collections
import threading
import time
import sys
import pygame

pygame.init()

pygame.font.init()
arial = pygame.font.SysFont('arial', 14)

### Audio buffer code taken from https://medium.com/@pmlachert/audio-buffer-in-python-f17264ada064
class AudioBuffer:

    RATE = 44100
    CHUNK = int(RATE / 100)
    
    def __init__(self, chunks: int = 5) -> None:
        self.chunks = chunks
        self.stream = pyaudio.PyAudio().open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            )
        self.thread = threading.Thread(target=self._collect_data, daemon=True)
        self.frames = collections.deque(maxlen=self.chunks)
        self.current_time = time.time()
        self.delta_t = 1/AudioBuffer.RATE
        self.n = self.CHUNK * self.chunks
        self.interval = self.delta_t*self.n
        self.delta_f = 1/self.interval

    def __call__(self):
        return np.concatenate(self.frames)
    
    def __len__(self):
        return self.CHUNK * self.chunks
    
    def is_full(self):
        return len(self.frames) == self.chunks
    
    def start(self):
        self.thread.start()
        while not self.is_full(): # wait until the buffer is filled
            time.sleep(1)
        
    def _collect_data(self):
        while True:
            try:
                raw_data = self.stream.read(self.CHUNK)
                decoded = np.frombuffer(raw_data, np.int16)
                self.frames.append(decoded)
                self.current_time = time.time()
            except OSError:
                self.stream = pyaudio.PyAudio().open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.RATE,
                    input=True,
                    frames_per_buffer=self.CHUNK,
                    )


if __name__ == "__main__":
    audio_buffer = AudioBuffer()
    audio_buffer.start()

    max_freq = 2000
    plot_time = 5
    max_amplitude = 10000000
    display_exponent = 0.3
    n_shifters = 4
    window_dims = (800, 800)

    window = pygame.display.set_mode(window_dims)
    # Figure out where the green lines go
    reference_note_freqs = [440*2**(i/12) for i in range(-36, 36)]
    note_names = [["A","A#","B","C","C#","D","D#","E","F","F#","G","G#"][i%12] + str(i//12+4) for i in range(-36, 36)]
    note_name_images = [arial.render(note_name, False, (0, 255, 0)) for note_name in note_names]
    note_display_heights = [window_dims[1]-f/max_freq*window_dims[1] for f in reference_note_freqs]

    lines_graph = pygame.surfarray.make_surface(np.zeros([window_dims[0], window_dims[1], 3])).convert_alpha()
    lines_graph_empty = pygame.surfarray.make_surface(np.zeros([window_dims[0], window_dims[1], 3])).convert_alpha()
    spectrogram = pygame.surfarray.make_surface(128*np.ones([window_dims[0], window_dims[1], 3]))
    spectrogram_time = time.time()
    shift_arrays = np.e**(-2j*np.pi/n_shifters/audio_buffer.n*np.arange(n_shifters)[:,np.newaxis]*np.arange(audio_buffer.n))  # shifter, shift
    display_low_center = 0
    display_high_center = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Get the audio data, compute the frequency spectrum
        data = audio_buffer()
        new_spectrogram_time = time.time()
        interval = new_spectrogram_time - spectrogram_time
        pixel_interval = int(window_dims[0]*new_spectrogram_time/plot_time) - int(window_dims[0]*spectrogram_time/plot_time)
        spectrogram_time = new_spectrogram_time
        max_fourier_mode = int(max_freq/audio_buffer.delta_f*n_shifters)
        spectrum = (np.abs(np.fft.fft(data*shift_arrays).T.reshape([-1])[:max_fourier_mode])/max_amplitude)**2

        # Run k-means clustering to group the frequencies
        split = int(np.sum(np.arange(spectrum.shape[0])*spectrum)/np.sum(spectrum))
        for i in range(5):
            low_center = np.sum(np.arange(spectrum[:split].shape[0])*spectrum[:split])/np.sum(spectrum[:split])
            high_center = np.sum((split+np.arange(spectrum[split:].shape[0]))*spectrum[split:])/np.sum(spectrum[split:])
            split = int((low_center+high_center)/2)

        # Find the frequency mode in each cluster
        low_center, high_center = np.argmax(spectrum[:split]), split+np.argmax(spectrum[split:])

        # Fine tune the positions of the frequency modes
        f_interval = 2*n_shifters
        for i in range(5):
            range_bottom, range_top = max(0, int(low_center-f_interval-1)), min(max_fourier_mode, int(low_center+f_interval+1))
            kernel = np.cos(np.clip((range_bottom+np.arange(range_top-range_bottom)-low_center)/f_interval, -1, 1)*np.pi/2)**2
            low_center = np.sum(kernel*spectrum[range_bottom:range_top]*(range_bottom+np.arange(range_top-range_bottom)))/np.sum(kernel*spectrum[range_bottom:range_top])
        for i in range(5):
            range_bottom, range_top = max(0, int(high_center-f_interval-1)), min(max_fourier_mode, int(high_center+f_interval+1))
            kernel = np.cos(np.clip((range_bottom+np.arange(range_top-range_bottom)-high_center)/f_interval, -1, 1)*np.pi/2)**2
            high_center = np.sum(kernel*spectrum[range_bottom:range_top]*(range_bottom+np.arange(range_top-range_bottom)))/np.sum(kernel*spectrum[range_bottom:range_top])
        low_center, high_center = low_center/n_shifters*audio_buffer.delta_f, high_center/n_shifters*audio_buffer.delta_f

        # Roll the spectrogram a bit
        display_spectrum = np.tile(np.clip(spectrum[np.newaxis,:,np.newaxis]**display_exponent*255, 0, 255).astype(np.uint8), [1, 1, 3])
        new_spectrogram_section = pygame.transform.scale(pygame.surfarray.make_surface(display_spectrum[:,::-1,:]), (pixel_interval+1, window_dims[1]))
        spectrogram.blit(spectrogram, (-pixel_interval, 0))
        spectrogram.blit(new_spectrogram_section, (window_dims[0]-pixel_interval, 0))

        # Draw the next bit of the pink frequency tracking lines
        last_display_low_center = display_low_center
        last_display_high_center = display_high_center
        display_low_center = window_dims[1]-low_center/max_freq*window_dims[1]
        display_high_center = window_dims[1]-high_center/max_freq*window_dims[1]
        lines_graph_empty.fill((0, 0, 0, 0))
        lines_graph_empty.blit(lines_graph, (-pixel_interval, 0))
        lines_graph, lines_graph_empty = lines_graph_empty, lines_graph
        if abs(display_low_center-last_display_low_center) < 10:
            pygame.draw.line(lines_graph, (255, 0, 255), (window_dims[0]-1, display_low_center), (window_dims[0]-1-pixel_interval, last_display_low_center), 1)
        if abs(display_high_center-last_display_high_center) < 10:
            pygame.draw.line(lines_graph, (255, 0, 255), (window_dims[0]-1, display_high_center), (window_dims[0]-1-pixel_interval, last_display_high_center), 1)

        window.blit(spectrogram, (0, 0))
        for note_name_image, note_display_height in zip(note_name_images, note_display_heights):
            pygame.draw.line(window, (0, 255, 0), (0, note_display_height), (window_dims[0], note_display_height), 1)
            window.blit(note_name_image, (window_dims[0]-note_name_image.get_width()-100, note_display_height-int(note_name_image.get_height()/2)))
        window.blit(lines_graph, (0, 0))

        # Draw the measurements of note frequencies and interval sizes
        pygame.draw.line(window, (255, 0, 255), (window_dims[0], display_low_center), (window_dims[0]-30, display_low_center), 1)
        pygame.draw.line(window, (255, 0, 255), (window_dims[0], display_high_center), (window_dims[0]-30, display_high_center), 1)
        pygame.draw.line(window, (255, 0, 255), (window_dims[0]-15, display_low_center), (window_dims[0]-15, display_high_center), 1)
        low_note = np.log2(low_center/440)*12
        offset_low = round(100*(low_note-round(low_note)))/100
        low_offset_str = ("" if str(offset_low)[0] == "-" else "+") + str(offset_low)
        low_text = arial.render(low_offset_str, False, (255, 0, 255))
        high_note = np.log2(high_center/440)*12
        offset_high = round(100*(high_note-round(high_note)))/100
        high_offset_str = ("" if str(offset_high)[0] == "-" else "+") + str(offset_high)
        high_text = arial.render(high_offset_str, False, (255, 0, 255))
        interval_note = high_note-low_note
        offset_interval = round(100*interval_note)/100
        interval_text = arial.render(str(offset_interval), False, (255, 0, 255))
        window.blit(low_text, (window_dims[0]-70, display_low_center-int(low_text.get_height()/2)))
        window.blit(high_text, (window_dims[0]-70, display_high_center-int(high_text.get_height()/2)))
        window.blit(interval_text, (window_dims[0]-70, int((display_low_center+display_high_center)/2-interval_text.get_height()/2)))
        pygame.display.update()
