# Video Background Extraction
Python program to extract the background from each frame of a video.

### Requirements
- python3
- openCV
- numpy
- statistics

### Code Flow
The frames of the video are extracted and stored.\
The mean, median and mode matrices are calculated from scratch and the meanFrame,
medianFrame and modeFrame are stored.\
From each frame, the mean, median and mode frame is subtracted, the difference frame is converted
to grayscale and otsu algorithm is run on it.\
The threshold received from otsu is used to extract the object and the background from each frame. The
results received by (frame-mean), (frame-median) and (frame-mode) for each frame are stored.
