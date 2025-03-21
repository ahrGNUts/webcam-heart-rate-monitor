"""
Webcam Heart Rate Monitor
Gilad Oved
December 2018
"""

import numpy as np
import cv2
import sys


# Helper Methods
def buildGauss(frame, levels):
    pyramid = [frame]
    for level in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid


def reconstructFrame(pyramid, index, levels):
    filteredFrame = pyramid[index]
    for level in range(levels):
        filteredFrame = cv2.pyrUp(filteredFrame)
    # Ensure dimensions match by resizing if necessary
    h, w = filteredFrame.shape[:2]
    if h != videoHeight or w != videoWidth:
        filteredFrame = cv2.resize(filteredFrame, (videoWidth, videoHeight))
    return filteredFrame

# Webcam Parameters
webcam = None
if len(sys.argv) == 2:
    webcam = cv2.VideoCapture(sys.argv[1])
else:
    webcam = cv2.VideoCapture(0)

# Check if webcam is opened successfully
if not webcam.isOpened():
    print("Error: Could not open webcam.")
    sys.exit(1)

# Increase resolution for better visibility
realWidth = 640
realHeight = 480
# Make the detection region smaller
videoWidth = 240
videoHeight = 180
videoChannels = 3
videoFrameRate = 15
webcam.set(3, realWidth)
webcam.set(4, realHeight)

# Output Videos
if len(sys.argv) != 2:
    originalVideoFilename = "original.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    originalVideoWriter = cv2.VideoWriter(originalVideoFilename, fourcc, videoFrameRate, (realWidth, realHeight), True)

outputVideoFilename = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
outputVideoWriter = cv2.VideoWriter(outputVideoFilename, fourcc, videoFrameRate, (realWidth, realHeight), True)

# Check if video writers opened successfully
if len(sys.argv) != 2 and not originalVideoWriter.isOpened():
    print(f"Error: Could not create original video file {originalVideoFilename}")
    sys.exit(1)

if not outputVideoWriter.isOpened():
    print(f"Error: Could not create output video file {outputVideoFilename}")
    sys.exit(1)

# Color Magnification Parameters
levels = 3
alpha = 170
minFrequency = 1.0
maxFrequency = 2.0
bufferSize = 150
bufferIndex = 0

# Output Display Parameters
font = cv2.FONT_HERSHEY_SIMPLEX
loadingTextLocation = (20, 40)
bpmTextLocation = (realWidth//2 - 40, 40)
fontScale = 1.0
fontColor = (255, 255, 255)
lineType = 2
boxColor = (0, 255, 0)
boxWeight = 2

# Initialize Gaussian Pyramid
firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
firstGauss = buildGauss(firstFrame, levels+1)[levels]
videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
fourierTransformAvg = np.zeros(bufferSize)

# Bandpass Filter for Specified Frequencies
frequencies = (1.0*videoFrameRate) * np.arange(bufferSize) / (1.0*bufferSize)
mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

# Heart Rate Calculation Variables
bpmCalculationFrequency = 15
bpmBufferIndex = 0
bpmBufferSize = 10
bpmBuffer = np.zeros(bpmBufferSize)

i = 0
while (True):
    ret, frame = webcam.read()
    if ret == False:
        break

    if len(sys.argv) != 2:
        originalFrame = frame.copy()
        originalVideoWriter.write(originalFrame)

    # Calculate center region coordinates
    x1 = (realWidth - videoWidth) // 2
    y1 = (realHeight - videoHeight) // 2
    x2 = x1 + videoWidth
    y2 = y1 + videoHeight
    
    # Extract the detection frame from the center
    detectionFrame = frame[y1:y2, x1:x2, :]

    # Construct Gaussian Pyramid
    videoGauss[bufferIndex] = buildGauss(detectionFrame, levels+1)[levels]
    fourierTransform = np.fft.fft(videoGauss, axis=0)

    # Bandpass Filter
    fourierTransform[mask == False] = 0

    # Grab a Pulse
    if bufferIndex % bpmCalculationFrequency == 0:
        i = i + 1
        for buf in range(bufferSize):
            fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
        # Find the peak frequency and validate it
        peak_idx = np.argmax(fourierTransformAvg)
        hz = frequencies[peak_idx]
        bpm = 60.0 * hz
        # Only update BPM if within reasonable range (30-180 BPM)
        if 30 <= bpm <= 180:
            bpmBuffer[bpmBufferIndex] = bpm
            bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

    # Amplify
    filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
    filtered = filtered * alpha

    # Reconstruct Resulting Frame
    filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
    # Ensure frames have compatible types for addition
    filteredFrame = filteredFrame.astype(np.float32)
    detectionFrame = detectionFrame.astype(np.float32)
    outputFrame = detectionFrame + filteredFrame
    outputFrame = cv2.convertScaleAbs(outputFrame)

    bufferIndex = (bufferIndex + 1) % bufferSize

    # Update the frame with the processed region
    frame[y1:y2, x1:x2, :] = outputFrame
    
    # Draw rectangle around the region of interest
    cv2.rectangle(frame, (x1, y1), (x2, y2), boxColor, boxWeight)
    
    if i > bpmBufferSize:
        # Draw background for BPM text
        text = "BPM: %d" % int(bpmBuffer.mean())
        text_size = cv2.getTextSize(text, font, fontScale, lineType)[0]
        text_x = bpmTextLocation[0]
        text_y = bpmTextLocation[1]
        cv2.rectangle(frame, 
                     (text_x - 10, text_y - text_size[1] - 10),
                     (text_x + text_size[0] + 10, text_y + 10),
                     (0, 0, 0), -1)  # Black background
        cv2.putText(frame, text, bpmTextLocation, font, fontScale, fontColor, lineType)
    else:
        # Draw background for loading text
        text = "Calculating BPM..."
        text_size = cv2.getTextSize(text, font, fontScale, lineType)[0]
        text_x = loadingTextLocation[0]
        text_y = loadingTextLocation[1]
        cv2.rectangle(frame, 
                     (text_x - 10, text_y - text_size[1] - 10),
                     (text_x + text_size[0] + 10, text_y + 10),
                     (0, 0, 0), -1)  # Black background
        cv2.putText(frame, text, loadingTextLocation, font, fontScale, fontColor, lineType)

    outputVideoWriter.write(frame)

    if len(sys.argv) != 2:
        # Resize window for better visibility
        cv2.namedWindow("Webcam Heart Rate Monitor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Webcam Heart Rate Monitor", realWidth, realHeight)
        cv2.imshow("Webcam Heart Rate Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

webcam.release()
cv2.destroyAllWindows()
outputVideoWriter.release()
if len(sys.argv) != 2:
    originalVideoWriter.release()
