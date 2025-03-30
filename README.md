# Detect-head-motion-events
This is a graphical user interface to detect tail bouts and head motion events in embedded fish.

# Installation
Needed packages:
```sh
opencv,
Numpy,
Scipy,
Pandas,
pylab,
matplotlib,
tkinter，
sklearn
```

# User guide
After launching the GUI, the user must first select the `parent directory` containing 3 folders: 
- `temp_check_vector_motion`
- `sleap_tracked_results_h5`
- `ztrack_results_h5`.

Next, the user needs to input the target video name (e.g., `FLIR_2024-08-01_F4_01-T1-0000`). Once the path and video name are confirmed, the user should wait a few seconds for the program to automatically compute the initial versions of tail bouts and head motion events.


The visualization supports the following interactive operations:

- Zooming: Hold the Shift key and scroll the mouse wheel to adjust the x-axis range.

- Navigation: Drag the progress bar at the bottom of the GUI to pan the x-axis.

- Adjust line colors: Click `Visualize Settings` to choose a favorite line color for each subplot.

For parameter adjustments:

- Navigate to `Adjust Thresholds` in the lower menu bar to modify detection thresholds for tail bouts and head motion events independently. The initial threshold will be shown during the manual correction.

- Alternatively, hold Shift while left-clicking and dragging the green threshold line to update thresholds in real time.

For segment-specific operations:

- Select a segment: Left-click once to highlight it.

- Open context menu: Right-click to access options for deleting the segment or exporting a superimposed video clip.

- Export videos: Two pop-up windows will prompt the user to select the folder where the extracted frames of every video are saved in the folders named after the video names and the folder to save the video clips of current events.

After finalizing selections, click `Save Segments` in the menu bar to export all detected events as a JSON file. The saved events include:

- Eye convergence events

- Tail bouts

- Swimbladder motion events (side & top)

- Head motion events