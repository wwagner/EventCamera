# ImageJ Integration Setup Guide

This guide explains how to set up real-time streaming from the Event Camera application to ImageJ/Fiji for image processing and analysis.

## What is ImageJ?

ImageJ is a powerful, open-source image processing program developed at the National Institutes of Health. Fiji is an enhanced distribution of ImageJ that comes pre-configured with many useful plugins.

## Installation

### Step 1: Download Fiji

1. Go to https://fiji.sc/
2. Click the download button for Windows
3. Extract the downloaded ZIP file to a location on your computer (e.g., `C:\Program Files\Fiji`)
4. No installation required - Fiji runs directly from the extracted folder

### Step 2: Configure Event Camera Streaming

The Event Camera application is already configured to stream to ImageJ! The settings are in `event_config.ini`:

```ini
# ImageJ streaming settings
imagej_streaming_enabled = 0                           # Enable real-time streaming (0=off, 1=on)
imagej_stream_fps = 10                                  # Frames per second to stream
imagej_stream_directory = C:\Users\wolfw\OneDrive\Desktop\imagej_stream
imagej_max_stream_files = 100                          # Maximum files to keep (auto-cleanup)
```

**You can adjust these settings:**
- Change `imagej_stream_fps` to control how many frames per second are streamed (1-30)
- Change `imagej_stream_directory` to set where images are saved
- Change `imagej_max_stream_files` to control how many images are kept before old ones are deleted

## Usage

### Quick Start

1. **Start Fiji:**
   - Navigate to your Fiji installation folder
   - Double-click `ImageJ-win64.exe` (or `ImageJ-win32.exe` for 32-bit)

2. **Load the monitoring macro:**
   - In Fiji, go to: **Plugins → Macros → Edit...**
   - Click **File → Open** and select `imagej_stream_monitor.ijm` from the Event Camera repository
   - Click **Run** at the bottom of the macro editor window

3. **Start Event Camera streaming:**
   - Run the Event Camera application (`run_event_camera.bat`)
   - In the Camera Settings panel, find the "ImageJ Streaming" section
   - Check the "Enable Streaming" checkbox

4. **Watch the stream:**
   - ImageJ will now display incoming frames in real-time
   - The console will show which frames are being processed
   - Press **ESC** in ImageJ to stop monitoring

### Alternative: Enable Streaming from Config

You can also enable streaming by editing `event_config.ini`:

```ini
imagej_streaming_enabled = 1    # Change 0 to 1 to enable
```

Then restart the Event Camera application.

## Image Processing

### Adding Custom Processing

The macro file `imagej_stream_monitor.ijm` has a "CUSTOM PROCESSING SECTION" where you can add your own image processing code. Here are some examples:

**Example 1: Auto-adjust brightness**
```javascript
run("Enhance Contrast", "saturated=0.35");
```

**Example 2: Edge detection**
```javascript
run("Find Edges");
```

**Example 3: Apply threshold**
```javascript
setAutoThreshold("Default dark");
run("Convert to Mask");
```

**Example 4: Gaussian blur**
```javascript
run("Gaussian Blur...", "sigma=2");
```

**Example 5: Measure image properties**
```javascript
run("Set Measurements...", "area mean centroid");
run("Measure");
```

Simply uncomment (remove the `//`) the examples you want to use, or add your own ImageJ macro commands.

### Recording Your Own Macros

To create your own processing pipeline:

1. In Fiji, go to **Plugins → Macros → Record...**
2. Perform the processing steps you want on a sample image
3. The macro recorder will show the equivalent macro commands
4. Copy these commands into the "CUSTOM PROCESSING SECTION" of `imagej_stream_monitor.ijm`

## Troubleshooting

### "Directory not found" error
- Make sure the `imagej_stream_directory` in `event_config.ini` exists or will be created
- The application will automatically create the directory when streaming starts

### No images appearing in ImageJ
1. Check that streaming is enabled in the Event Camera UI
2. Verify the stream directory path in both `event_config.ini` and `imagej_stream_monitor.ijm` match
3. Look in the Event Camera console output for streaming messages
4. Check if PNG files are being created in the stream directory

### ImageJ is too slow
- Reduce the `imagej_stream_fps` setting (try 5 FPS or 1 FPS)
- Reduce the complexity of your image processing pipeline
- Close other programs to free up CPU resources

### Stream files filling up disk space
- Reduce `imagej_max_stream_files` to keep fewer files
- The application automatically deletes old files, keeping only the most recent ones

## Advanced Usage

### Batch Processing Multiple Streams

You can run multiple instances of the macro to monitor different directories:

1. Modify `imagej_stream_monitor.ijm` to change `streamDir`
2. Save as a new file (e.g., `imagej_stream_monitor_2.ijm`)
3. Run both macros simultaneously in Fiji

### Saving Processed Results

Add this to the macro after your processing:

```javascript
// Save processed image
saveAs("PNG", "C:\\Users\\wolfw\\Desktop\\processed\\processed_" + latestFile);
```

### Integration with Other Tools

The stream directory can be monitored by other applications too:
- Python scripts using `watchdog` library
- MATLAB using `dir()` polling
- Any application that can read PNG files from a directory

## Resources

- **Fiji Documentation**: https://fiji.sc/
- **ImageJ Macro Language**: https://imagej.nih.gov/ij/developer/macro/macros.html
- **Built-in Fiji Plugins**: https://imagej.net/plugins/
- **ImageJ Forum**: https://forum.image.sc/

## Support

If you encounter issues:
1. Check the Event Camera console output for error messages
2. Verify the ImageJ console (Window → Console) for errors
3. Ensure both applications have read/write permissions to the stream directory
