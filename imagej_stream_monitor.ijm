// ImageJ Macro: Event Camera Stream Monitor
// This macro monitors a directory for new images from the Event Camera application
// and displays them in real-time in ImageJ.
//
// USAGE:
// 1. Install Fiji/ImageJ from https://fiji.sc/
// 2. Open this macro in Fiji: Plugins > Macros > Edit...
// 3. Click "Run" to start monitoring
// 4. Enable streaming in the Event Camera application
//
// The macro will automatically:
// - Load new frames as they arrive
// - Display them in a window
// - Apply any processing you configure below
//

// Configuration
streamDir = "C:\\Users\\wolfw\\OneDrive\\Desktop\\imagej_stream\\";
refreshInterval = 100;  // milliseconds between checks
windowTitle = "Event Camera Live Stream";

// Initialize
print("\\Clear");
print("Event Camera Stream Monitor");
print("============================");
print("Monitoring directory: " + streamDir);
print("Refresh interval: " + refreshInterval + " ms");
print("");
print("To stop: Press ESC or click 'Kill' button in toolbar");
print("");

lastFile = "";
frameCount = 0;

// Main monitoring loop
while (true) {
    // Get list of PNG files in stream directory
    fileList = getFileList(streamDir);
    pngFiles = newArray();

    for (i = 0; i < fileList.length; i++) {
        if (endsWith(fileList[i], ".png")) {
            pngFiles = Array.concat(pngFiles, fileList[i]);
        }
    }

    // Sort to get the latest file
    if (pngFiles.length > 0) {
        Array.sort(pngFiles);
        latestFile = pngFiles[pngFiles.length - 1];
        fullPath = streamDir + latestFile;

        // Check if this is a new file
        if (latestFile != lastFile) {
            // Open the image
            open(fullPath);

            // Rename window for consistency
            rename(windowTitle);

            frameCount++;
            print("Frame " + frameCount + ": " + latestFile);

            // ========================================
            // CUSTOM PROCESSING SECTION
            // Add your image processing here!
            // ========================================

            // Example 1: Auto-adjust brightness/contrast
            // run("Enhance Contrast", "saturated=0.35");

            // Example 2: Apply Gaussian blur
            // run("Gaussian Blur...", "sigma=2");

            // Example 3: Find edges
            // run("Find Edges");

            // Example 4: Threshold
            // setAutoThreshold("Default dark");
            // run("Convert to Mask");

            // Example 5: Measure properties
            // run("Set Measurements...", "area mean standard modal min centroid center perimeter bounding fit shape feret's integrated median skewness kurtosis area_fraction stack display redirect=None decimal=3");
            // run("Measure");

            // ========================================

            lastFile = latestFile;
        }
    }

    // Wait before next check
    // To stop monitoring: Press ESC or click the "Kill" button in ImageJ toolbar
    wait(refreshInterval);
}

print("Stream monitoring ended");
print("Total frames processed: " + frameCount);
