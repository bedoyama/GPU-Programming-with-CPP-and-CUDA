#!/bin/bash

# Script to download the actual stb_image libraries

echo "Downloading stb_image.h..."
curl -o stb_image.h https://raw.githubusercontent.com/nothings/stb/master/stb_image.h

echo "Downloading stb_image_write.h..."
curl -o stb_image_write.h https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h

echo "Download complete!"
echo ""
echo "The stb libraries have been downloaded successfully."
echo "You can now build the project with:"
echo "  mkdir build && cd build"
echo "  cmake .."
echo "  make"
