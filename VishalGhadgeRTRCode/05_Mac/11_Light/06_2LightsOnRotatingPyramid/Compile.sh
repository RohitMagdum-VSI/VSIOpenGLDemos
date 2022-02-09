mkdir -p Window.app/Contents/MacOS

Clang++ -o Window.app/Contents/MacOS/Window 06_2LightsOnRotatingPyramid.mm -framework Cocoa -framework QuartzCore -framework OpenGL

