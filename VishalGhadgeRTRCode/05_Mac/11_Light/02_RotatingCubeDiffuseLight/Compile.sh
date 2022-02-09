mkdir -p Window.app/Contents/MacOS

Clang++ -o Window.app/Contents/MacOS/Window 02_RotatingCubeDiffuseLight.mm -framework Cocoa -framework QuartzCore -framework OpenGL

