mkdir -p Window.app/Contents/MacOS

Clang++ -o Window.app/Contents/MacOS/Window 07_ColorTriangleSquare.mm -framework Cocoa -framework QuartzCore -framework OpenGL

