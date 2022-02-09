mkdir -p Window.app/Contents/MacOS

Clang++ -o Window.app/Contents/MacOS/Window 04_PerspectiveTriangle.mm -framework Cocoa -framework QuartzCore -framework OpenGL

