mkdir -p Window.app/Contents/MacOS

Clang++ -o Window.app/Contents/MacOS/Window 05_MultiColorTriangle.mm -framework Cocoa -framework QuartzCore -framework OpenGL

