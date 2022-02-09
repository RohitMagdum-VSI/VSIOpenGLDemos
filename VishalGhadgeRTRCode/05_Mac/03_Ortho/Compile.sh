mkdir -p Window.app/Contents/MacOS

Clang++ -o Window.app/Contents/MacOS/Window 03_Ortho.mm -framework Cocoa -framework QuartzCore -framework OpenGL

