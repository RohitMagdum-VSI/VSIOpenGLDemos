mkdir -p Window.app/Contents/MacOS

Clang -o Window.app/Contents/MacOS/Window 01_Window.m -framework Cocoa
