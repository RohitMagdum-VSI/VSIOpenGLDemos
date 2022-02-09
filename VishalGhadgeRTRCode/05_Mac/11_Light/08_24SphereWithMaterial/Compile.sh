mkdir -p Window.app/Contents/MacOS

Clang++ -o Window.app/Contents/MacOS/Window 08_24SphereWithMaterial.mm -framework Cocoa -framework QuartzCore -framework OpenGL

