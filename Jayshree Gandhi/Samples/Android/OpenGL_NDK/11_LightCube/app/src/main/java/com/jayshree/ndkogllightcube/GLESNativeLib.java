package com.jayshree.ndkogllightcube;

public class GLESNativeLib{
    static{
        System.loadLibrary("native-lib");
    }

    public static native void init();
    public static native void resize(int width, int height);
    public static native void display();
}
