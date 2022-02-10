package com.jayshree.ndkoglthreedtextureandroid;

public class GLESNativeLib{
    static{
        System.loadLibrary("native-lib");
    }

    public static native void init();
    public static native void resize(int width, int height);
    public static native void display();
    public static native void loadBitMap(int[] pixels, int width, int height, String texture);
}
