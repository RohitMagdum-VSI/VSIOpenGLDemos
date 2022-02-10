package com.jayshree.ndkoglthreedtextureandroid;

import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

public class BitmapHelper{
    private int[] pixels;
    private int width;
    private int height;

    public BitmapHelper(int resourceID, Resources resources){
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inScaled = false;

        //read the resources
        Bitmap bitmap = BitmapFactory.decodeResource(resources, resourceID, options);
        this.width = bitmap.getWidth();
        this.height = bitmap.getHeight();

        int nPixels = bitmap.getWidth() * bitmap.getHeight();
        int[] pixels_returned = new int[nPixels];
        bitmap.getPixels(pixels_returned, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        this.pixels = pixels_returned;
    }

    public int[] getPixels(){
        return this.pixels;
    }

    public int getWidth(){
        return this.width;
    }

    public int getHeight(){
        return this.height;
    }
}
