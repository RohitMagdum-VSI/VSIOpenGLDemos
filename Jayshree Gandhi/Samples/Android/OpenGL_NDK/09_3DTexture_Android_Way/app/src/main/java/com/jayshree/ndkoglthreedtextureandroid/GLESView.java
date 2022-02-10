package com.jayshree.ndkoglthreedtextureandroid;

import android.content.Context;
import android.view.Gravity;
import android.graphics.Color;

import android.view.MotionEvent;
import android.view.GestureDetector;
import android.view.GestureDetector.OnGestureListener;
import android.view.GestureDetector.OnDoubleTapListener;

//for opengl
import android.opengl.GLSurfaceView;
import android.opengl.GLES32;

import javax.microedition.khronos.opengles.GL10;
import javax.microedition.khronos.egl.EGLConfig;

public class GLESView extends GLSurfaceView implements GestureDetector.OnGestureListener, GestureDetector.OnDoubleTapListener {

    private static String TAG = "GLESView";
    private GestureDetector gestureDetector;


    public GLESView(Context context) {
        super(context);
        //set EGLContext to current supported version of OpenGL-ES
        setEGLContextClientVersion(3);

        // set Renderer for drawing on the GLSurfaceView
        MyRenderer renderer = new MyRenderer();
        setRenderer(renderer);

        // Render the view only when there is a change in the drawing data
        //setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);

        gestureDetector = new GestureDetector(context, this, null, false); // this means 'handler' i.e. who is going to handle
        gestureDetector.setOnDoubleTapListener(this); // this means 'handler' i.e. who is going to handle
    }

    private static class MyRenderer implements GLSurfaceView.Renderer {

        @Override
        public void onDrawFrame(GL10 gl) {
            GLESNativeLib.display();
        }

        @Override
        public void onSurfaceChanged(GL10 gl, int width, int height) {
            GLESNativeLib.resize(width, height);
        }

        @Override
        public void onSurfaceCreated(GL10 gl, EGLConfig config) {
            GLESNativeLib.init();
        }
    }

    @Override
    public boolean onTouchEvent(MotionEvent e){
        int eventaction = e.getAction();
        if(!gestureDetector.onTouchEvent(e))
            super.onTouchEvent(e);
        return(true);
    }

    @Override
    public boolean onSingleTapConfirmed(MotionEvent e) {
        return false;
    }

    @Override
    public boolean onDoubleTap(MotionEvent e) {
        return false;
    }

    @Override
    public boolean onDoubleTapEvent(MotionEvent e) {
        return false;
    }

    @Override
    public boolean onDown(MotionEvent e) {
        return false;
    }

    @Override
    public void onShowPress(MotionEvent e) {

    }

    @Override
    public boolean onSingleTapUp(MotionEvent e) {
        return false;
    }

    @Override
    public boolean onScroll(MotionEvent e1, MotionEvent e2, float distanceX, float distanceY) {
        System.exit(0);
        return true;
    }

    @Override
    public void onLongPress(MotionEvent e) {

    }

    @Override
    public boolean onFling(MotionEvent e1, MotionEvent e2, float velocityX, float velocityY) {
        return false;
    }
}

