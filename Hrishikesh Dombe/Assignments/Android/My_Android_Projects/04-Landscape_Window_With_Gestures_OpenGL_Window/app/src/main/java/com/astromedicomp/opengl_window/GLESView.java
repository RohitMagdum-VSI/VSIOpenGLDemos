package com.astromedicomp.opengl_window;

import android.content.Context; //For drawing context related
import android.opengl.GLSurfaceView; //For OpenGL Surface View and all related
import javax.microedition.khronos.opengles.GL10; //For OpenGLES 1.0 needed as param type GL10
import javax.microedition.khronos.egl.EGLConfig; //For EGLConfig needed as param type EGLConfig
import android.opengl.GLES30; // For OpenGLES 3.0
import android.view.Gravity;
import android.view.MotionEvent; // For "MotionEvent"
import android.view.GestureDetector; // For GestureDetector
import android.view.GestureDetector.OnGestureListener; // OnGestureListener
import android.view.GestureDetector.OnDoubleTapListener; // OnDoubleTapListener

//A View for OpenGLES3 graphics which also receives touch events
public class GLESView extends GLSurfaceView implements GLSurfaceView.Renderer, OnGestureListener, OnDoubleTapListener
{
	private final Context context;

	private GestureDetector gestureDetector;

	public GLESView(Context drawingContext)
	{
		super(drawingContext);

		context = drawingContext;

		//Accordingly set EGLContext to current supported version of OpenGL-ES
		setEGLContextClientVersion(3);

		//Set Renderer for drawing on GLSurfaceView
		setRenderer(this);

		//Render the view only when there is a change in drawing data
		setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);

		//this means 'handle' i.e who is going to handle 
		gestureDetector = new GestureDetector(context,this,null,false);
	
		//this means 'handle' i.e who is going to handle
		gestureDetector.setOnDoubleTapListener(this);
	}

	//overriden method of GLSurfaceView.Renderer(Init Code)
	@Override
	public void onSurfaceCreated(GL10 gl,EGLConfig config)
	{
		//OpenGL-ES version check
		String version = gl.glGetString(GL10.GL_VERSION);
		System.out.println("HAD: "+version);//"+" for concatination

		initialize(gl);
	}

	//overriden method of GLSurfaceView.Renderer(Change Size Code)
	@Override
	public void onSurfaceChanged(GL10 unused,int width,int height)
	{
		resize(width,height);
	}

	//overriden method of GLSurfaceView.Renderer(Rendering Code)
	@Override
	public void onDrawFrame(GL10 unused)
	{
		draw();
	}

	//Handling 'onTouchEvent' Is The Most IMPORTANT, Because It Triggers All Gesture And Events
	@Override
	public boolean onTouchEvent(MotionEvent e)
	{
		//code
		int eventaction=e.getAction();
		if(!gestureDetector.onTouchEvent(e))
			super.onTouchEvent(e);
		return(true);
	}

	//abstract method from OnDoubleTapListener so must be implemented
	@Override
	public boolean onDoubleTap(MotionEvent e)
	{
		System.out.println("HAD: "+"Double Tap");
		return(true);
	}

	//abstract method from OnDoubleTapListener so must be implemented
	@Override
	public boolean onDoubleTapEvent(MotionEvent e)
	{
		//Do not write any code here because already written 'onDOubleTap'
		return(true);
	}

	//abstract method from OnDoubleTapListener so must be implemented
	@Override
	public boolean onSingleTapConfirmed(MotionEvent e)
	{
		System.out.println("HAD: "+"Single Tap");
		return(true);
	}

	//abstract method from OnGestureListener so must be implemented
	@Override
	public boolean onDown(MotionEvent e)
	{
		//Do not write any code here because already written 'onSingleTapConfirmed'
		return(true);
	}

	//abstract method from OnGestureListener so must be implemented
	@Override
	public boolean onFling(MotionEvent e1,MotionEvent e2,float velocityX,float velocityY)
	{
		return(true);
	}

	//abstract method from OnGestureListener so must be implemented
	@Override
	public void onLongPress(MotionEvent e)
	{
		System.out.println("HAD: "+"Long Press");
	}

	//abstract method from OnGestureListener so must be implemented
	@Override
	public boolean onScroll(MotionEvent e1,MotionEvent e2,float distanceX,float distanceY)
	{
		System.out.println("HAD: "+"Scroll");
		System.exit(0);
		return(true);
	}

	//abstract method from OnGestureListener so must be implemented
	@Override
	public void onShowPress(MotionEvent e)
	{

	}

	//abstract method from OnGestureListener so must be implemented
	@Override
	public boolean onSingleTapUp(MotionEvent e)
	{
		return(true);
	}

	private void initialize(GL10 gl)
	{
		//Set the background frame color
		GLES30.glClearColor(0.0f,0.0f,1.0f,1.0f);
	}

	private void resize(int width,int height)
	{
		//Adjust the viewport based on geometry changes such as screen rotation
		GLES30.glViewport(0,0,width,height);
	}

	public void draw()
	{
		//Draw background color
		GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT|GLES30.GL_DEPTH_BUFFER_BIT);
	}

}