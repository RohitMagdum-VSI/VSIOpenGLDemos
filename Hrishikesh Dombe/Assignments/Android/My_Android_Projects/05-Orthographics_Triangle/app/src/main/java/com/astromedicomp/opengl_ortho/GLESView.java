package com.astromedicomp.opengl_ortho;

import android.content.Context; //For drawing context related
import android.opengl.GLSurfaceView; //For OpenGL Surface View and all related
import javax.microedition.khronos.opengles.GL10; //For OpenGLES 1.0 needed as param type GL10
import javax.microedition.khronos.egl.EGLConfig; //For EGLConfig needed as param type EGLConfig
import android.opengl.GLES31; // For OpenGLES 3.2
import android.view.Gravity;
import android.view.MotionEvent; // For "MotionEvent"
import android.view.GestureDetector; // For GestureDetector
import android.view.GestureDetector.OnGestureListener; // OnGestureListener
import android.view.GestureDetector.OnDoubleTapListener; // OnDoubleTapListener

//For vbo
import java.nio.ByteBuffer;//nio For non-blocking I/O
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

//For Matrix math
import android.opengl.Matrix;

//A View for OpenGLES3 graphics which also receives touch events
public class GLESView extends GLSurfaceView implements GLSurfaceView.Renderer, OnGestureListener, OnDoubleTapListener
{
	//Class variables or class fields
	private final Context context;

	private GestureDetector gestureDetector;

	//Shader Object
	private int vertexShaderObject;
	private int fragmentShaderObject;

	//Program Object
	private int shaderProgramObject;

	//As there are no pointers in java, when we want to use any variable as out parameter
	//we use array with only one member.
	//Vertex Array Object
	private int[] vao = new int[1];
	private int[] vbo = new int[1];

	private int mvpUniform;

	//4 x 4 matrix
	private float orthographicProjectionMatrix[] = new float[16];

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
		System.out.println("HAD: OpenGL-ES Version"+version);//"+" for concatination

		String glslVersion=gl.glGetString(GLES31.GL_SHADING_LANGUAGE_VERSION);
		System.out.println("HAD: GLSL Version"+glslVersion);

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
	}

	//abstract method from OnGestureListener so must be implemented
	@Override
	public boolean onScroll(MotionEvent e1,MotionEvent e2,float distanceX,float distanceY)
	{
		uninitialize();
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
		//Vertex Shader
		//Create Vertex Shader

		vertexShaderObject=GLES31.glCreateShader(GLES31.GL_VERTEX_SHADER);

		//vertex shader source code
		final String vertexShaderSourceCode = String.format
		(
			"#version 310 es"+
			"\n"+
			"in vec4 vPosition;"+
			"uniform mat4 u_mvp_matrix;"+
			"void main(void)"+
			"{"+
			"gl_Position = u_mvp_matrix * vPosition;"+
			"}"
		);

		//Provide Source Code to Shader
		GLES31.glShaderSource(vertexShaderObject,vertexShaderSourceCode);

		//Compile Shader & Check errors
		GLES31.glCompileShader(vertexShaderObject);
		int[] iShaderCompiledStatus = new int[1];
		int[] iInfoLogLength = new int[1];
		String szInfoLog = null; 
		GLES31.glGetShaderiv(vertexShaderObject,GLES31.GL_COMPILE_STATUS,iShaderCompiledStatus,0);
		if(iShaderCompiledStatus[0] == GLES31.GL_FALSE)
		{
			GLES31.glGetShaderiv(vertexShaderObject,GLES31.GL_INFO_LOG_LENGTH,iInfoLogLength,0);
			if(iInfoLogLength[0] > 0)
			{
				szInfoLog = GLES31.glGetShaderInfoLog(vertexShaderObject);
				System.out.println("HAD: Vertex Shader Compilation Log = "+szInfoLog);
				uninitialize();
				System.exit(0);
			}
		}

		//Fragment Shader
		//Create Shader
		fragmentShaderObject = GLES31.glCreateShader(GLES31.GL_FRAGMENT_SHADER);

		//Fragment Shader Source Code
		final String fragmentShaderSourceCode = String.format
		(
			"#version 310 es"+
			"\n"+
			"precision highp float;"+
			"out vec4 FragColor;"+
			"void main(void)"+
			"{"+
			"FragColor = vec4(1.0,1.0,1.0,1.0);"+
			"}"
		);

		GLES31.glShaderSource(fragmentShaderObject,fragmentShaderSourceCode);

		//Compile Shader and check for errors
		GLES31.glCompileShader(fragmentShaderObject);
		iShaderCompiledStatus[0] = 0;//reinitialize
		iInfoLogLength[0] = 0;//reinitialize
		szInfoLog=null;//reinitialize

		GLES31.glGetShaderiv(fragmentShaderObject,GLES31.GL_COMPILE_STATUS,iShaderCompiledStatus,0);
		if(iShaderCompiledStatus[0] == GLES31.GL_FALSE)
		{
			GLES31.glGetShaderiv(fragmentShaderObject,GLES31.GL_INFO_LOG_LENGTH,iInfoLogLength,0);
			if(iInfoLogLength[0] > 0)
			{
				szInfoLog = GLES31.glGetShaderInfoLog(fragmentShaderObject);
				System.out.println("HAD: Fragment Shader Compilation Log = "+szInfoLog);
				uninitialize();
				System.exit(0);
			}
		}

		//Create Shader Program
		shaderProgramObject = GLES31.glCreateProgram();

		//Attach Vertex Shader to Program Object
		GLES31.glAttachShader(shaderProgramObject,vertexShaderObject);

		//Attach Fragment Shader to Program Object
		GLES31.glAttachShader(shaderProgramObject,fragmentShaderObject);

		//Pre-link binding of shader program object with vertex shader attributes
		GLES31.glBindAttribLocation(shaderProgramObject,GLESMacros.HAD_ATTRIBUTE_VERTEX,"vPosition");

		//Link 2 shaders together to shader program object
		GLES31.glLinkProgram(shaderProgramObject);
		int[] iShaderProgramLinkStatus = new int[1];
		iInfoLogLength[0] = 0;//reinitialize
		szInfoLog = null;//reinitialize

		GLES31.glGetProgramiv(shaderProgramObject,GLES31.GL_LINK_STATUS,iShaderProgramLinkStatus,0);
		if(iShaderProgramLinkStatus[0] == GLES31.GL_FALSE)
		{
			GLES31.glGetProgramiv(shaderProgramObject,GLES31.GL_INFO_LOG_LENGTH,iInfoLogLength,0);
			if(iInfoLogLength[0] > 0)
			{
				szInfoLog = GLES31.glGetProgramInfoLog(shaderProgramObject);
				System.out.println("HAD: Shader Program Link Log = "+szInfoLog);
				uninitialize();
				System.exit(0);
			}
		}

		//Get MVP uniform location
		mvpUniform = GLES31.glGetUniformLocation(shaderProgramObject,"u_mvp_matrix");

		//Vertices, Color,Shader Attribs, Vbo, Vao initializations
		final float triangleVertices[] = new float[]
		{
			0.0f,50.0f,0.0f,
			-50.0f,-50.0f,0.0f,
			50.0f,-50.0f,0.0f
		};

		GLES31.glGenVertexArrays(1,vao,0);
		GLES31.glBindVertexArray(vao[0]);

		GLES31.glGenBuffers(1,vbo,0);
		GLES31.glBindBuffer(GLES31.GL_ARRAY_BUFFER,vbo[0]);

		ByteBuffer byteBuffer=ByteBuffer.allocateDirect(triangleVertices.length * 4);
		byteBuffer.order(ByteOrder.nativeOrder());
		FloatBuffer verticesBuffer=byteBuffer.asFloatBuffer();
		verticesBuffer.put(triangleVertices);
		verticesBuffer.position(0);

		GLES31.glBufferData(GLES31.GL_ARRAY_BUFFER,triangleVertices.length*4,verticesBuffer,GLES31.GL_STATIC_DRAW);

		GLES31.glVertexAttribPointer(GLESMacros.HAD_ATTRIBUTE_VERTEX,3,GLES31.GL_FLOAT,false,0,0);

		GLES31.glEnableVertexAttribArray(GLESMacros.HAD_ATTRIBUTE_VERTEX);

		GLES31.glBindBuffer(GLES31.GL_ARRAY_BUFFER,0);
		GLES31.glBindVertexArray(0);

		//Enable DepthTest
		GLES31.glEnable(GLES31.GL_DEPTH_TEST);

		//Specify Depth test to be done
		GLES31.glDepthFunc(GLES31.GL_LEQUAL);

		//We will always cull the back faces for better performance
		GLES31.glEnable(GLES31.GL_CULL_FACE);

		//Set the background frame color
		GLES31.glClearColor(0.0f,0.0f,1.0f,1.0f);

		//Set ProjectionMatrix to identity matrix
		Matrix.setIdentityM(orthographicProjectionMatrix,0);
	}

	private void resize(int width,int height)
	{
		//Adjust the viewport based on geometry changes such as screen rotation
		GLES31.glViewport(0,0,width,height);

		//Orthographic Projection
		if(width <= height)
			Matrix.orthoM(orthographicProjectionMatrix,0,-100.0f,100.0f,(-100.0f*(height/width)),(100.0f*(height/width)),-100.0f,100.0f);
		else
			Matrix.orthoM(orthographicProjectionMatrix,0,(-100.0f*(width/height)),(100.0f*(width/height)),-100.0f,100.0f,-100.0f,100.0f);
	}

	public void draw()
	{
		//Draw background color
		GLES31.glClear(GLES31.GL_COLOR_BUFFER_BIT|GLES31.GL_DEPTH_BUFFER_BIT);

		//Use Shader Program
		GLES31.glUseProgram(shaderProgramObject);

		float modelViewMatrix[] = new float[16];
		float modelViewProjectionMatrix[] = new float[16];

		//set ModelView and ModelViewProjection matrices to identity
		Matrix.setIdentityM(modelViewMatrix,0);
		Matrix.setIdentityM(modelViewProjectionMatrix,0);

		//multiply the modelview and projection matrix to get modelviewprojection matrix
		Matrix.multiplyMM(modelViewProjectionMatrix,0,orthographicProjectionMatrix,0,modelViewMatrix,0);

		//pass above modelviewprojection matrix to the vertex shader in 'u_mvp_matrix' shader variable
		//whose position value we already calculated in initWithFrame() by using glGetUniformLocation()
		GLES31.glUniformMatrix4fv(mvpUniform,1,false,modelViewProjectionMatrix,0);

		//Bind Vao
		GLES31.glBindVertexArray(vao[0]);

		//Draw either by glDrawTriangles() or glDrawArrays() or glDrawElements()
		GLES31.glDrawArrays(GLES31.GL_TRIANGLES,0,3); // 3 each with its x,y,z

		//unbind vao
		GLES31.glBindVertexArray(0);

		//un-use shader program
		GLES31.glUseProgram(0);

		//Like SwapBuffers() in Windows
		requestRender();
	}

	void uninitialize()
	{
		if(vao[0] != 0)
		{
			GLES31.glDeleteVertexArrays(1,vao,0);
			vao[0]=0;
		}

		if(vbo[0] != 0)
		{
			GLES31.glDeleteBuffers(1,vbo,0);
			vbo[0]=0;
		}

		if(shaderProgramObject != 0)
		{
			if(vertexShaderObject != 0)
			{
				GLES31.glDetachShader(shaderProgramObject,vertexShaderObject);
				GLES31.glDeleteShader(vertexShaderObject);
				vertexShaderObject=0;
			}

			if(fragmentShaderObject != 0)
			{
				GLES31.glDetachShader(shaderProgramObject,fragmentShaderObject);
				GLES31.glDeleteShader(fragmentShaderObject);
				fragmentShaderObject=0;
			}
		}

		if(shaderProgramObject != 0)
		{
			GLES31.glDeleteProgram(shaderProgramObject);
			shaderProgramObject=0;
		}
	}
}