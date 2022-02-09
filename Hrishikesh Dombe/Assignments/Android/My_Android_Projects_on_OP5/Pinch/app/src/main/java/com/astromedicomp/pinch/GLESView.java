package com.astromedicomp.pinch;

import android.content.Context; //For drawing context related
import android.opengl.GLSurfaceView; //For OpenGL Surface View and all related
import javax.microedition.khronos.opengles.GL10; //For OpenGLES 1.0 needed as param type GL10
import javax.microedition.khronos.egl.EGLConfig; //For EGLConfig needed as param type EGLConfig
import android.opengl.GLES32; // For OpenGLES 3.2
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
	private int[] vao_cube = new int[1];
	private int[] vbo_position = new int[1];
	private int[] vbo_normal = new int[1];

	private float angleCube;

	private int modelViewMatrixUniform, projectionMatrixUniform;
	private int ldUniform,kdUniform,lightPositionUniform;

	private int doubleTapUniform;

	//4 x 4 matrix
	private float perspectiveProjectionMatrix[] = new float[16];

	private int singleTap,doubleTap;

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

		String glslVersion=gl.glGetString(GLES32.GL_SHADING_LANGUAGE_VERSION);
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
		doubleTap++;
		if(doubleTap>1)
			doubleTap=0;
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
		singleTap++;
		if(singleTap>1)
			singleTap=0;
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

		vertexShaderObject=GLES32.glCreateShader(GLES32.GL_VERTEX_SHADER);

		//vertex shader source code
		final String vertexShaderSourceCode = String.format
		(
			"#version 320 es"+
			"\n"+
			"in vec4 vPosition;"+
			"in vec3 vNormal;"+
			"uniform mat4 u_model_view_matrix;"+
			"uniform mat4 u_projection_matrix;"+
			"uniform mediump int u_double_tap;"+
			"uniform vec3 u_Ld;"+
			"uniform vec3 u_Kd;"+
			"uniform vec4 u_light_position;"+
			"out vec3 diffuse_light;"+
			"void main(void)"+
			"{"+
			"if(u_double_tap==1)"+
			"{"+
			"vec4 eyeCoordinates = u_model_view_matrix * vPosition;"+
			"vec3 tnorm = normalize(mat3(u_model_view_matrix)*vNormal);"+
			"vec3 s = normalize(vec3(u_light_position - eyeCoordinates));"+
			"diffuse_light = u_Ld * u_Kd * max(dot(s,tnorm),0.0);"+
			"}"+
			"gl_Position = u_projection_matrix * u_model_view_matrix * vPosition;"+
			"}"
		);

		//Provide Source Code to Shader
		GLES32.glShaderSource(vertexShaderObject,vertexShaderSourceCode);

		//Compile Shader & Check errors
		GLES32.glCompileShader(vertexShaderObject);
		int[] iShaderCompiledStatus = new int[1];
		int[] iInfoLogLength = new int[1];
		String szInfoLog = null; 
		GLES32.glGetShaderiv(vertexShaderObject,GLES32.GL_COMPILE_STATUS,iShaderCompiledStatus,0);
		if(iShaderCompiledStatus[0] == GLES32.GL_FALSE)
		{
			GLES32.glGetShaderiv(vertexShaderObject,GLES32.GL_INFO_LOG_LENGTH,iInfoLogLength,0);
			if(iInfoLogLength[0] > 0)
			{
				szInfoLog = GLES32.glGetShaderInfoLog(vertexShaderObject);
				System.out.println("HAD: Vertex Shader Compilation Log = "+szInfoLog);
				uninitialize();
				System.exit(0);
			}
		}

		//Fragment Shader
		//Create Shader
		fragmentShaderObject = GLES32.glCreateShader(GLES32.GL_FRAGMENT_SHADER);

		//Fragment Shader Source Code
		final String fragmentShaderSourceCode = String.format
		(
			"#version 320 es"+
			"\n"+
			"precision highp float;"+
			"in vec3 diffuse_light;"+
			"out vec4 FragColor;"+
			"uniform int u_double_tap;"+
			"void main(void)"+
			"{"+
			"vec4 color;"+
			"if(u_double_tap == 1)"+
			"{"+
			"color = vec4(diffuse_light,1.0);"+
			"}"+
			"else"+
			"{"+
			"color = vec4(1.0,1.0,1.0,1.0);"+
			"}"+
			"FragColor = color;"+
			"}"
		);

		GLES32.glShaderSource(fragmentShaderObject,fragmentShaderSourceCode);

		//Compile Shader and check for errors
		GLES32.glCompileShader(fragmentShaderObject);
		iShaderCompiledStatus[0] = 0;//reinitialize
		iInfoLogLength[0] = 0;//reinitialize
		szInfoLog=null;//reinitialize

		GLES32.glGetShaderiv(fragmentShaderObject,GLES32.GL_COMPILE_STATUS,iShaderCompiledStatus,0);
		if(iShaderCompiledStatus[0] == GLES32.GL_FALSE)
		{
			GLES32.glGetShaderiv(fragmentShaderObject,GLES32.GL_INFO_LOG_LENGTH,iInfoLogLength,0);
			if(iInfoLogLength[0] > 0)
			{
				szInfoLog = GLES32.glGetShaderInfoLog(fragmentShaderObject);
				System.out.println("HAD: Fragment Shader Compilation Log = "+szInfoLog);
				uninitialize();
				System.exit(0);
			}
		}

		//Create Shader Program
		shaderProgramObject = GLES32.glCreateProgram();

		//Attach Vertex Shader to Program Object
		GLES32.glAttachShader(shaderProgramObject,vertexShaderObject);

		//Attach Fragment Shader to Program Object
		GLES32.glAttachShader(shaderProgramObject,fragmentShaderObject);

		//Pre-link binding of shader program object with vertex shader attributes
		GLES32.glBindAttribLocation(shaderProgramObject,GLESMacros.HAD_ATTRIBUTE_VERTEX,"vPosition");
		GLES32.glBindAttribLocation(shaderProgramObject,GLESMacros.HAD_ATTRIBUTE_NORMAL,"vNormal");

		//Link 2 shaders together to shader program object
		GLES32.glLinkProgram(shaderProgramObject);
		int[] iShaderProgramLinkStatus = new int[1];
		iInfoLogLength[0] = 0;//reinitialize
		szInfoLog = null;//reinitialize

		GLES32.glGetProgramiv(shaderProgramObject,GLES32.GL_LINK_STATUS,iShaderProgramLinkStatus,0);
		if(iShaderProgramLinkStatus[0] == GLES32.GL_FALSE)
		{
			GLES32.glGetProgramiv(shaderProgramObject,GLES32.GL_INFO_LOG_LENGTH,iInfoLogLength,0);
			if(iInfoLogLength[0] > 0)
			{
				szInfoLog = GLES32.glGetProgramInfoLog(shaderProgramObject);
				System.out.println("HAD: Shader Program Link Log = "+szInfoLog);
				uninitialize();
				System.exit(0);
			}
		}

		//Get MVP uniform location
		modelViewMatrixUniform = GLES32.glGetUniformLocation(shaderProgramObject,"u_model_view_matrix");
		projectionMatrixUniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_projection_matrix");
		doubleTapUniform = GLES32.glGetUniformLocation(shaderProgramObject,"u_double_tap");
		ldUniform = GLES32.glGetUniformLocation(shaderProgramObject,"u_Ld");
		kdUniform = GLES32.glGetUniformLocation(shaderProgramObject,"u_Kd");
		lightPositionUniform = GLES32.glGetUniformLocation(shaderProgramObject,"u_light_position");

		//Vertices, Color,Shader Attribs, Vbo, Vao initializations
		final float cubeVertices[] = new float[]
		{
			1.0f,1.0f,1.0f,
			-1.0f,1.0f,1.0f,
			-1.0f,-1.0f,1.0f,
			1.0f,-1.0f,1.0f,

			1.0f,1.0f,-1.0f,
			1.0f,1.0f,1.0f,
			1.0f,-1.0f,1.0f,
			1.0f,-1.0f,-1.0f,

			-1.0f,1.0f,-1.0f,
			1.0f,1.0f,-1.0f,
			1.0f,-1.0f,-1.0f,
			-1.0f,-1.0f,-1.0f,
			
			-1.0f,1.0f,1.0f,
			-1.0f,1.0f,-1.0f,
			-1.0f,-1.0f,-1.0f,
			-1.0f,-1.0f,1.0f,

			1.0f,1.0f,-1.0f,
			-1.0f,1.0f,-1.0f,
			-1.0f,1.0f,1.0f,
			1.0f,1.0f,1.0f,

			1.0f,-1.0f,1.0f,
			-1.0f,-1.0f,1.0f,
			-1.0f,-1.0f,-1.0f,
			1.0f,-1.0f,-1.0f
		}; 

		final float cubeNormal[] =new float[]
		{
			0.0f,0.0f,1.0f,
			0.0f,0.0f,1.0f,
			0.0f,0.0f,1.0f,
			0.0f,0.0f,1.0f,
			
			1.0f,0.0f,0.0f,
			1.0f,0.0f,0.0f,
			1.0f,0.0f,0.0f,
			1.0f,0.0f,0.0f,

			0.0f,0.0f,-1.0f,
			0.0f,0.0f,-1.0f,
			0.0f,0.0f,-1.0f,
			0.0f,0.0f,-1.0f,

			-1.0f,0.0f,0.0f,
			-1.0f,0.0f,0.0f,
			-1.0f,0.0f,0.0f,
			-1.0f,0.0f,0.0f,

			0.0f,1.0f,0.0f,
			0.0f,1.0f,0.0f,
			0.0f,1.0f,0.0f,
			0.0f,1.0f,0.0f,

			0.0f,-1.0f,0.0f,
			0.0f,-1.0f,0.0f,
			0.0f,-1.0f,0.0f,
			0.0f,-1.0f,0.0f,
		};

		/*****************Square*****************/
		GLES32.glGenVertexArrays(1,vao_cube,0);
		GLES32.glBindVertexArray(vao_cube[0]);

		/****************Square Position**************/
		GLES32.glGenBuffers(1,vbo_position,0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER,vbo_position[0]);

		ByteBuffer byteBuffer=ByteBuffer.allocateDirect(cubeVertices.length*4);
		byteBuffer.order(ByteOrder.nativeOrder());
		FloatBuffer verticesBuffer=byteBuffer.asFloatBuffer();
		verticesBuffer.put(cubeVertices);
		verticesBuffer.position(0);

		GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,cubeVertices.length*4,verticesBuffer,GLES32.GL_STATIC_DRAW);

		GLES32.glVertexAttribPointer(GLESMacros.HAD_ATTRIBUTE_VERTEX,3,GLES32.GL_FLOAT,false,0,0);

		GLES32.glEnableVertexAttribArray(GLESMacros.HAD_ATTRIBUTE_VERTEX);

		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER,0);

		/****************Square Color**************/
		GLES32.glGenBuffers(1,vbo_normal,0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER,vbo_normal[0]);

		byteBuffer=ByteBuffer.allocateDirect(cubeNormal.length*4);
		byteBuffer.order(ByteOrder.nativeOrder());
		FloatBuffer nomralBuffer=byteBuffer.asFloatBuffer();
		nomralBuffer.put(cubeNormal);
		nomralBuffer.position(0);

		GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,cubeNormal.length*4,nomralBuffer,GLES32.GL_STATIC_DRAW);

		GLES32.glVertexAttribPointer(GLESMacros.HAD_ATTRIBUTE_NORMAL,3,GLES32.GL_FLOAT,false,0,0);

		GLES32.glEnableVertexAttribArray(GLESMacros.HAD_ATTRIBUTE_NORMAL);

		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER,0);
		GLES32.glBindVertexArray(0);

		//Enable DepthTest
		GLES32.glEnable(GLES32.GL_DEPTH_TEST);

		//Specify Depth test to be done
		GLES32.glDepthFunc(GLES32.GL_LEQUAL);

		//We will always cull the back faces for better performance
		//GLES32.glEnable(GLES32.GL_CULL_FACE);

		//Set the background frame color
		GLES32.glClearColor(0.0f,0.0f,0.0f,1.0f);

		singleTap=0;
		doubleTap=0;

		//Set ProjectionMatrix to identity matrix
		Matrix.setIdentityM(perspectiveProjectionMatrix,0);
	}

	private void resize(int width,int height)
	{
		//Adjust the viewport based on geometry changes such as screen rotation
		GLES32.glViewport(0,0,width,height);

		//Perspective Projection
		Matrix.perspectiveM(perspectiveProjectionMatrix,0,45.0f,(float)width/(float)height,0.1f,100.0f);
	}

	public void draw()
	{
		//Draw background color
		GLES32.glClear(GLES32.GL_COLOR_BUFFER_BIT|GLES32.GL_DEPTH_BUFFER_BIT);

		//Use Shader Program
		GLES32.glUseProgram(shaderProgramObject);

		if(doubleTap == 1)
		{
			GLES32.glUniform1i(doubleTapUniform,1);

			GLES32.glUniform3f(ldUniform,1.0f,0.0f,0.0f);
			GLES32.glUniform3f(kdUniform,1.0f,1.0f,1.0f);
			float[] light_Position = {0.0f,0.0f,2.0f,1.0f};
			GLES32.glUniform4fv(lightPositionUniform,1,light_Position,0);
		}
		else
		{
			GLES32.glUniform1i(doubleTap,0);
		}

		float modelMatrix[] = new float[16];
		float modelViewMatrix[] = new float[16];
		float rotationMatrix[] = new float[16];
		float scaleMatrix[] = new float[16];

		//set ModelView and ModelViewProjection matrices to identity
		Matrix.setIdentityM(modelMatrix,0);
		Matrix.setIdentityM(modelViewMatrix,0);
		Matrix.setIdentityM(rotationMatrix,0);
		Matrix.setIdentityM(scaleMatrix,0);


		Matrix.translateM(modelMatrix,0,0.0f,0.0f,-6.0f);

		Matrix.scaleM(scaleMatrix,0,0.75f,0.75f,0.75f);

		Matrix.multiplyMM(modelViewMatrix,0,modelMatrix,0,scaleMatrix,0);

		Matrix.rotateM(rotationMatrix,0,angleCube,1.0f,0.0f,0.0f);
		Matrix.rotateM(rotationMatrix,0,angleCube,0.0f,1.0f,0.0f);
		Matrix.rotateM(rotationMatrix,0,angleCube,0.0f,0.0f,1.0f);

		Matrix.multiplyMM(modelViewMatrix,0,modelMatrix,0,rotationMatrix,0);

		GLES32.glUniformMatrix4fv(modelViewMatrixUniform,1,false,modelViewMatrix,0);

		GLES32.glUniformMatrix4fv(projectionMatrixUniform,1,false,perspectiveProjectionMatrix,0);

		GLES32.glBindVertexArray(vao_cube[0]);

		GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN,0,4);
		GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN,4,4);
		GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN,8,4);
		GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN,12,4);
		GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN,16,4);
		GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN,20,4);

		GLES32.glBindVertexArray(0);

		//un-use shader program
		GLES32.glUseProgram(0);

		update();
		//Like SwapBuffers() in Windows
		requestRender();
	}

	private void update()
	{
		if(singleTap==1)
		{
			angleCube=angleCube+1.0f;
			if(angleCube>=360.0f)
				angleCube=angleCube-360.0f;
		}
	}

	public void uninitialize()
	{
		if(vao_cube[0] != 0)
		{
			GLES32.glDeleteVertexArrays(1,vao_cube,0);
			vao_cube[0]=0;
		}

		if(vbo_position[0] != 0)
		{
			GLES32.glDeleteBuffers(1,vbo_position,0);
			vbo_position[0]=0;
		}

		if(vbo_normal[0] != 0)
		{
			GLES32.glDeleteBuffers(1,vbo_normal,0);
			vbo_normal[0]=0;
		}

		if(shaderProgramObject != 0)
		{
			if(vertexShaderObject != 0)
			{
				GLES32.glDetachShader(shaderProgramObject,vertexShaderObject);
				GLES32.glDeleteShader(vertexShaderObject);
				vertexShaderObject=0;
			}

			if(fragmentShaderObject != 0)
			{
				GLES32.glDetachShader(shaderProgramObject,fragmentShaderObject);
				GLES32.glDeleteShader(fragmentShaderObject);
				fragmentShaderObject=0;
			}
		}

		if(shaderProgramObject != 0)
		{
			GLES32.glDeleteProgram(shaderProgramObject);
			shaderProgramObject=0;
		}
	}
}