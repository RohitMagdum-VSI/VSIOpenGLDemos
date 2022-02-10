package com.rohit_r_jadhav.ortho_shader;

import android.opengl.GLSurfaceView;
import android.opengl.GLES32;

import javax.microedition.khronos.opengles.GL10;
import javax.microedition.khronos.egl.EGLConfig;

//For glGenBuffers()
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

//For Matrix Operation
import android.opengl.Matrix;

//For Event Handling
import android.view.MotionEvent;
import android.view.GestureDetector;
import android.view.GestureDetector.OnGestureListener;
import android.view.GestureDetector.OnDoubleTapListener;
import android.content.Context;

public class GLESView extends GLSurfaceView implements GLSurfaceView.Renderer, OnGestureListener, OnDoubleTapListener{

	//For Event
	private GestureDetector gestureDetector;
	private final Context context;

	//For Shaders
	private int vertexShaderObject;
	private int fragmentShaderObject;
	private int shaderProgramObject;

	//For Triangle
	private int vao_Triangle[] = new int[1];
	private int vbo_Triangle_Position[] = new int[1];

	//For Uniform
	private int mvpUniform;

	//For Projection
	private float orthoProjectionMatrix[] = new float[4 * 4];



	GLESView(Context drawingContext){
		super(drawingContext);
		context = drawingContext;

		gestureDetector = new GestureDetector(drawingContext, this, null, false);
		gestureDetector.setOnDoubleTapListener(this);


		setEGLContextClientVersion(3);
		setRenderer(this);
		setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);
	}


	@Override
	public boolean onTouchEvent(MotionEvent event){
		int eventaction = event.getAction();
		if(!gestureDetector.onTouchEvent(event))
			super.onTouchEvent(event);
		return(true);
	}

	/********** Methods in OnDoubleTapListener Interface **********/

	@Override
	public boolean onDoubleTap(MotionEvent e){
		return(true);
	} 

	@Override
	public boolean onDoubleTapEvent(MotionEvent e){
		return(true);
	}

	@Override
	public boolean onSingleTapConfirmed(MotionEvent e){
		return(true);
	}


	/********** Methods in OnGestureListener Interface **********/

	@Override
	public boolean onDown(MotionEvent e){
		return(true);
	}

	@Override
	public boolean onFling(MotionEvent e1, MotionEvent e2, float velocityX, float velocityY){
		return(true);
	}

	@Override
	public boolean onSingleTapUp(MotionEvent e){
		return(true);
	}

	@Override
	public void onLongPress(MotionEvent e){

	}

	@Override
	public void onShowPress(MotionEvent e){

	}

	@Override 
	public boolean onScroll(MotionEvent e1, MotionEvent e2, float distanceX, float distanceY){
		uninitialize();
		System.exit(0);
		return(true);
	}



	/********** Methods in GLSurfaceView.Renderer Interface **********/

	@Override
	public void onSurfaceCreated(GL10 gl, EGLConfig config){
		
		String version = gl.glGetString(GL10.GL_VERSION);
		/*String glsl_version = gl.glGetString(GLES32.GL_SHADING_LANGUAGE_VERSION);
		String vender = gl.glGetString(GLES32.GL_VENDOR);
		String renderer = gl.glGetString(GLES32.GL_RENDERER);*/

		String glsl_version = gl.glGetString(GLES32.GL_SHADING_LANGUAGE_VERSION);
		String vender = gl.glGetString(GL10.GL_VENDOR);
		String renderer = gl.glGetString(GL10.GL_RENDERER);


		System.out.println("RTR: OpenGL ES Version: "+ version);
		System.out.println("RTR: OpenGLSL Version: " + glsl_version);
		System.out.println("RTR: OpenGL Vender:" + vender);
		System.out.println("RTR: OpenGL Renderer: " + renderer);

		initialize();
	}

	@Override
	public void onSurfaceChanged(GL10 unused, int width, int height){
		resize(width, height);
	}

	@Override
	public void onDrawFrame(GL10 unused){
		display();
	}


	/********** Now OpenGL **********/


	private void initialize(){

		System.out.println("RTR: in Initialize!!");


		/********** Vertex Shader **********/
		vertexShaderObject = GLES32.glCreateShader(GLES32.GL_VERTEX_SHADER);

		System.out.println("RTR: VS : Shader Created!!");

		final String vertexShaderSourceCode = String.format(
				"#version 320 es" +
				"\n" +
				"in vec4 vPosition;" +
				"uniform mat4 u_mvp_matrix;" +
				"void main(void)" +
				"{" +
					"gl_Position = u_mvp_matrix * vPosition;" +
				"}"
			);


		GLES32.glShaderSource(vertexShaderObject, vertexShaderSourceCode);

		System.out.println("RTR: VS : Shader Source Done!!");

		GLES32.glCompileShader(vertexShaderObject);

		System.out.println("RTR: VS : Shader Compiled!!");

		int iShaderCompileStatus[] = new int[1];
		int iInfoLogLength[] = new int[1];
		String szInfoLog = null;

		GLES32.glGetShaderiv(vertexShaderObject, GLES32.GL_COMPILE_STATUS, iShaderCompileStatus, 0);
		if(iShaderCompileStatus[0] == GLES32.GL_FALSE){
			GLES32.glGetShaderiv(vertexShaderObject, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if(iInfoLogLength[0] > 0){

				szInfoLog = GLES32.glGetShaderInfoLog(vertexShaderObject);
				System.out.println("RTR: Vertex Shader Compilation Error: "+ szInfoLog);

				uninitialize();
				System.exit(0);
			}

		}


		/********** Fragment Shader **********/
		fragmentShaderObject = GLES32.glCreateShader(GLES32.GL_FRAGMENT_SHADER);

		System.out.println("RTR: FS : Shader Created!!");

		String fragmentShaderSourceCode = String.format(
				"#version 320 es" +
				"\n" +
				"precision highp float;" +
				"out vec4 FragColor;" +
				"void main(void)" +
				"{" +
					"FragColor = vec4(1.0, 1.0, 0.0, 0.0);" +
				"}"
			);

		GLES32.glShaderSource(fragmentShaderObject, fragmentShaderSourceCode);

		System.out.println("RTR: FS : Shader Source Done!!");


		GLES32.glCompileShader(fragmentShaderObject);

		System.out.println("RTR: FS : Shader Compiled!!");


		GLES32.glGetShaderiv(fragmentShaderObject, GLES32.GL_COMPILE_STATUS, iShaderCompileStatus, 0);
		if(iShaderCompileStatus[0] == GLES32.GL_FALSE){
			GLES32.glGetShaderiv(fragmentShaderObject, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if(iInfoLogLength[0] > 0){

				szInfoLog = GLES32.glGetShaderInfoLog(fragmentShaderObject);
				System.out.println("RTR: Fragment Shader Compilation Error: " + szInfoLog);

				uninitialize();
				System.exit(0);
			}

		}



		/********** Shader Program Object **********/

		shaderProgramObject = GLES32.glCreateProgram();

		GLES32.glAttachShader(shaderProgramObject, vertexShaderObject);
		GLES32.glAttachShader(shaderProgramObject, fragmentShaderObject);

		GLES32.glBindAttribLocation(shaderProgramObject, GLESMacros.AMC_ATTRIBUTE_POSITION, "vPosition");

		GLES32.glLinkProgram(shaderProgramObject);

		System.out.println("RTR: SPO : Program Linked!!");


		int iProgramLinkStatus[] = new int[1];

		GLES32.glGetProgramiv(shaderProgramObject, GLES32.GL_LINK_STATUS, iProgramLinkStatus, 0);
		if(iProgramLinkStatus[0] == GLES32.GL_FALSE){
			GLES32.glGetProgramiv(shaderProgramObject, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if(iInfoLogLength[0] > 0){

				szInfoLog = GLES32.glGetProgramInfoLog(shaderProgramObject);
				System.out.println("RTR: Shader Program Linking Error: " + szInfoLog);

				uninitialize();
				System.exit(0);

			}
		}

		mvpUniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_mvp_matrix");


		/*********** Position **********/
		final float Triangle_Position[] = new float[] {
			0.0f, 50.0f, 0.0f,
			-50.0f, -50.0f, 0.0f,
			50.0f, -50.0f, 0.0f			
		};


		/********** Triangle **********/
		GLES32.glGenVertexArrays(1, vao_Triangle, 0);
		GLES32.glBindVertexArray(vao_Triangle[0]);

			/********** Position **********/
			GLES32.glGenBuffers(1, vbo_Triangle_Position, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Triangle_Position[0]);

				/********** Preparation For glBufferData **********/

				//Native Memory Allocate Kar
				ByteBuffer byteBuffer = ByteBuffer.allocateDirect(Triangle_Position.length * 4);

				//Arrange the ByteOrder to the Native ByteOrder
				byteBuffer.order(ByteOrder.nativeOrder());

				//Convert Out byteBuffer into Float Buffer
				FloatBuffer positionBuffer = byteBuffer.asFloatBuffer();

				//Put Data into positionBuffer
				positionBuffer.put(Triangle_Position);

				//Set positionArray at 0 index/Position
				positionBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
												Triangle_Position.length * 4,
												positionBuffer,
												GLES32.GL_STATIC_DRAW);

			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_POSITION, 3, GLES32.GL_FLOAT, false, 0, 0);

			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);

			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		GLES32.glBindVertexArray(0);


		System.out.println("RTR: VS : All Triangle Operations done!!");


		GLES32.glEnable(GLES32.GL_DEPTH_TEST);
		GLES32.glDepthFunc(GLES32.GL_LEQUAL);
		//GLES32.glClearDepth(1.0f);

		GLES32.glClearColor(0.0f, 0.0f, 1.0f, 0.0f);

		Matrix.setIdentityM(orthoProjectionMatrix, 0);
	}



	private void uninitialize(){

		if(vbo_Triangle_Position[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Triangle_Position, 0);
			vbo_Triangle_Position[0] = 0;
		}

		if(vao_Triangle[0] != 0){
			GLES32.glDeleteBuffers(1, vao_Triangle, 0);
			vao_Triangle[0] = 0;
		}

		if(shaderProgramObject != 0){

			/*int iShaderCount[] = new int[1];
			int iShaderNo;

			GLES32.glUseProgram(shaderProgramObject);

				GLES32.glGetProgramiv(shaderProgramObject, GLES32.GL_ATTACHED_SHADERS, iShaderCount, 0);
				if(iShaderCount[0] != 0){

					int iShaders[] = new int[iShaderCount[0]];

					GLES32.glGetAttachedShaders(shaderProgramObject, iShaderCount[0], iShaderCount, 0, iShaders, 0);

					for(iShaderNo = 0; iShaderNo < iShaderCount[0]; iShaderNo++){

							GLES32.glDetachShader(shaderProgramObject, iShaders[iShaderNo]);
							GLES32.glDeleteShader(iShaders[iShaderNo]);
							iShaders[iShaderNo] = 0;
					}

				}*/

				if(fragmentShaderObject != 0){
					GLES32.glDetachShader(shaderProgramObject, fragmentShaderObject);
					GLES32.glDeleteShader(fragmentShaderObject);
					fragmentShaderObject = 0;
				}

				if(vertexShaderObject != 0){
					GLES32.glDetachShader(shaderProgramObject, vertexShaderObject);
					GLES32.glDeleteShader(vertexShaderObject);
					vertexShaderObject = 0;
				}


			GLES32.glUseProgram(0);
			GLES32.glDeleteProgram(shaderProgramObject);
			shaderProgramObject = 0;

		}

	}


	private void resize(int width, int height){
		if(height == 0)
			height = 1;

		GLES32.glViewport(0, 0, width, height);

		Matrix.setIdentityM(orthoProjectionMatrix, 0);

		if(width <= height){
				Matrix.orthoM(orthoProjectionMatrix, 0,	
										-100.0f, 100.0f,
										(-100.0f * (float)height / (float)width),
										(100.0f * (float)height / (float)width),
										-100.0f, 100.0f);
		}
		else{
				Matrix.orthoM(orthoProjectionMatrix, 0,
										(-100.0f * (float)width / (float)height),
										(100.0f * (float) width / (float)height),
										-100.0f, 100.0f,
										-100.0f, 100.0f);

		}
	}


	private void display(){

		float modelViewMatrix[] = new float[4 * 4];
		float modelViewProjectionMatrix[] = new float[4 * 4];

		GLES32.glClear(GLES32.GL_COLOR_BUFFER_BIT | GLES32.GL_DEPTH_BUFFER_BIT);


		GLES32.glUseProgram(shaderProgramObject);

			/********** Triangle **********/
			Matrix.setIdentityM(modelViewMatrix, 0);
			Matrix.setIdentityM(modelViewProjectionMatrix, 0);

			Matrix.multiplyMM(modelViewProjectionMatrix, 0,
										orthoProjectionMatrix, 0,
										modelViewMatrix, 0);

			GLES32.glUniformMatrix4fv(mvpUniform, 1, false, modelViewProjectionMatrix, 0);

			GLES32.glBindVertexArray(vao_Triangle[0]);

				GLES32.glDrawArrays(GLES32.GL_TRIANGLES, 0, 3);

			GLES32.glBindVertexArray(0);

		GLES32.glUseProgram(0);

		requestRender();
	}


};

