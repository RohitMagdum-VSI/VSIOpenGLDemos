package com.rohit_r_jadhav.static_india;


import android.opengl.GLSurfaceView;
import android.opengl.GLES32;
import android.opengl.Matrix;

import javax.microedition.khronos.opengles.GL10;
import javax.microedition.khronos.egl.EGLConfig;

import android.view.MotionEvent;
import android.view.GestureDetector;
import android.view.GestureDetector.OnGestureListener;
import android.view.GestureDetector.OnDoubleTapListener;
import android.content.Context;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

public class GLESView extends GLSurfaceView implements GLSurfaceView.Renderer, OnGestureListener, OnDoubleTapListener{

	private GestureDetector gestureDetector;
	private Context context;

	GLESView(Context drawingContext){
		super(drawingContext);
		context = drawingContext;

		setEGLContextClientVersion(3);
		setRenderer(this);
		setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);

		gestureDetector = new GestureDetector(drawingContext, this, null, false);
		gestureDetector.setOnDoubleTapListener(this);
	}


	@Override
	public boolean onTouchEvent(MotionEvent e){
		int action = e.getAction();
		if(!gestureDetector.onTouchEvent(e)){
			super.onTouchEvent(e);
		}
		return(true);
	}



	/********** Methods from OnDoubleTapListener **********/
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



	/********** Methods From OnGestureDetector Interface **********/
	@Override
	public boolean onSingleTapUp(MotionEvent e){
		return(true);
	}

	@Override
	public boolean onDown(MotionEvent e){
		return(true);
	}

	@Override
	public boolean onFling(MotionEvent e1, MotionEvent e2, float velocityX, float velocityY){
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





	/********** Methods From GLSurfaceView.Renderer Interface **********/
	@Override
	public void onSurfaceCreated(GL10 gl, EGLConfig config){
		String version = gl.glGetString(GL10.GL_VERSION);
		String glsl_version = gl.glGetString(GLES32.GL_SHADING_LANGUAGE_VERSION);
		String vendor = gl.glGetString(GL10.GL_VENDOR);
		String renderer = gl.glGetString(GL10.GL_RENDERER);

		System.out.println("RTR: OpenGL Version: " + version);
		System.out.println("RTR: OpenGLSL Version: " + glsl_version);
		System.out.println("RTR: Vendor: " + vendor);
		System.out.println("RTR: Renderer: " + renderer);

		initialize();
	}


	@Override
	public void onSurfaceChanged(GL10 unused, int width, int height){
		resize(width, height);
	}

	@Override
	public void onDrawFrame(GL10 unuesd){
		display();
	}





	/********** NOW OpenGL **********/

	//For I
	private int vao_I[] = new int[1];
	private int vbo_I_Position[] = new int[1];
	private int vbo_I_Color[] = new int[1];

	//For N
	private int vao_N[] = new int[1];
	private int vbo_N_Position[] = new int[1];
	private int vbo_N_Color[] = new int[1];

	//For D
	private int vao_D[] = new int[1];
	private int vbo_D_Position[] = new int[1];
	private int vbo_D_Color[] = new int[1];

	//For A
	private int vao_A[] = new int[1];
	private int vbo_A_Position[] = new int[1];
	private int vbo_A_Color[] = new int[1];

	//For Flag
	private int vao_Flag[] = new int[1];
	private int vbo_Flag_Position[] = new int[1];
	private int vbo_Flag_Color[] = new int[1];

	//For Uniform
	private int mvpUniform;

	//For Shader
	private int vertexShaderObject;
	private int fragmentShaderObject;
	private int shaderProgramObject;

	//For Projection
	float perspectiveProjectionMatrix[] = new float[4 * 4];




	private void initialize(){


		/********** Vertex Shader **********/
		vertexShaderObject = GLES32.glCreateShader(GLES32.GL_VERTEX_SHADER);

		final String vertexShaderSourceCode = String.format(
				"#version 320 es" +
				"\n" +
				"in vec4 vPosition;" +
				"in vec4 vColor;" +
				"out vec4 outColor;" +
				"uniform mat4 u_mvp_matrix;" +
				"void main(void)" +
				"{" + 
					"gl_Position = u_mvp_matrix * vPosition;" +
					"outColor = vColor;" +
				"}" 
			);

		GLES32.glShaderSource(vertexShaderObject, vertexShaderSourceCode);

		GLES32.glCompileShader(vertexShaderObject);

		int iShaderCompileStatus[] = new int[1];
		int iInfoLogLength[] = new int[1];
		String szInfoLog = null;

		GLES32.glGetShaderiv(vertexShaderObject, GLES32.GL_COMPILE_STATUS, iShaderCompileStatus, 0);
		if(iShaderCompileStatus[0] == GLES32.GL_FALSE){
			GLES32.glGetShaderiv(vertexShaderObject, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if(iInfoLogLength[0] > 0){

				szInfoLog = GLES32.glGetShaderInfoLog(vertexShaderObject);
				System.out.println("RTR: Vertex Shader Compilation Error: " + szInfoLog);

				uninitialize();
				System.exit(0);
			
			}
		}


		/********** Fragment Shader **********/
		fragmentShaderObject = GLES32.glCreateShader(GLES32.GL_FRAGMENT_SHADER);

		final String fragmentShaderSourceCode = String.format(
				"#version 320 es" +
				"\n" +
				"precision highp float;" +
				"in vec4 outColor;" +
				"out vec4 FragColor;" +
				"void main(void)" +
				"{" +
					"FragColor = outColor;" +
				"}"
			);

		GLES32.glShaderSource(fragmentShaderObject, fragmentShaderSourceCode);

		GLES32.glCompileShader(fragmentShaderObject);

		GLES32.glGetShaderiv(fragmentShaderObject, GLES32.GL_COMPILE_STATUS, iShaderCompileStatus, 0);
		if(iShaderCompileStatus[0] == GLES32.GL_FALSE){
			GLES32.glGetShaderiv(fragmentShaderObject, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if(iInfoLogLength[0] > 0){
				szInfoLog = GLES32.glGetShaderInfoLog(fragmentShaderObject);
				System.out.println("RTR: Fragment Shader Compilation Error: "+ szInfoLog);
				uninitialize();
				System.exit(0);
			}
		}


		/********** Shader Program Object **********/
		shaderProgramObject = GLES32.glCreateProgram();

		GLES32.glAttachShader(shaderProgramObject, vertexShaderObject);
		GLES32.glAttachShader(shaderProgramObject, fragmentShaderObject);

		GLES32.glBindAttribLocation(shaderProgramObject, GLESMacros.AMC_ATTRIBUTE_POSITION, "vPosition");
		GLES32.glBindAttribLocation(shaderProgramObject, GLESMacros.AMC_ATTRIBUTE_COLOR, "vColor");

		GLES32.glLinkProgram(shaderProgramObject);

		int iProgramLinkStatus[] = new int[1];

		GLES32.glGetProgramiv(shaderProgramObject, GLES32.GL_LINK_STATUS, iProgramLinkStatus, 0);
		if(iProgramLinkStatus[0] == GLES32.GL_FALSE){
			GLES32.glGetProgramiv(shaderProgramObject, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if(iInfoLogLength[0] > 0){
				szInfoLog = GLES32.glGetProgramInfoLog(shaderProgramObject);
				System.out.println("RTR: Shader Program Linking Error: "+ szInfoLog);
				uninitialize();
				System.exit(0);
			}
		}

		mvpUniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_mvp_matrix");


		/********** Position And Color **********/
		float I_Position[] = new float[]{
			-0.3f, 1.0f, 0.0f,
			0.3f, 1.0f, 0.0f,

			0.0f, 1.0f, 0.0f,
			0.0f, -1.0f, 0.0f,

			-0.3f, -1.0f, 0.0f,
			0.3f, -1.0f, 0.0f
		};


		float I_Color[] = new float[]{
			1.0f, 0.6f, 0.2f,
			1.0f, 0.6f, 0.2f,

			1.0f, 0.6f, 0.2f,
			0.0705f, 0.533f, 0.0274f,

			0.0705f, 0.533f, 0.0274f,
			0.0705f, 0.533f, 0.0274f
		};

		float N_Position[] = new float[]{
			0.0f, 1.06f, 0.0f,
			0.0f, -1.06f, 0.0f,

			0.75f, 1.06f, 0.0f,
			0.75f, -1.06f, 0.0f,

			0.0f, 1.06f, 0.0f,
			0.75f, -1.06f, 0.0f
		};


		float N_Color[] = new float[]{
			1.0f, 0.6f, 0.2f,
			0.0705f, 0.533f, 0.0274f,

			1.0f, 0.6f, 0.2f,
			0.0705f, 0.533f, 0.0274f,

			1.0f, 0.6f, 0.2f,
			0.0705f, 0.533f, 0.0274f
		};

		float D_Position[] = new float[]{
			0.0f, 1.0f, 0.0f,
			0.0f, -1.0f, 0.0f,

			-0.1f, 1.0f, 0.0f,
			0.6f, 1.0f, 0.0f,

			-0.1f, -1.0f, 0.0f,
			0.6f, -1.0f, 0.0f,

			0.6f, 1.0f, 0.0f,
			0.6f, -1.0f, 0.0f
		};

		float D_Color[] = new float[]{
			1.0f, 0.6f, 0.2f,
			0.0705f, 0.533f, 0.0274f,

			1.0f, 0.6f, 0.2f,
			1.0f, 0.6f, 0.2f,

			0.0705f, 0.533f, 0.0274f,
			0.0705f, 0.533f, 0.0274f,

			1.0f, 0.6f, 0.2f,
			0.0705f, 0.533f, 0.0274f
		};


		float A_Position[] = new float[]{
			0.0f, 1.06f, 0.0f,
			-0.5f, -1.06f, 0.0f,

			0.0f, 1.06f, 0.0f,
			0.5f, -1.06f, 0.0f
		};


		float A_Color[] = new float[]{
			1.0f, 0.6f, 0.2f,
			0.0705f, 0.533f, 0.0274f,

			1.0f, 0.6f, 0.2f,
			0.0705f, 0.533f, 0.0274f
		};


		float Flag_Position[] = new float[]{
			-0.207f, 0.06f, 0.0f,
			0.207f, 0.06f, 0.0f,

			-0.218f, 0.0f, 0.0f,
			0.219f, 0.0f, 0.0f,

			-0.235f, -0.06f, 0.0f,
			0.235f, -0.06f, 0.0f
		};


		float Flag_Color[] = new float[]{
			1.0f, 0.6f, 0.2f,
			1.0f, 0.6f, 0.2f,

			1.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f,

			0.0705f, 0.533f, 0.0274f,
			0.0705f, 0.533f, 0.0274f
		};




		/********** I **********/
		GLES32.glGenVertexArrays(1, vao_I, 0);
		GLES32.glBindVertexArray(vao_I[0]);

			/********** Position **********/
			GLES32.glGenBuffers(1, vbo_I_Position, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_I_Position[0]);

				/********** For glBufferData ***********/
				ByteBuffer iPosition_ByteBuffer = ByteBuffer.allocateDirect(I_Position.length * 4);
				iPosition_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer iPosition_FloatBuffer = iPosition_ByteBuffer.asFloatBuffer();
				iPosition_FloatBuffer.put(I_Position);
				iPosition_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							I_Position.length * 4,
							iPosition_FloatBuffer,
							GLES32.GL_STATIC_DRAW);

			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_POSITION,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

			/********** Color **********/
			GLES32.glGenBuffers(1, vbo_I_Color, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_I_Color[0]);

				/********** For glBufferData **********/
				ByteBuffer iColor_ByteBuffer = ByteBuffer.allocateDirect(I_Color.length * 4);
				iColor_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer iColor_FloatBuffer = iColor_ByteBuffer.asFloatBuffer();
				iColor_FloatBuffer.put(I_Color);
				iColor_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER, 
							I_Color.length * 4,
							iColor_FloatBuffer,
							GLES32.GL_STATIC_DRAW);
			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_COLOR,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_COLOR);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		GLES32.glBindVertexArray(0);






		/********** N **********/
		GLES32.glGenVertexArrays(1, vao_N, 0);
		GLES32.glBindVertexArray(vao_N[0]);

			/********** Position **********/
			GLES32.glGenBuffers(1, vbo_N_Position, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_N_Position[0]);

				/********** For glBufferData ***********/
				ByteBuffer nPosition_ByteBuffer = ByteBuffer.allocateDirect(N_Position.length * 4);
				nPosition_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer nPosition_FloatBuffer = nPosition_ByteBuffer.asFloatBuffer();
				nPosition_FloatBuffer.put(N_Position);
				nPosition_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							N_Position.length * 4,
							nPosition_FloatBuffer,
							GLES32.GL_STATIC_DRAW);

			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_POSITION,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

			/********** Color **********/
			GLES32.glGenBuffers(1, vbo_N_Color, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_N_Color[0]);

				/********** For glBufferData **********/
				ByteBuffer nColor_ByteBuffer = ByteBuffer.allocateDirect(N_Color.length * 4);
				nColor_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer nColor_FloatBuffer = nColor_ByteBuffer.asFloatBuffer();
				nColor_FloatBuffer.put(N_Color);
				nColor_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER, 
							N_Color.length * 4,
							nColor_FloatBuffer,
							GLES32.GL_STATIC_DRAW);
			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_COLOR,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_COLOR);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		GLES32.glBindVertexArray(0);




		/********** D **********/
		GLES32.glGenVertexArrays(1, vao_D, 0);
		GLES32.glBindVertexArray(vao_D[0]);

			/********** Position **********/
			GLES32.glGenBuffers(1, vbo_D_Position, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_D_Position[0]);

				/********** For glBufferData ***********/
				ByteBuffer dPosition_ByteBuffer = ByteBuffer.allocateDirect(D_Position.length * 4);
				dPosition_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer dPosition_FloatBuffer = dPosition_ByteBuffer.asFloatBuffer();
				dPosition_FloatBuffer.put(D_Position);
				dPosition_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							D_Position.length * 4,
							dPosition_FloatBuffer,
							GLES32.GL_STATIC_DRAW);

			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_POSITION,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

			/********** Color **********/
			GLES32.glGenBuffers(1, vbo_D_Color, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_D_Color[0]);

				/********** For glBufferData **********/
				ByteBuffer dColor_ByteBuffer = ByteBuffer.allocateDirect(D_Color.length * 4);
				dColor_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer dColor_FloatBuffer = dColor_ByteBuffer.asFloatBuffer();
				dColor_FloatBuffer.put(D_Color);
				dColor_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER, 
							D_Color.length * 4,
							dColor_FloatBuffer,
							GLES32.GL_STATIC_DRAW);
			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_COLOR,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_COLOR);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		GLES32.glBindVertexArray(0);





		/********** A **********/
		GLES32.glGenVertexArrays(1, vao_A, 0);
		GLES32.glBindVertexArray(vao_A[0]);

			/********** Position **********/
			GLES32.glGenBuffers(1, vbo_A_Position, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_A_Position[0]);

				/********** For glBufferData ***********/
				ByteBuffer aPosition_ByteBuffer = ByteBuffer.allocateDirect(A_Position.length * 4);
				aPosition_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer aPosition_FloatBuffer = aPosition_ByteBuffer.asFloatBuffer();
				aPosition_FloatBuffer.put(A_Position);
				aPosition_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							A_Position.length * 4,
							aPosition_FloatBuffer,
							GLES32.GL_STATIC_DRAW);

			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_POSITION,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

			/********** Color **********/
			GLES32.glGenBuffers(1, vbo_A_Color, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_A_Color[0]);

				/********** For glBufferData **********/
				ByteBuffer aColor_ByteBuffer = ByteBuffer.allocateDirect(A_Color.length * 4);
				aColor_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer aColor_FloatBuffer = aColor_ByteBuffer.asFloatBuffer();
				aColor_FloatBuffer.put(A_Color);
				aColor_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER, 
							A_Color.length * 4,
							aColor_FloatBuffer,
							GLES32.GL_STATIC_DRAW);
			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_COLOR,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_COLOR);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		GLES32.glBindVertexArray(0);







		/********** Flag **********/
		GLES32.glGenVertexArrays(1, vao_Flag, 0);
		GLES32.glBindVertexArray(vao_Flag[0]);

			/********** Position **********/
			GLES32.glGenBuffers(1, vbo_Flag_Position, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Flag_Position[0]);

				/********** For glBufferData ***********/
				ByteBuffer flagPosition_ByteBuffer = ByteBuffer.allocateDirect(Flag_Position.length * 4);
				flagPosition_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer flagPosition_FloatBuffer = flagPosition_ByteBuffer.asFloatBuffer();
				flagPosition_FloatBuffer.put(Flag_Position);
				flagPosition_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							Flag_Position.length * 4,
							flagPosition_FloatBuffer,
							GLES32.GL_STATIC_DRAW);

			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_POSITION,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

			/********** Color **********/
			GLES32.glGenBuffers(1, vbo_Flag_Color, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Flag_Color[0]);

				/********** For glBufferData **********/
				ByteBuffer flagColor_ByteBuffer = ByteBuffer.allocateDirect(Flag_Color.length * 4);
				flagColor_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer flagColor_FloatBuffer = flagColor_ByteBuffer.asFloatBuffer();
				flagColor_FloatBuffer.put(Flag_Color);
				flagColor_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER, 
							Flag_Color.length * 4,
							flagColor_FloatBuffer,
							GLES32.GL_STATIC_DRAW);
			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_COLOR,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_COLOR);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);


		GLES32.glBindVertexArray(0);




		GLES32.glEnable(GLES32.GL_DEPTH_TEST);
		GLES32.glDepthFunc(GLES32.GL_LEQUAL);

		GLES32.glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	}


	private void uninitialize(){

		//Flag
		if (vbo_Flag_Color[0] != 0) {
			GLES32.glDeleteBuffers(1, vbo_Flag_Color, 0);
			vbo_Flag_Color[0] = 0;
		}

		if (vbo_Flag_Position[0] != 0) {
			GLES32.glDeleteBuffers(1, vbo_Flag_Position, 0);
			vbo_Flag_Position[0] = 0;
		}

		if (vao_Flag[0] != 0) {
			GLES32.glDeleteVertexArrays(1, vao_Flag, 0);
			vao_Flag[0] = 0;
		}


		//A
		if (vbo_A_Color[0] != 0) {
			GLES32.glDeleteBuffers(1, vbo_A_Color, 0);
			vbo_A_Color[0] = 0;
		}

		if (vbo_A_Position[0] != 0) {
			GLES32.glDeleteBuffers(1, vbo_A_Position, 0);
			vbo_A_Position[0] = 0;
		}

		if (vao_A[0] != 0) {
			GLES32.glDeleteVertexArrays(1, vao_A, 0);
			vao_A[0] = 0;
		}


		//D	
		if (vbo_D_Color[0] != 0) {
			GLES32.glDeleteBuffers(1, vbo_D_Color, 0);
			vbo_D_Color[0] = 0;
		}

		if (vbo_D_Position[0] != 0) {
			GLES32.glDeleteBuffers(1, vbo_D_Position, 0);
			vbo_D_Position[0] = 0;
		}

		if (vao_D[0] != 0) {
			GLES32.glDeleteVertexArrays(1, vao_D, 0);
			vao_D[0] = 0;
		}		

		//N
		if (vbo_N_Color[0] != 0) {
			GLES32.glDeleteBuffers(1, vbo_N_Color, 0);
			vbo_N_Color[0] = 0;
		}

		if (vbo_N_Position[0] != 0) {
			GLES32.glDeleteBuffers(1, vbo_N_Position, 0);
			vbo_N_Position[0] = 0;
		}

		if (vao_N[0] != 0) {
			GLES32.glDeleteVertexArrays(1, vao_N, 0);
			vao_N[0] = 0;
		}

		//I
		if (vbo_I_Color[0] != 0) {
			GLES32.glDeleteBuffers(1, vbo_I_Color, 0);
			vbo_I_Color[0] = 0;
		}

		if (vbo_I_Position[0] != 0) {
			GLES32.glDeleteBuffers(1, vbo_I_Position, 0);
			vbo_I_Position[0] = 0;
		}

		if (vao_I[0] != 0) {
			GLES32.glDeleteVertexArrays(1, vao_I, 0);
			vao_I[0] = 0;
		}


		if(shaderProgramObject != 0){

			GLES32.glUseProgram(shaderProgramObject);

				if(fragmentShaderObject != 0){
					GLES32.glDetachShader(shaderProgramObject, fragmentShaderObject);
					GLES32.glDeleteShader(fragmentShaderObject);
					fragmentShaderObject = 0;
					System.out.println("RTR: Fragment Shader Detached and Deleted!!");
				}

				if(vertexShaderObject != 0){
					GLES32.glDetachShader(shaderProgramObject, vertexShaderObject);
					GLES32.glDeleteShader(vertexShaderObject);
					vertexShaderObject = 0;
					System.out.println("RTR: Vertex Shader Detached and Deleted!!");
				}
				
			GLES32.glUseProgram(0);
			GLES32.glDeleteProgram(shaderProgramObject);
			shaderProgramObject = 0;
		}

		/*if(shaderProgramObject != 0){

			int iShaderCount[] = new int[1];
			int iShaderNo;

			GLES32.glUseProgram(shaderProgramObject);
				GLES32.glGetProgramiv(shaderProgramObject, GLES32.GL_ATTACHED_SHADERS, iShaderCount, 0);
				System.out.println("RTR: ShaderCount: "+ iShaderCount[0]);
				int iShaders[] = new int[iShaderCount[0]];
				GLES32.glGetAttachedShaders(shaderProgramObject, iShaderCount[0],
										iShaderCount, 0,
										iShaders, 0);

				for(iShaderNo = 0; iShaderNo < iShaderCount[0] ; iShaderNo++){
					GLES32.glDetachShader(shaderProgramObject, iShaders[iShaderNo]);
					GLES32.glDeleteShader(iShaders[iShaderNo]);
					iShaders[iShaderNo] = 0;
					System.out.println("RTR: Shader Deleted!!");
				}
			GLES32.glUseProgram(0);
			GLES32.glDeleteProgram(shaderProgramObject);
			shaderProgramObject = 0;
		}*/

	}


	private void resize(int width, int height){
		if(height == 0)
			height = 1;

		GLES32.glViewport(0, 0, width, height);

		Matrix.setIdentityM(perspectiveProjectionMatrix, 0);
		Matrix.perspectiveM(perspectiveProjectionMatrix, 0,
						45.0f, 
						(float)width / (float)height,
						0.1f,
						100.0f);
	}



	float translateMatrix[] = new float[4 * 4];
	float modelViewMatrix[] = new float[4 * 4];
	float modelViewProjectionMatrix[] = new float[4 * 4];



	private void display(){

		GLES32.glClear(GLES32.GL_COLOR_BUFFER_BIT | GLES32.GL_DEPTH_BUFFER_BIT);
		
		GLES32.glUseProgram(shaderProgramObject);


			//I
			My_I(-2.0f, 0.0f, -8.0f, 20.0f);

			//N
			My_N(-1.35f, 0.0f, -8.0f, 20.0f);

			//D
			My_D(-0.15f, 0.0f, -8.0f, 20.0f);

			//I
			My_I(1.02f, 0.0f, -8.0f, 20.0f);

			//A
			My_A(2.0f, 0.0f, -8.0f, 20.0f);

			//Flag
			My_Flag(2.0f, 0.0f, -8.0f, 20.0f);



		GLES32.glUseProgram(0);

		requestRender(); 
	}

	private void My_I(float x, float y, float z, float fWidth){

		Matrix.setIdentityM(translateMatrix, 0);
		Matrix.setIdentityM(modelViewMatrix, 0);
		Matrix.setIdentityM(modelViewProjectionMatrix, 0);

		Matrix.translateM(translateMatrix, 0, x, y, z);
		Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, translateMatrix, 0);
		Matrix.multiplyMM(modelViewProjectionMatrix, 0, perspectiveProjectionMatrix, 0, modelViewMatrix, 0);

		GLES32.glUniformMatrix4fv(mvpUniform, 1, false,  modelViewProjectionMatrix, 0);

		GLES32.glLineWidth(15.0f);

		GLES32.glBindVertexArray(vao_I[0]);
			GLES32.glDrawArrays(GLES32.GL_LINES, 0, 6);
		GLES32.glBindVertexArray(0);
	}


	private void My_N(float x, float y, float z, float fWidth){

		Matrix.setIdentityM(translateMatrix, 0);
		Matrix.setIdentityM(modelViewMatrix, 0);
		Matrix.setIdentityM(modelViewProjectionMatrix, 0);

		Matrix.translateM(translateMatrix, 0, x, y, z);
		Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, translateMatrix, 0);
		Matrix.multiplyMM(modelViewProjectionMatrix, 0, perspectiveProjectionMatrix, 0, modelViewMatrix, 0);

		GLES32.glUniformMatrix4fv(mvpUniform, 1, false,  modelViewProjectionMatrix, 0);
		GLES32.glLineWidth(15.0f);

		GLES32.glBindVertexArray(vao_N[0]);
			GLES32.glDrawArrays(GLES32.GL_LINES, 0, 6);
		GLES32.glBindVertexArray(0);
	}


	private void My_D(float x, float y, float z, float fWidth){

		Matrix.setIdentityM(translateMatrix, 0);
		Matrix.setIdentityM(modelViewMatrix, 0);
		Matrix.setIdentityM(modelViewProjectionMatrix, 0);

		Matrix.translateM(translateMatrix, 0, x, y, z);
		Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, translateMatrix, 0);
		Matrix.multiplyMM(modelViewProjectionMatrix, 0, perspectiveProjectionMatrix, 0, modelViewMatrix, 0);

		GLES32.glUniformMatrix4fv(mvpUniform, 1, false,  modelViewProjectionMatrix, 0);
		GLES32.glLineWidth(15.0f);

		GLES32.glBindVertexArray(vao_D[0]);
			GLES32.glDrawArrays(GLES32.GL_LINES, 0, 8);
		GLES32.glBindVertexArray(0);
	}


	private void My_A(float x, float y, float z, float fWidth){

		Matrix.setIdentityM(translateMatrix, 0);
		Matrix.setIdentityM(modelViewMatrix, 0);
		Matrix.setIdentityM(modelViewProjectionMatrix, 0);

		Matrix.translateM(translateMatrix, 0, x, y, z);
		Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, translateMatrix, 0);
		Matrix.multiplyMM(modelViewProjectionMatrix, 0, perspectiveProjectionMatrix, 0, modelViewMatrix, 0);

		GLES32.glUniformMatrix4fv(mvpUniform, 1, false,  modelViewProjectionMatrix, 0);
		GLES32.glLineWidth(15.0f);

		GLES32.glBindVertexArray(vao_A[0]);
			GLES32.glDrawArrays(GLES32.GL_LINES, 0, 4);
		GLES32.glBindVertexArray(0);
	}




	private void My_Flag(float x, float y, float z, float fWidth){

		Matrix.setIdentityM(translateMatrix, 0);
		Matrix.setIdentityM(modelViewMatrix, 0);
		Matrix.setIdentityM(modelViewProjectionMatrix, 0);

		Matrix.translateM(translateMatrix, 0, x, y, z);
		Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, translateMatrix, 0);
		Matrix.multiplyMM(modelViewProjectionMatrix, 0, perspectiveProjectionMatrix, 0, modelViewMatrix, 0);

		GLES32.glUniformMatrix4fv(mvpUniform, 1, false,  modelViewProjectionMatrix, 0);
		GLES32.glLineWidth(15.0f);

		GLES32.glBindVertexArray(vao_Flag[0]);
			GLES32.glDrawArrays(GLES32.GL_LINES, 0, 6);
		GLES32.glBindVertexArray(0);
	}



}

