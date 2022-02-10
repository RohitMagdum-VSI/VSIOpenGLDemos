package com.rohit_r_jadhav.dynamic_india;


import android.opengl.GLSurfaceView;
import android.opengl.GLES32;
import android.opengl.Matrix;
import java.lang.Math;

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
        private float D_Color[] = new float[]{
		1.0f, 0.6f, 0.2f, 0.0f,
		0.0705f, 0.533f, 0.0274f, 0.0f,

		1.0f, 0.6f, 0.2f, 0.0f,
		1.0f, 0.6f, 0.2f, 0.0f,

		0.0705f, 0.533f, 0.0274f, 0.0f,
		0.0705f, 0.533f, 0.0274f, 0.0f,

		1.0f, 0.6f, 0.2f, 0.0f,
		0.0705f, 0.533f, 0.0274f, 0.0f
		};

	private float fD_Fading = 0.0f;



	//For A
        private int vao_A[] = new int[1];
        private int vbo_A_Position[] = new int[1];
        private int vbo_A_Color[] = new int[1];

	//For V A used in INDIA is Without - therfore V verticaly inverted
        private int vao_V[] = new int[1];
        private int vbo_V_Position[] = new int[1];
        private int vbo_V_Color[] = new int[1];

	//For F
        private int vao_F[] = new int[1];
        private int vbo_F_Position[] = new int[1];
        private int vbo_F_Color[] = new int[1];

	//For Flag
        private int vao_Flag[] = new int[1];
        private int vbo_Flag_Position[] = new int[1];
        private int vbo_Flag_Color[] = new int[1];

	//For Plane's Triangle Part
        private int vao_Plane_Triangle[] = new int[1];
        private int vbo_Plane_Triangle_Position[] = new int[1];
        private int vbo_Plane_Triangle_Color[] = new int[1];

	//For Plane's Rectangle Part
        private int vao_Plane_Rect[] = new int[1];
        private int vbo_Plane_Rect_Position[] = new int[1];
        private int vbo_Plane_Rect_Color[] = new int[1];

	//For Plane's Polygon Part
        private int vao_Plane_Polygon[] = new int[1];
        private int vbo_Plane_Polygon_Position[] = new int[1];
        private int vbo_Plane_Polygon_Color[] = new int[1];

	//For Fading Flag
        private int vao_Fading_Flag[] = new int[1];
        private int vbo_Fading_Flag_Position[] = new int[1];
        private int vbo_Fading_Flag_Color[] = new int[1];


	//For Plane Movement and Translation
	private final int NOT_REACH =  0;
        private final int HALF_WAY = 1;
        private final int REACH = 2;
        private final int END = 3;

        private float Plane1_Count = 1000.0f;
        private float Plane2_Count = 1000.0f;
        private float Plane3_Count = 1000.0f;

	private int bPlane1Reached = NOT_REACH;
	private int bPlane2Reached = NOT_REACH;
	private int bPlane3Reached = NOT_REACH;
	private int iFadingFlag1 = 0;
	private int iFadingFlag2 = 0;
	private int iFadingFlag3 = 0;


	//For Sequence
        private int iSequence = 1;


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



		float A_Position[] = new float[]{
			0.0f, 1.06f, 0.0f,
			-0.5f, -1.06f, 0.0f,

			0.0f, 1.06f, 0.0f,
			0.5f, -1.06f, 0.0f,

			-0.250f, 0.0f, 0.0f,
			0.25f, 0.0f, 0.0f
		};


		float A_Color[] = new float[] {
			1.0f, 0.6f, 0.2f,
			0.0705f, 0.533f, 0.0274f,

			1.0f, 0.6f, 0.2f,
			0.0705f, 0.533f, 0.0274f,

			1.0f, 0.6f, 0.2f,
			0.0705f, 0.533f, 0.0274f
		};


		//For Inverted A
		float V_Position[] = new float[]{
			0.0f, 1.06f, 0.0f,
			-0.5f, -1.06f, 0.0f,

			0.0f, 1.06f, 0.0f,
			0.5f, -1.06f, 0.0f
		};


		float V_Color[] = new float[]{
			1.0f, 0.6f, 0.2f,
			0.0705f, 0.533f, 0.0274f,

			1.0f, 0.6f, 0.2f,
			0.0705f, 0.533f, 0.0274f
		};



		float F_Position[] = new float[]{
			0.10f, 1.0f, 0.0f,
			0.10f, -1.0f, 0.0f,

			0.00f, 1.0f, 0.0f,
			0.90f, 1.0f, 0.0f,

			0.10f, 0.1f, 0.0f,
			0.80f, 0.1f, 0.0f
		};


		float F_Color[] = new float[]{
			1.0f, 0.6f, 0.2f,
			0.0705f, 0.533f, 0.0274f,

			1.0f, 0.6f, 0.2f,
			1.0f, 0.6f, 0.2f,

			0.0705f, 0.533f, 0.0274f,
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
			0.0f, 0.0f, 0.0f,
			1.0f, 0.6f, 0.2f,

			0.0f, 0.0f, 0.0f,
			1.0f, 1.0f, 1.0f,

			0.0f, 0.0f, 0.0f,
			0.0705f, 0.533f, 0.0274f
		};


		/***** Plane Parts!! *****/
		float Plane_Triangle_Position[] = new float[]{
			//Front
			5.0f, 0.0f, 0.0f,
			2.50f, 0.65f, 0.0f,
			2.50f, -0.65f, 0.0f
		};

	        float Plane_Triangle_Color[] = new float[]{
			//Front
			0.7294f, 0.8862f, 0.9333f,	//Power Blue
			0.7294f, 0.8862f, 0.9333f,
			0.7294f, 0.8862f, 0.9333f
		};


	        float Plane_Rect_Position[] = new float[]{
			//Middle
			2.50f, 0.65f, 0.0f,
			-2.50f, 0.65f, 0.0f,
			-2.50f, -0.65f, 0.0f,
			2.50f, -0.65f, 0.0f,

			//Upper_Fin
			0.75f, 0.65f, 0.0f,
			-1.20f, 2.5f, 0.0f,
			-2.50f, 2.5f, 0.0f,
			-2.0f, 0.65f, 0.0f,

			//Lower_Fin
			0.75f, -0.65f, 0.0f,
			-1.20f, -2.50f, 0.0f,
			-2.50f, -2.50f, 0.0f,
			-2.0f, -0.65f, 0.0f,

			//Back
			-2.50f, 0.65f, 0.0f,
			-3.0f, 0.75f, 0.0f,
			-3.0f, -0.75f, 0.0f,
			-2.5f, -0.65f, 0.0f
		};


	        float Plane_Rect_Color[] = new float[]{
			//Middle
			0.7294f, 0.8862f, 0.9333f,
			0.7294f, 0.8862f, 0.9333f,
			0.7294f, 0.8862f, 0.9333f,
			0.7294f, 0.8862f, 0.9333f,

			//Upper_Fin
			0.7294f, 0.8862f, 0.9333f,
			0.7294f, 0.8862f, 0.9333f,
			0.7294f, 0.8862f, 0.9333f,
			0.7294f, 0.8862f, 0.9333f,

			//Lower_Fin
			0.7294f, 0.8862f, 0.9333f,
			0.7294f, 0.8862f, 0.9333f,
			0.7294f, 0.8862f, 0.9333f,
			0.7294f, 0.8862f, 0.9333f,

			//Back
			0.7294f, 0.8862f, 0.9333f,
			0.7294f, 0.8862f, 0.9333f,
			0.7294f, 0.8862f, 0.9333f,
			0.7294f, 0.8862f, 0.9333f
		};

	        float Plane_Polygon_Position[] = new float[]{
			//Upper Tail
			-3.0f, 0.75f, 0.0f,
			-3.90f, 1.5f, 0.0f,
			-4.5f, 1.5f, 0.0f,
			-4.0f, 0.0f, 0.0f,
			-3.0f, 0.0f, 0.0f,

			//Lower Tail
			-3.0f, -0.75f, 0.0f,
			-3.90f, -1.5f, 0.0f,
			-4.5f, -1.5f, 0.0f,
			-4.0f, 0.0f, 0.0f,
			-3.0f, 0.0f, 0.0f
		};

	        float Plane_Polygon_Color[] = new float[]{
			//Upper Tail
			0.7294f, 0.8862f, 0.9333f,
			0.7294f, 0.8862f, 0.9333f,
			0.7294f, 0.8862f, 0.9333f,
			0.7294f, 0.8862f, 0.9333f,
			0.7294f, 0.8862f, 0.9333f,

			//Lower Tail
			0.7294f, 0.8862f, 0.9333f,
			0.7294f, 0.8862f, 0.9333f,
			0.7294f, 0.8862f, 0.9333f,
			0.7294f, 0.8862f, 0.9333f,
			0.7294f, 0.8862f, 0.9333f
		};


	        float Fading_Flag_Color[] = new float[]{
			0.0f, 0.0f, 0.0f,
			1.0f, 0.6f, 0.2f,

			0.0f, 0.0f, 0.0f,
			1.0f, 1.0f, 1.0f,

			0.0f, 0.0f, 0.0f,
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




		/********** D  Fading!! **********/
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

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER, 
							8 * 4 * 4,
							null,
		/****/				GLES32.GL_DYNAMIC_DRAW);
			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_COLOR,
									4,
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




		/********** V Inverted A ***********/
		GLES32.glGenVertexArrays(1, vao_V, 0);
		GLES32.glBindVertexArray(vao_V[0]);

			/********** Position **********/
			GLES32.glGenBuffers(1, vbo_V_Position, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_V_Position[0]);

				/********** For glBufferData ***********/
				ByteBuffer vPosition_ByteBuffer = ByteBuffer.allocateDirect(V_Position.length * 4);
				vPosition_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer vPosition_FloatBuffer = vPosition_ByteBuffer.asFloatBuffer();
				vPosition_FloatBuffer.put(V_Position);
				vPosition_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							V_Position.length * 4,
							vPosition_FloatBuffer,
							GLES32.GL_STATIC_DRAW);

			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_POSITION,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

			/********** Color **********/
			GLES32.glGenBuffers(1, vbo_V_Color, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_V_Color[0]);

				/********** For glBufferData **********/
				ByteBuffer vColor_ByteBuffer = ByteBuffer.allocateDirect(V_Color.length * 4);
				vColor_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer vColor_FloatBuffer = vColor_ByteBuffer.asFloatBuffer();
				vColor_FloatBuffer.put(V_Color);
				vColor_FloatBuffer.position(0);

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






		/********** F **********/
		GLES32.glGenVertexArrays(1, vao_F, 0);
		GLES32.glBindVertexArray(vao_F[0]);

			/********** Position **********/
			GLES32.glGenBuffers(1, vbo_F_Position, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_F_Position[0]);

				/********** For glBufferData ***********/
				ByteBuffer fPosition_ByteBuffer = ByteBuffer.allocateDirect(F_Position.length * 4);
				fPosition_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer fPosition_FloatBuffer = fPosition_ByteBuffer.asFloatBuffer();
				fPosition_FloatBuffer.put(F_Position);
				fPosition_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							F_Position.length * 4,
							fPosition_FloatBuffer,
							GLES32.GL_STATIC_DRAW);

			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_POSITION,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

			/********** Color **********/
			GLES32.glGenBuffers(1, vbo_F_Color, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_F_Color[0]);

				/********** For glBufferData **********/
				ByteBuffer fColor_ByteBuffer = ByteBuffer.allocateDirect(F_Color.length * 4);
				fColor_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer fColor_FloatBuffer = fColor_ByteBuffer.asFloatBuffer();
				fColor_FloatBuffer.put(F_Color);
				fColor_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER, 
							F_Color.length * 4,
							fColor_FloatBuffer,
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



		/******************** Plane Parts ********************/


		/********** Triangle **********/
		GLES32.glGenVertexArrays(1, vao_Plane_Triangle, 0);
		GLES32.glBindVertexArray(vao_Plane_Triangle[0]);

			/********** Position **********/
			GLES32.glGenBuffers(1, vbo_Plane_Triangle_Position, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Plane_Triangle_Position[0]);

				/********** For glBufferData ***********/
				ByteBuffer planeTrianglePosition_ByteBuffer = ByteBuffer.allocateDirect(Plane_Triangle_Position.length * 4);
				planeTrianglePosition_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer planeTrianglePosition_FloatBuffer = planeTrianglePosition_ByteBuffer.asFloatBuffer();
				planeTrianglePosition_FloatBuffer.put(Plane_Triangle_Position);
				planeTrianglePosition_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							Plane_Triangle_Position.length * 4,
							planeTrianglePosition_FloatBuffer,
							GLES32.GL_STATIC_DRAW);

			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_POSITION,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

			/********** Color **********/
			GLES32.glGenBuffers(1, vbo_Plane_Triangle_Color, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Plane_Triangle_Color[0]);

				/********** For glBufferData **********/
				ByteBuffer planeTriangleColor_ByteBuffer = ByteBuffer.allocateDirect(Plane_Triangle_Color.length * 4);
				planeTriangleColor_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer planeTriangleColor_FloatBuffer = planeTriangleColor_ByteBuffer.asFloatBuffer();
				planeTriangleColor_FloatBuffer.put(Plane_Triangle_Color);
				planeTriangleColor_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER, 
							Plane_Triangle_Color.length * 4,
							planeTriangleColor_FloatBuffer,
							GLES32.GL_STATIC_DRAW);
			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_COLOR,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_COLOR);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		GLES32.glBindVertexArray(0);




		/********** Rectangle **********/
		GLES32.glGenVertexArrays(1, vao_Plane_Rect, 0);
		GLES32.glBindVertexArray(vao_Plane_Rect[0]);

			/********** Position **********/
			GLES32.glGenBuffers(1, vbo_Plane_Rect_Position, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Plane_Rect_Position[0]);

				/********** For glBufferData ***********/
				ByteBuffer planeRectPosition_ByteBuffer = ByteBuffer.allocateDirect(Plane_Rect_Position.length * 4);
				planeRectPosition_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer planeRectPosition_FloatBuffer = planeRectPosition_ByteBuffer.asFloatBuffer();
				planeRectPosition_FloatBuffer.put(Plane_Rect_Position);
				planeRectPosition_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							Plane_Rect_Position.length * 4,
							planeRectPosition_FloatBuffer,
							GLES32.GL_STATIC_DRAW);

			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_POSITION,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

			/********** Color **********/
			GLES32.glGenBuffers(1, vbo_Plane_Rect_Color, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Plane_Rect_Color[0]);

				/********** For glBufferData **********/
				ByteBuffer planeRectColor_ByteBuffer = ByteBuffer.allocateDirect(Plane_Rect_Color.length * 4);
				planeRectColor_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer planeRectColor_FloatBuffer = planeRectColor_ByteBuffer.asFloatBuffer();
				planeRectColor_FloatBuffer.put(Plane_Rect_Color);
				planeRectColor_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER, 
							Plane_Rect_Color.length * 4,
							planeRectColor_FloatBuffer,
							GLES32.GL_STATIC_DRAW);
			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_COLOR,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_COLOR);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		GLES32.glBindVertexArray(0);




		/********** Polygon **********/
		GLES32.glGenVertexArrays(1, vao_Plane_Polygon, 0);
		GLES32.glBindVertexArray(vao_Plane_Polygon[0]);

			/********** Position **********/
			GLES32.glGenBuffers(1, vbo_Plane_Polygon_Position, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Plane_Polygon_Position[0]);

				/********** For glBufferData ***********/
				ByteBuffer planePolygonPosition_ByteBuffer = ByteBuffer.allocateDirect(Plane_Polygon_Position.length * 4);
				planePolygonPosition_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer planePolygonPosition_FloatBuffer = planePolygonPosition_ByteBuffer.asFloatBuffer();
				planePolygonPosition_FloatBuffer.put(Plane_Polygon_Position);
				planePolygonPosition_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							Plane_Polygon_Position.length * 4,
							planePolygonPosition_FloatBuffer,
							GLES32.GL_STATIC_DRAW);

			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_POSITION,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

			/********** Color **********/
			GLES32.glGenBuffers(1, vbo_Plane_Polygon_Color, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Plane_Polygon_Color[0]);

				/********** For glBufferData **********/
				ByteBuffer planePolygonColor_ByteBuffer = ByteBuffer.allocateDirect(Plane_Polygon_Color.length * 4);
				planePolygonColor_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer planePolygonColor_FloatBuffer = planePolygonColor_ByteBuffer.asFloatBuffer();
				planePolygonColor_FloatBuffer.put(Plane_Polygon_Color);
				planePolygonColor_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER, 
							Plane_Polygon_Color.length * 4,
							planePolygonColor_FloatBuffer,
							GLES32.GL_STATIC_DRAW);
			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_COLOR,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_COLOR);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		GLES32.glBindVertexArray(0);



		/********** Fading Flag **********/
		GLES32.glGenVertexArrays(1, vao_Fading_Flag, 0);
		GLES32.glBindVertexArray(vao_Fading_Flag[0]);

			/********** Position **********/
			GLES32.glGenBuffers(1, vbo_Fading_Flag_Position, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Fading_Flag_Position[0]);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							3 * 6 * 4,
							null,
		/****/				GLES32.GL_DYNAMIC_DRAW);

			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_POSITION,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

			/********** Color **********/
			GLES32.glGenBuffers(1, vbo_Fading_Flag_Color, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Fading_Flag_Color[0]);

				/********** For glBufferData **********/
				ByteBuffer fadingflagColor_ByteBuffer = ByteBuffer.allocateDirect(Fading_Flag_Color.length * 4);
				fadingflagColor_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer fadingflagColor_FloatBuffer = fadingflagColor_ByteBuffer.asFloatBuffer();
				fadingflagColor_FloatBuffer.put(Fading_Flag_Color);
				fadingflagColor_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER, 
							Flag_Color.length * 4,
							fadingflagColor_FloatBuffer,
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

		GLES32.glEnable(GLES32.GL_BLEND);
		GLES32.glBlendFunc(GLES32.GL_SRC_ALPHA, GLES32.GL_ONE_MINUS_SRC_ALPHA);



		GLES32.glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	}


	private void uninitialize(){


		//Fading Flag
		if (vbo_Fading_Flag_Color[0] != 0) {
			GLES32.glDeleteBuffers(1, vbo_Fading_Flag_Color, 0);
			vbo_Fading_Flag_Color[0] = 0;
		}

		if (vbo_Fading_Flag_Position[0] != 0) {
			GLES32.glDeleteBuffers(1, vbo_Fading_Flag_Position, 0);
			vbo_Fading_Flag_Position[0] = 0;
		}

		if (vao_Fading_Flag[0] != 0) {
			GLES32.glDeleteVertexArrays(1, vao_Fading_Flag, 0);
			vao_Fading_Flag[0] = 0;
		}


		//Plane Polygon
		if(vbo_Plane_Polygon_Color[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Plane_Polygon_Color, 0);
			vbo_Plane_Polygon_Color[0] = 0; 
		}

		if(vbo_Plane_Polygon_Position[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Plane_Polygon_Position, 0);
			vbo_Plane_Polygon_Position[0] = 0;
		}

		if(vao_Plane_Polygon[0] != 0){
			GLES32.glDeleteVertexArrays(1, vao_Plane_Polygon, 0);
			vao_Plane_Polygon[0] = 0;
		}


		//Plane Rect
		if(vbo_Plane_Rect_Color[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Plane_Rect_Color, 0);
			vbo_Plane_Rect_Color[0] = 0; 
		}

		if(vbo_Plane_Rect_Position[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Plane_Rect_Position, 0);
			vbo_Plane_Rect_Position[0] = 0;
		}

		if(vao_Plane_Rect[0] != 0){
			GLES32.glDeleteVertexArrays(1, vao_Plane_Rect, 0);
			vao_Plane_Rect[0] = 0;
		}



		//Plane Triangle
		if(vbo_Plane_Triangle_Color[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Plane_Triangle_Color, 0);
			vbo_Plane_Triangle_Color[0] = 0; 
		}

		if(vbo_Plane_Triangle_Position[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Plane_Triangle_Position, 0);
			vbo_Plane_Triangle_Position[0] = 0;
		}

		if(vao_Plane_Triangle[0] != 0){
			GLES32.glDeleteVertexArrays(1, vao_Plane_Triangle, 0);
			vao_Plane_Triangle[0] = 0;
		}

		
		

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


		//F
		if (vbo_F_Color[0] != 0) {
			GLES32.glDeleteBuffers(1, vbo_F_Color, 0);
			vbo_F_Color[0] = 0;
		}

		if (vbo_F_Position[0] != 0) {
			GLES32.glDeleteBuffers(1, vbo_F_Position, 0);
			vbo_F_Position[0] = 0;
		}

		if (vao_F[0] != 0) {
			GLES32.glDeleteVertexArrays(1, vao_F, 0);
			vao_F[0] = 0;
		}



		//V Inverted A
		if (vbo_V_Color[0] != 0) {
			GLES32.glDeleteBuffers(1, vbo_V_Color, 0);
			vbo_V_Color[0] = 0;
		}

		if (vbo_V_Position[0] != 0) {
			GLES32.glDeleteBuffers(1, vbo_V_Position, 0);
			vbo_V_Position[0] = 0;
		}

		if (vao_V[0] != 0) {
			GLES32.glDeleteVertexArrays(1, vao_V, 0);
			vao_V[0] = 0;
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
	float scaleMatrix[] = new float[4 * 4];
	float rotateMatrix[] = new float[4 * 4];
	float modelViewMatrix[] = new float[4 * 4];
	float modelViewProjectionMatrix[] = new float[4 * 4];
	//For India
	float fXTranslation = 0.0f;
	float fYTranslation = 0.0f;

	//For Plane
	float angle_Plane1 = (float)(Math.PI);
	float angle_Plane3 = (float)(Math.PI);

	float XTrans_Plane1 = 0.0f;
	float YTrans_Plane1 = 0.0f;

	float XTrans_Plane2 = 0.0f;

	float XTrans_Plane3 = 0.0f;
	float YTrans_Plane3 = 0.0f;

	float ZRot_Plane1 = -60.0f;
	float ZRot_Plane3 = 60.0f;



	private void display(){

		GLES32.glClear(GLES32.GL_COLOR_BUFFER_BIT | GLES32.GL_DEPTH_BUFFER_BIT);

		
		GLES32.glUseProgram(shaderProgramObject);


			switch (iSequence) {
				case 1:

					My_Letters('I', -7.50f + fXTranslation, 0.0f, -8.0f, 20.0f);
					fXTranslation = fXTranslation + 0.015f;;
					if ((-7.5f + fXTranslation) >= -2.0f) {
						fXTranslation = 0.0f;
						iSequence = 2;
					}
					break;

				case 2:
					My_Letters('I', -2.0f, 0.0f, -8.0f, 20.0f);
					My_Letters('V', 8.50f - fXTranslation, 0.0f, -8.0f, 20.0f);
					fXTranslation = fXTranslation + 0.015f;;
					if ((8.5f - fXTranslation) <= 2.0f) {
						fXTranslation = 0.0f;
						iSequence = 3;
					}
					break;

				case 3:
					My_Letters('I', -2.0f, 0.0f, -8.0f, 20.0f);
					My_Letters('V', 2.0f, 0.0f, -8.0f, 20.0f);
					My_Letters('N', -1.35f, (6.0f - fYTranslation), -8.0f, 20.0f);
					fYTranslation = fYTranslation + 0.015f;
					if ((6.0f - fYTranslation) < 0.0f) {
						fYTranslation = 0.0f;
						iSequence = 4;
					}
					break;

				case 4:
					My_Letters('I', -2.0f, 0.0f, -8.0f, 20.0f);
					My_Letters('V', 2.0f, 0.0f, -8.0f, 20.0f);
					My_Letters('N', -1.35f, 0.0f, -8.0f, 20.0f);
					My_Letters('I', 1.02f, (-5.0f + fYTranslation), -8.0f, 20.0f);
					fYTranslation = fYTranslation + 0.015f;
					if ((-5.0f + fYTranslation) > 0.0f) {
						fYTranslation = 0.0f;
						iSequence = 5;
					}
					break;

				case 5:

					D_Color[3] = fD_Fading;
					D_Color[7] = fD_Fading;
					D_Color[11] = fD_Fading;
					D_Color[15] = fD_Fading;
					D_Color[19] = fD_Fading;
					D_Color[23] = fD_Fading;
					D_Color[27] = fD_Fading;
					D_Color[31] = fD_Fading;

					My_Letters('I', -2.0f, 0.0f, -8.0f, 20.0f);
					My_Letters('N', -1.35f, 0.0f, -8.0f, 20.0f);
					My_D(-0.15f, 0.0f, -8.0f, 20.0f);
					My_Letters('I', 1.02f, 0.0f, -8.0f, 20.0f);
					My_Letters('V', 2.0f, 0.0f, -8.0f, 20.0f);

					if (fD_Fading > 1.0f) {
						iSequence = 6;
					}
					else
						fD_Fading = fD_Fading + 0.001f;
					break;


				case 6:
					My_Letters('I', -2.0f, 0.0f, -8.0f, 20.0f);
					My_Letters('N', -1.35f, 0.0f, -8.0f, 20.0f);
					My_D(-0.15f, 0.0f, -8.0f, 20.0f);
					My_Letters('I', 1.02f, 0.0f, -8.0f, 20.0f);
					My_Letters('V', 2.0f, 0.0f, -8.0f, 20.0f);



					/********** Plane 1 **********/
					if (bPlane1Reached == NOT_REACH) {
						XTrans_Plane1 = (float)((3.2 * Math.cos(angle_Plane1)) + (-2.5f));
						YTrans_Plane1 = (float)((4.0f * Math.sin(angle_Plane1)) + (4.0f));
						angle_Plane1 = angle_Plane1 + 0.005f;
						ZRot_Plane1 = ZRot_Plane1 + 0.2f;


						if (angle_Plane1 >= (3.0f * Math.PI) / 2.0f) {
							bPlane1Reached = HALF_WAY;
							YTrans_Plane1 = 0.00f;

						}
						else if (ZRot_Plane1 >= 0.0)
							ZRot_Plane1 = 0.0f;

					}
					else if (bPlane1Reached == HALF_WAY) {
						XTrans_Plane1 = XTrans_Plane1 + 0.010f;
						YTrans_Plane1 = 0.00f;

						if (XTrans_Plane1 >= 3.00f) {	//2.6
							bPlane1Reached = REACH;
							angle_Plane1 = (float)(3.0f * Math.PI) / 2.0f;
							ZRot_Plane1 = 0.0f;
						}
					}
					else if (bPlane1Reached == REACH) {

						if (Plane1_Count <= 0.0f) {
							iFadingFlag1 = 2;
							XTrans_Plane1 = (float)((3.0f * Math.cos(angle_Plane1)) + (3.0f));		//2.6
							YTrans_Plane1 = (float)((4.0f * Math.sin(angle_Plane1)) + (4.0f));

							if (XTrans_Plane1 >= 6.00f || YTrans_Plane1 >= 4.0f)
								bPlane1Reached = END;

								angle_Plane1 = angle_Plane1 + 0.005f;
								ZRot_Plane1 = ZRot_Plane1 + 0.2f;
							}
							else
								iFadingFlag1 = 1;

							Plane1_Count = Plane1_Count - 1.0f;
						}
					else if (bPlane1Reached == END) {
						angle_Plane1 = 0.0f;
						ZRot_Plane1 = 0.0f;
					}

		
					/*********** Fading Flag ***********/
					if (bPlane1Reached == NOT_REACH)
						My_Fading_Flag(XTrans_Plane1, YTrans_Plane1, -8.0f, ZRot_Plane1);

					My_Plane(XTrans_Plane1, YTrans_Plane1, -8.0f, 0.18f, 0.18f, 0.0f, ZRot_Plane1);





					/********** Plane 2 **********/
					if (bPlane2Reached == NOT_REACH) {
						if ((-6.0f + XTrans_Plane2) > -2.50f) {
							bPlane2Reached = HALF_WAY;
						}
						else
							XTrans_Plane2 = XTrans_Plane2 + 0.011f;

					}
					else if (bPlane2Reached == HALF_WAY) {
						XTrans_Plane2 = XTrans_Plane2 + 0.010f;
						if ((-6.0f + XTrans_Plane2) >= 3.0f) {	//2.6
							bPlane2Reached = REACH;
						}
					}
					else if (bPlane2Reached == REACH) {
						if (Plane2_Count <= 0.00f) {
							iFadingFlag2 = 2;
							XTrans_Plane2 = XTrans_Plane2 + 0.010f;
						}
						else
							iFadingFlag2 = 1;


						if ((-6.0f + XTrans_Plane2) >= 8.0f)
							bPlane2Reached = END;


						Plane2_Count = Plane2_Count - 1.0f;
					}
					else if (bPlane2Reached == END) {
						XTrans_Plane2 = 14.0f;
					}

					/*********** Fading_Flag **********/
					if (iFadingFlag2 < 2)
						My_Fading_Flag((-6.0f + XTrans_Plane2), 0.0f, -8.0f, 0.0f);

					My_Plane((-6.0f + XTrans_Plane2), 0.0f, -8.0f, 0.18f, 0.18f, 0.0f, 0.0f);







					/********** Plane 3 **********/
					if (bPlane3Reached == NOT_REACH) {
						XTrans_Plane3 = (float)((3.2 * Math.cos(angle_Plane3)) + (-2.5f));
						YTrans_Plane3 = (float)((4.0f * Math.sin(angle_Plane3)) + (-4.0f));
						angle_Plane3 = angle_Plane3 - 0.005f;
						ZRot_Plane3 = ZRot_Plane3 - 0.2f;


						if (angle_Plane3 < (Math.PI) / 2.0f) {
							bPlane3Reached = HALF_WAY;
							YTrans_Plane3 = 0.00f;

						}
						else if (ZRot_Plane3 < 0.0)
							ZRot_Plane3 = 0.0f;

					}
					else if (bPlane3Reached == HALF_WAY) {
						XTrans_Plane3 = XTrans_Plane3 + 0.010f;
						YTrans_Plane3 = 0.00f;

						if (XTrans_Plane3 >= 3.00f) {	//2.6
							bPlane3Reached = REACH;
							angle_Plane3 = (float)(Math.PI) / 2.0f;
							ZRot_Plane3 = 0.0f;
						}
					}
					else if (bPlane3Reached == REACH) {

						if (Plane3_Count <= 0.0f) {
							iFadingFlag3 = 2;
							XTrans_Plane3 = (float)((3.0f * Math.cos(angle_Plane3)) + (3.0f));		//2.6
							YTrans_Plane3 = (float)((4.0f * Math.sin(angle_Plane3)) + (-4.0f));

							if (XTrans_Plane3 >= 6.00f || YTrans_Plane3 < -4.0f)
								bPlane3Reached = END;

							angle_Plane3 = angle_Plane3 - 0.005f;
							ZRot_Plane3 = ZRot_Plane3 - 0.2f;
						}
						else
							iFadingFlag3 = 1;

						Plane3_Count = Plane3_Count - 1.0f;
					}
					else if (bPlane3Reached == END) {
						angle_Plane3 = 0.0f;
						ZRot_Plane3 = 0.0f;
					}



					/*********** Fading Flag ***********/
					if (bPlane2Reached == NOT_REACH)
						My_Fading_Flag(XTrans_Plane3, YTrans_Plane3, -8.0f, ZRot_Plane3);


					My_Plane(XTrans_Plane3, YTrans_Plane3, -8.0f, 0.18f, 0.18f, 0.0f, ZRot_Plane3);


					if (iFadingFlag1 == 2 || iFadingFlag2 == 2 || iFadingFlag3 == 2)
						My_Flag(2.0f, 0.0f, -8.0f, 30.0f);

					break;


				}


		GLES32.glUseProgram(0);

		requestRender(); 
	}



	private void My_Letters(char  c,float x, float y, float z, float fWidth){

		Matrix.setIdentityM(translateMatrix, 0);
		Matrix.setIdentityM(modelViewMatrix, 0);
		Matrix.setIdentityM(modelViewProjectionMatrix, 0);

		Matrix.translateM(translateMatrix, 0, x, y, z);
		Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, translateMatrix, 0);
		Matrix.multiplyMM(modelViewProjectionMatrix, 0, perspectiveProjectionMatrix, 0, modelViewMatrix, 0);

		GLES32.glUniformMatrix4fv(mvpUniform, 1, false,  modelViewProjectionMatrix, 0);

		GLES32.glLineWidth(30.0f);
		switch(c){
			case 'I':
				GLES32.glBindVertexArray(vao_I[0]);
				GLES32.glDrawArrays(GLES32.GL_LINES, 0, 6);
				GLES32.glBindVertexArray(0);
				break;

			case 'N':
				GLES32.glBindVertexArray(vao_N[0]);
				GLES32.glDrawArrays(GLES32.GL_LINES, 0, 6);
				GLES32.glBindVertexArray(0);
				break;

			case 'A':
				GLES32.glBindVertexArray(vao_A[0]);
				GLES32.glDrawArrays(GLES32.GL_LINES, 0, 6);
				GLES32.glBindVertexArray(0);
				break;

			case 'V':
				GLES32.glBindVertexArray(vao_V[0]);
				GLES32.glDrawArrays(GLES32.GL_LINES, 0, 4);
				GLES32.glBindVertexArray(0);
				break;

			case 'F':
				GLES32.glBindVertexArray(vao_F[0]);
				GLES32.glDrawArrays(GLES32.GL_LINES, 0, 6);
				GLES32.glBindVertexArray(0);
				break;
		}

		
	}


	private void My_D(float x, float y, float z, float fWidth){

		Matrix.setIdentityM(translateMatrix, 0);
		Matrix.setIdentityM(modelViewMatrix, 0);
		Matrix.setIdentityM(modelViewProjectionMatrix, 0);

		Matrix.translateM(translateMatrix, 0, x, y, z);
		Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, translateMatrix, 0);
		Matrix.multiplyMM(modelViewProjectionMatrix, 0, perspectiveProjectionMatrix, 0, modelViewMatrix, 0);

		GLES32.glUniformMatrix4fv(mvpUniform, 1, false,  modelViewProjectionMatrix, 0);
		GLES32.glLineWidth(30.0f);

		GLES32.glBindVertexArray(vao_D[0]);

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
							GLES32.GL_DYNAMIC_DRAW);

			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

			GLES32.glDrawArrays(GLES32.GL_LINES, 0, 8);
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
		GLES32.glLineWidth(30.0f);

		GLES32.glBindVertexArray(vao_Flag[0]);
			GLES32.glDrawArrays(GLES32.GL_LINES, 0, 6);
		GLES32.glBindVertexArray(0);
	}


		


		float Fading_Flag_Position[] = new float[]{
			-1.0f, 0.09f, 0.0f,
			-0.50f, 0.09f, 0.0f,

			-1.0f, 0.0f, 0.0f,
			-0.50f, 0.0f, 0.0f,

			-1.0f, -0.09f, 0.0f,
			-0.50f, -0.09f, 0.0f
		};

	private void My_Fading_Flag(float x, float y, float z, float fAngle) {


		

		if (bPlane2Reached != REACH) {
			Fading_Flag_Position[0] -= 0.005f;
			Fading_Flag_Position[6] -= 0.005f;
			Fading_Flag_Position[12] -= 0.005f;

		}
		else if (bPlane2Reached == REACH) {

			Fading_Flag_Position[0] += 0.007f;
			Fading_Flag_Position[6] += 0.007f;
			Fading_Flag_Position[12] += 0.007f;
		}

		
		Matrix.setIdentityM(translateMatrix, 0);
		Matrix.setIdentityM(rotateMatrix, 0);
		Matrix.setIdentityM(modelViewMatrix, 0);
		Matrix.setIdentityM(modelViewProjectionMatrix, 0);

		Matrix.translateM(translateMatrix, 0, x, y, z);
		Matrix.rotateM(rotateMatrix, 0, fAngle, 0.0f, 0.0f, 1.0f);
		Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, translateMatrix, 0);
		Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, rotateMatrix, 0);
		Matrix.multiplyMM(modelViewProjectionMatrix, 0, perspectiveProjectionMatrix, 0, modelViewMatrix, 0);

		GLES32.glUniformMatrix4fv(mvpUniform,
			1,
			false,
			modelViewProjectionMatrix, 0);

		GLES32.glLineWidth(30.0f);

		GLES32.glBindVertexArray(vao_Fading_Flag[0]);

			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Fading_Flag_Position[0]);

				/********** For glBufferData ***********/
				ByteBuffer fadingflagPosition_ByteBuffer = ByteBuffer.allocateDirect(Fading_Flag_Position.length * 4);
				fadingflagPosition_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer fadingflagPosition_FloatBuffer = fadingflagPosition_ByteBuffer.asFloatBuffer();
				fadingflagPosition_FloatBuffer.put(Fading_Flag_Position);
				fadingflagPosition_FloatBuffer.position(0);


			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
				Fading_Flag_Position.length * 4,
				fadingflagPosition_FloatBuffer,
				GLES32.GL_DYNAMIC_DRAW);

			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

			GLES32.glDrawArrays(GLES32.GL_LINES, 0, 6);

		GLES32.glBindVertexArray(0);

	}





	private void My_Plane(float x, float y, float z, float scaleX, float scaleY, float scaleZ, float ZRot_Angle) {



		Matrix.setIdentityM(translateMatrix, 0);
		Matrix.setIdentityM(scaleMatrix, 0);
		Matrix.setIdentityM(rotateMatrix, 0);
		Matrix.setIdentityM(modelViewMatrix, 0);
		Matrix.setIdentityM(modelViewProjectionMatrix, 0);

		Matrix.translateM(translateMatrix, 0, x, y, z);
		Matrix.scaleM(scaleMatrix, 0, scaleX, scaleY, scaleZ);
		Matrix.rotateM(rotateMatrix, 0, ZRot_Angle, 0.0f, 0.0f, 1.0f);
		Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, translateMatrix, 0);
		Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, scaleMatrix, 0);
		Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, rotateMatrix, 0);
		Matrix.multiplyMM(modelViewProjectionMatrix, 0, perspectiveProjectionMatrix, 0, modelViewMatrix, 0);


		GLES32.glUniformMatrix4fv(mvpUniform, 1, false, modelViewProjectionMatrix, 0);


		//Triangle
		GLES32.glBindVertexArray(vao_Plane_Triangle[0]);
		GLES32.glDrawArrays(GLES32.GL_TRIANGLES, 0, 3);
		GLES32.glBindVertexArray(0);

		//Rectangle
		GLES32.glBindVertexArray(vao_Plane_Rect[0]);

			//For Middle
			GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 0, 4);

			//For Upper and Lower Fin
			GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 4, 8);

			//For Back
			GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 12, 4);
		GLES32.glBindVertexArray(0);


		//Polygon
		GLES32.glBindVertexArray(vao_Plane_Polygon[0]);
		GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 0, 10);
		GLES32.glBindVertexArray(0);



		//I



		Matrix.setIdentityM(translateMatrix, 0);
		Matrix.setIdentityM(scaleMatrix, 0);
		Matrix.setIdentityM(rotateMatrix, 0);
		Matrix.setIdentityM(modelViewMatrix, 0);
		Matrix.setIdentityM(modelViewProjectionMatrix, 0);

		Matrix.translateM(translateMatrix, 0, x - 0.30f, y, z);
		Matrix.scaleM(scaleMatrix, 0, scaleX - 0.05f, scaleY - 0.05f, scaleZ);
		Matrix.rotateM(rotateMatrix, 0, ZRot_Angle, 0.0f, 0.0f, 1.0f);
		Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, translateMatrix, 0);
		Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, scaleMatrix, 0);
		Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, rotateMatrix, 0);
		Matrix.multiplyMM(modelViewProjectionMatrix, 0, perspectiveProjectionMatrix, 0, modelViewMatrix, 0);


		GLES32.glUniformMatrix4fv(mvpUniform,
			1,
			false,
			modelViewProjectionMatrix, 0);

		GLES32.glBindVertexArray(vao_I[0]);
		GLES32.glDrawArrays(GLES32.GL_LINES, 0, 6);
		GLES32.glBindVertexArray(0);






		//A


		Matrix.setIdentityM(translateMatrix, 0);
		Matrix.setIdentityM(scaleMatrix, 0);
		Matrix.setIdentityM(rotateMatrix, 0);
		Matrix.setIdentityM(modelViewMatrix, 0);
		Matrix.setIdentityM(modelViewProjectionMatrix, 0);

		Matrix.translateM(translateMatrix, 0, x - 0.15f, y, z);
		Matrix.scaleM(scaleMatrix, 0, scaleX - 0.05f, scaleY - 0.05f, scaleZ);
		Matrix.rotateM(rotateMatrix, 0, ZRot_Angle, 0.0f, 0.0f, 1.0f);
		Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, translateMatrix, 0);
		Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, scaleMatrix, 0);
		Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, rotateMatrix, 0);
		Matrix.multiplyMM(modelViewProjectionMatrix, 0, perspectiveProjectionMatrix, 0, modelViewMatrix, 0);

		GLES32.glUniformMatrix4fv(mvpUniform,
			1,
			false,
			modelViewProjectionMatrix, 0);

		GLES32.glBindVertexArray(vao_A[0]);
		GLES32.glDrawArrays(GLES32.GL_LINES, 0, 6);
		GLES32.glBindVertexArray(0);




		//F

		
		Matrix.setIdentityM(translateMatrix, 0);
		Matrix.setIdentityM(scaleMatrix, 0);
		Matrix.setIdentityM(rotateMatrix, 0);
		Matrix.setIdentityM(modelViewMatrix, 0);
		Matrix.setIdentityM(modelViewProjectionMatrix, 0);

		Matrix.translateM(translateMatrix, 0, x - 0.050f, y, z);
		Matrix.scaleM(scaleMatrix, 0, scaleX - 0.05f, scaleY - 0.05f, scaleZ);
		Matrix.rotateM(rotateMatrix, 0, ZRot_Angle, 0.0f, 0.0f, 1.0f);
		Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, translateMatrix, 0);
		Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, scaleMatrix, 0);
		Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, rotateMatrix, 0);
		Matrix.multiplyMM(modelViewProjectionMatrix, 0, perspectiveProjectionMatrix, 0, modelViewMatrix, 0);

		GLES32.glUniformMatrix4fv(mvpUniform,
			1,
			false,
			modelViewProjectionMatrix, 0);

		GLES32.glBindVertexArray(vao_F[0]);
		GLES32.glDrawArrays(GLES32.GL_LINES, 0, 6);
		GLES32.glBindVertexArray(0);

	}







}

