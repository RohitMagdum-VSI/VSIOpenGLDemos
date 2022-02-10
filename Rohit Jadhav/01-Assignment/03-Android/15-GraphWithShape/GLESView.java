package com.rohit_r_jadhav.graph_withshape;

import android.opengl.GLSurfaceView;
import android.opengl.GLES32;
import android.opengl.Matrix;

import java.lang.Math;

import javax.microedition.khronos.opengles.GL10;
import javax.microedition.khronos.egl.EGLConfig;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import android.view.MotionEvent;
import android.view.GestureDetector;
import android.view.GestureDetector.OnGestureListener;
import android.view.GestureDetector.OnDoubleTapListener;
import android.content.Context;

public class GLESView extends GLSurfaceView implements GLSurfaceView.Renderer, OnGestureListener, OnDoubleTapListener{

	private GestureDetector gestureDetector;
	private Context context;

	private int vertexShaderObject;
	private int fragmentShaderObject;
	private int shaderProgramObject;

	private int mvpUniform;

	private float perspectiveProjectionMatrix[] = new float[4 * 4];

	//For Axis
	private int vao_Axis[] = new int[1];
	private int vbo_Axis_Position[] = new int[1];
	private int vbo_Axis_Color[] = new int[1];

	//For Lines
	private int vao_Lines[] = new int[1];
	private int vbo_Lines_Position[] = new int[1];
	private int vbo_Lines_Color[] = new int[1];

	//For Triangle
	private int vao_Tri[] = new int[1];
	private int vbo_Tri_Position[] = new int[1];;
	private int vbo_Tri_Color[] = new int[1];

	//For Rect
	private int vao_Rect[] = new int[1];
	private int vbo_Rect_Position[] = new int[1];
	private int vbo_Rect_Color[] = new int[1];

	//For Circle
	private int vao_Circle[] = new int[1];
	private int vbo_Circle_Position[] = new int[1];
	private int vbo_Circle_Color[] = new int[1];
	private final int CIRCLE_POINTS = 6000;


	//For InCirlce
	private float Incircle_Radius;
	private float Incircle_Center[] = new float[3];



	GLESView(Context drawingContext){
		super(drawingContext);
		context = drawingContext;
		
		gestureDetector = new GestureDetector(context, this, null, false);
		gestureDetector.setOnDoubleTapListener(this);

		setEGLContextClientVersion(3);
		setRenderer(this);
		setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);
	}


	@Override
	public boolean onTouchEvent(MotionEvent e){
		int eventaction = e.getAction();
		if(!gestureDetector.onTouchEvent(e))
			super.onTouchEvent(e);
		return(true);
	}


	/********** Methods from OnDoubleTapListener Interface **********/
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


	/********** Methods From OnGestureListener **********/
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


	/********** Methods From GLSurfaceView.Render Interface **********/
	@Override
	public void onSurfaceCreated(GL10 gl, EGLConfig config){
		String version = gl.glGetString(GL10.GL_VERSION);
		String glsl_version = gl.glGetString(GLES32.GL_SHADING_LANGUAGE_VERSION);
		String vendor = gl.glGetString(GL10.GL_VENDOR);
		String renderer = gl.glGetString(GL10.GL_RENDERER);

		System.out.println("RTR: OpenGL Version: " + version);
		System.out.println("RTR: OpenGLSL Verions: " + glsl_version);
		System.out.println("RTR: Vendor: " + vendor);
		System.out.println("RTR: Renderer: " + renderer);

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



	/*********** Now OpenGL **********/


	private void initialize(){

		/*********** Vertex Shader **********/
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
				szInfoLog = null;
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

		iShaderCompileStatus[0] = 0;
		iInfoLogLength[0] = 0;
		szInfoLog = null;

		GLES32.glGetShaderiv(fragmentShaderObject, GLES32.GL_COMPILE_STATUS, iShaderCompileStatus, 0);
		if(iShaderCompileStatus[0] == GLES32.GL_FALSE){
			GLES32.glGetShaderiv(fragmentShaderObject, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if(iInfoLogLength[0] > 0){
				szInfoLog = GLES32.glGetShaderInfoLog(fragmentShaderObject);
				System.out.println("RTR: Fragment Shader Compilation Error: " + szInfoLog);
				szInfoLog = null;
				uninitialize();
				System.exit(0);
			}
		}



		/********** Program Object **********/
		shaderProgramObject = GLES32.glCreateProgram();

		GLES32.glAttachShader(shaderProgramObject, vertexShaderObject);
		GLES32.glAttachShader(shaderProgramObject, fragmentShaderObject);

		GLES32.glBindAttribLocation(shaderProgramObject, GLESMacros.AMC_ATTRIBUTE_POSITION, "vPosition");
		GLES32.glBindAttribLocation(shaderProgramObject, GLESMacros.AMC_ATTRIBUTE_COLOR, "vColor");

		GLES32.glLinkProgram(shaderProgramObject);

		int iProgramLinkStatus[] = new int[1];
		iInfoLogLength[0] = 0;
		szInfoLog = null;		

		GLES32.glGetProgramiv(shaderProgramObject, GLES32.GL_LINK_STATUS, iProgramLinkStatus, 0);
		if(iProgramLinkStatus[0] == GLES32.GL_FALSE){
			GLES32.glGetProgramiv(shaderProgramObject, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if(iInfoLogLength[0] > 0){
				szInfoLog = GLES32.glGetProgramInfoLog(shaderProgramObject);
				System.out.println("RTR: Shader Program Linking Error: " + szInfoLog);
				szInfoLog = null;
				uninitialize();
				System.exit(0);
			}
		}


		mvpUniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_mvp_matrix");



		/*********** Shape and its Colors **********/
		float line_position[] = new float[40 * 3 * 2]; 	//Total 40 Lines with each of 3-3 coordinate

		float tri_position[] = new float[] {
			0.0f, 0.7f, 0.0f,
			-0.7f, -0.7f, 0.0f,
			0.7f, -0.7f, 0.0f
		};

		float tri_color[] = new float[] {
			1.0f, 1.0f, 0.0f,
			1.0f, 1.0f, 0.0f,
			1.0f, 1.0f, 0.0f
		};

		float rect_position[] = new float[]{
			0.7f, 0.7f, 0.0f,
			-0.7f, 0.7f, 0.0f,
			-0.7f, -0.7f, 0.0f,
			0.7f, -0.7f, 0.0f
		};

		float rect_color[] = new float[]{
			1.0f, 1.0f, 0.0f,
			1.0f, 1.0f, 0.0f,
			1.0f, 1.0f, 0.0f,
			1.0f, 1.0f, 0.0f
		};


		//For Incenter Calculations
		Calculation(tri_position);



		/********** Axis **********/
		GLES32.glGenVertexArrays(1, vao_Axis, 0);
		GLES32.glBindVertexArray(vao_Axis[0]);

			/********** Position **********/
			GLES32.glGenBuffers(1, vbo_Axis_Position, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Axis_Position[0]);
			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							3 * 2 * 4,
							null,
							GLES32.GL_DYNAMIC_DRAW);
			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_POSITION,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

			/********** Color **********/
			GLES32.glGenBuffers(1, vbo_Axis_Color, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Axis_Color[0]);
			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER, 
								3 * 2 * 4,
								null,
								GLES32.GL_DYNAMIC_DRAW);
			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_COLOR,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_COLOR);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		GLES32.glBindVertexArray(0);





		Fill_LinesData(line_position, 10);

		/********** Lines **********/
		GLES32.glGenVertexArrays(1, vao_Lines, 0);
		GLES32.glBindVertexArray(vao_Lines[0]);

			/********** Position **********/
			GLES32.glGenBuffers(1, vbo_Lines_Position, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Lines_Position[0]);

				/********** For glBufferData **********/
				ByteBuffer linePosition_ByteBuffer = ByteBuffer.allocateDirect(line_position.length * 4);
				linePosition_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer linePosition_FloatBuffer = linePosition_ByteBuffer.asFloatBuffer();
				linePosition_FloatBuffer.put(line_position);
				linePosition_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							line_position.length * 4,
							linePosition_FloatBuffer,
							GLES32.GL_STATIC_DRAW);
			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_POSITION,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

			/********** Color **********/
			GLES32.glVertexAttrib3f(GLESMacros.AMC_ATTRIBUTE_COLOR, 0.0f, 0.0f, 1.0f);

		GLES32.glBindVertexArray(0);





		/********** Triangle **********/
		GLES32.glGenVertexArrays(1, vao_Tri, 0);
		GLES32.glBindVertexArray(vao_Tri[0]);

			/********** Position ***********/
			GLES32.glGenBuffers(1, vbo_Tri_Position, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Tri_Position[0]);

				/********** For glBufferData **********/
				ByteBuffer triPosition_ByteBuffer = ByteBuffer.allocateDirect(tri_position.length * 4);
				triPosition_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer triPosition_FloatBuffer = triPosition_ByteBuffer.asFloatBuffer();
				triPosition_FloatBuffer.put(tri_position);
				triPosition_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							tri_position.length * 4,
							triPosition_FloatBuffer,
							GLES32.GL_STATIC_DRAW);
			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_POSITION,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);


			/********** Color **********/
			GLES32.glGenBuffers(1, vbo_Tri_Color, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Tri_Color[0]);

				/********** For glBufferData **********/
				ByteBuffer triColor_ByteBuffer = ByteBuffer.allocateDirect(tri_color.length * 4);
				triColor_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer triColor_FloatBuffer = triColor_ByteBuffer.asFloatBuffer();
				triColor_FloatBuffer.put(tri_color);
				triColor_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							tri_color.length * 4,
							triColor_FloatBuffer,
							GLES32.GL_STATIC_DRAW);
			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_COLOR,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_COLOR);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		GLES32.glBindVertexArray(0);




		/********** Rectangle ************/
		GLES32.glGenVertexArrays(1, vao_Rect, 0);
		GLES32.glBindVertexArray(vao_Rect[0]);


			/********** Position ***********/
			GLES32.glGenBuffers(1, vbo_Rect_Position, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Rect_Position[0]);

				/********** For glBufferData **********/
				ByteBuffer rectPosition_ByteBuffer = ByteBuffer.allocateDirect(rect_position.length * 4);
				rectPosition_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer rectPosition_FloatBuffer = rectPosition_ByteBuffer.asFloatBuffer();
				rectPosition_FloatBuffer.put(rect_position);
				rectPosition_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							rect_position.length * 4,
							rectPosition_FloatBuffer,
							GLES32.GL_STATIC_DRAW);
			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_POSITION,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);



			/********** Color **********/
			GLES32.glGenBuffers(1, vbo_Rect_Color, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Rect_Color[0]);

				/********** For glBufferData **********/
				ByteBuffer rectColor_ByteBuffer = ByteBuffer.allocateDirect(rect_color.length * 4);
				rectColor_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer rectColor_FloatBuffer = rectColor_ByteBuffer.asFloatBuffer();
				rectColor_FloatBuffer.put(rect_color);
				rectColor_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							rect_color.length * 4,
							rectColor_FloatBuffer,
							GLES32.GL_STATIC_DRAW);
			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_COLOR,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_COLOR);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		GLES32.glBindVertexArray(0);	




		/********** Circle **********/
		GLES32.glGenVertexArrays(1, vao_Circle, 0);
		GLES32.glBindVertexArray(vao_Circle[0]);

			/********** Position **********/
			GLES32.glGenBuffers(1, vbo_Circle_Position, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Circle_Position[0]);
			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							3 * CIRCLE_POINTS * 4,
							null,
							GLES32.GL_DYNAMIC_DRAW);
			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_POSITION,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

			/********** Color **********/
			GLES32.glGenBuffers(1, vbo_Circle_Color, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER,  vbo_Circle_Color[0]);
			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							3 * CIRCLE_POINTS * 4,
							null,
							GLES32.GL_DYNAMIC_DRAW);
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

		Matrix.setIdentityM(perspectiveProjectionMatrix, 0);
	}

	private void uninitialize(){

		//Circle
		if(vbo_Circle_Color[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Circle_Color, 0);
			vbo_Circle_Color[0] = 0;
		}

		if(vbo_Circle_Position[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Circle_Position, 0);
			vbo_Circle_Position[0] = 0;
		}

		if(vao_Circle[0] != 0){
			GLES32.glDeleteVertexArrays(1, vao_Circle, 0);
			vao_Circle[0] = 0;
		}

		//Rect
		if(vbo_Rect_Color[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Rect_Color, 0);
			vbo_Rect_Color[0] = 0;
		}

		if(vbo_Rect_Position[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Rect_Position, 0);
			vbo_Rect_Position[0] = 0;
		}

		if(vao_Rect[0] != 0){
			GLES32.glDeleteVertexArrays(1, vao_Rect, 0);
			vao_Rect[0] = 0;
		}


		//Tri
		if(vbo_Tri_Color[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Tri_Color, 0);
			vbo_Tri_Color[0] = 0;
		}

		if(vbo_Tri_Position[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Tri_Position, 0);
			vbo_Tri_Position[0] = 0;
		}

		if(vao_Tri[0] != 0){
			GLES32.glDeleteVertexArrays(1, vao_Tri, 0);
			vao_Tri[0] = 0;
		}


		//Lines
		if(vbo_Lines_Position[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Lines_Position, 0);
			vbo_Lines_Position[0] = 0;
		}

		if(vao_Lines[0] != 0){
			GLES32.glDeleteVertexArrays(1, vao_Lines, 0);
			vao_Lines[0] = 0;
		}

		if(vbo_Axis_Color[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Axis_Color, 0);
			vbo_Axis_Color[0] = 0;
		}


		//Axis
		if(vbo_Axis_Position[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Axis_Position, 0);
			vbo_Axis_Position[0] = 0;
		}

		if(vao_Axis[0] != 0){
			GLES32.glDeleteVertexArrays(1, vao_Axis, 0);
			vao_Axis[0] = 0;
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



	private float axis_position[] = new float[3 * 2];
	private float axis_color[] = new float[3 * 2];

	private float circle_position[] = new float[CIRCLE_POINTS * 3];
	private float circle_color[] = new float[CIRCLE_POINTS * 3];


	private void display(){

		float translateMatrix[] = new float[4 * 4];
		float modelViewMatrix[] = new float[4 * 4];
		float modelViewProjectionMatrix[] = new float[4 * 4];

		

		GLES32.glClear(GLES32.GL_COLOR_BUFFER_BIT | GLES32.GL_DEPTH_BUFFER_BIT);

		GLES32.glUseProgram(shaderProgramObject);


		for(int i = 1; i <= 2; i++){
		
			if(i == 1){
				Fill_AxisData(axis_position, axis_color, 'R');
			}
			else if(i == 2)
				Fill_AxisData(axis_position, axis_color, 'B');




			/********** Axis **********/
			Matrix.setIdentityM(translateMatrix, 0);
			Matrix.setIdentityM(modelViewMatrix, 0);
			Matrix.setIdentityM(modelViewProjectionMatrix, 0);

			Matrix.translateM(translateMatrix, 0, 0.0f, 0.0f, -3.0f);
			Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, translateMatrix, 0);
			Matrix.multiplyMM(modelViewProjectionMatrix, 0, perspectiveProjectionMatrix, 0, modelViewMatrix, 0);

			GLES32.glUniformMatrix4fv(mvpUniform, 1, false, modelViewProjectionMatrix, 0);

			GLES32.glBindVertexArray(vao_Axis[0]);

				/********** Position **********/
				GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Axis_Position[0]);

					/********** For glBufferData **********/
					ByteBuffer axisPosition_ByteBuffer = ByteBuffer.allocateDirect(axis_position.length * 4);
					axisPosition_ByteBuffer.order(ByteOrder.nativeOrder());
					FloatBuffer axisPosition_FloatBuffer = axisPosition_ByteBuffer.asFloatBuffer();
					axisPosition_FloatBuffer.put(axis_position);
					axisPosition_FloatBuffer.position(0);

					GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER, 
									axis_position.length * 4,
									axisPosition_FloatBuffer,
									GLES32.GL_DYNAMIC_DRAW);

				GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);


				/*********** Color **********/
				GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Axis_Color[0]);

					/********** For glBufferData **********/
					ByteBuffer axisColor_ByteBuffer = ByteBuffer.allocateDirect(axis_color.length * 4);
					axisColor_ByteBuffer.order(ByteOrder.nativeOrder());
					FloatBuffer axisColor_FloatBuffer = axisColor_ByteBuffer.asFloatBuffer();
					axisColor_FloatBuffer.put(axis_color);
					axisColor_FloatBuffer.position(0);

					GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER, 
									axis_color.length * 4,
									axisColor_FloatBuffer,
									GLES32.GL_DYNAMIC_DRAW);
				GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);


				/*********** Draw **********/
				GLES32.glDrawArrays(GLES32.GL_LINES, 0, 2);

			GLES32.glBindVertexArray(0);
		}





		/************ Lines ************/
		Matrix.setIdentityM(translateMatrix, 0);
		Matrix.setIdentityM(modelViewMatrix, 0);
		Matrix.setIdentityM(modelViewProjectionMatrix, 0);

		Matrix.translateM(translateMatrix, 0, 0.0f, 0.0f, -3.0f);
		Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, translateMatrix, 0);
		Matrix.multiplyMM(modelViewProjectionMatrix, 0, perspectiveProjectionMatrix, 0, modelViewMatrix, 0);

		GLES32.glUniformMatrix4fv(mvpUniform, 1, false, modelViewProjectionMatrix, 0);

		GLES32.glBindVertexArray(vao_Lines[0]);
			GLES32.glDrawArrays(GLES32.GL_LINES, 0, 40 * 2);
		GLES32.glBindVertexArray(0);





		/********** Triangle **********/
		Matrix.setIdentityM(translateMatrix, 0);
		Matrix.setIdentityM(modelViewMatrix, 0);
		Matrix.setIdentityM(modelViewProjectionMatrix, 0);

		Matrix.translateM(translateMatrix, 0, 0.0f, 0.0f, -3.0f);
		Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, translateMatrix, 0);
		Matrix.multiplyMM(modelViewProjectionMatrix, 0, perspectiveProjectionMatrix, 0, modelViewMatrix, 0);

		GLES32.glUniformMatrix4fv(mvpUniform, 1, false, modelViewProjectionMatrix, 0);

		GLES32.glBindVertexArray(vao_Tri[0]);
			GLES32.glDrawArrays(GLES32.GL_LINE_LOOP, 0, 3);
		GLES32.glBindVertexArray(0);

		

		/********** Rectangle **********/
		Matrix.setIdentityM(translateMatrix, 0);
		Matrix.setIdentityM(modelViewMatrix, 0);
		Matrix.setIdentityM(modelViewProjectionMatrix, 0);

		Matrix.translateM(translateMatrix, 0, 0.0f, 0.0f, -3.0f);
		Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, translateMatrix, 0);
		Matrix.multiplyMM(modelViewProjectionMatrix, 0, perspectiveProjectionMatrix, 0, modelViewMatrix, 0);

		GLES32.glUniformMatrix4fv(mvpUniform, 1, false, modelViewProjectionMatrix, 0);

		GLES32.glBindVertexArray(vao_Rect[0]);
			GLES32.glDrawArrays(GLES32.GL_LINE_LOOP, 0, 4);
		GLES32.glBindVertexArray(0);





		/********** Circles **********/
		for(int  i = 1; i <= 2; i++){

			if(i == 1)
				Fill_CircleData(circle_position, circle_color, 1);
			else 
				Fill_CircleData(circle_position, circle_color, 2);

			
			Matrix.setIdentityM(translateMatrix, 0);
			Matrix.setIdentityM(modelViewMatrix, 0);
			Matrix.setIdentityM(modelViewProjectionMatrix, 0);

			Matrix.translateM(translateMatrix, 0, 0.0f, 0.0f, -3.0f);
			Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, translateMatrix, 0);
			Matrix.multiplyMM(modelViewProjectionMatrix, 0, perspectiveProjectionMatrix, 0, modelViewMatrix, 0);

			GLES32.glUniformMatrix4fv(mvpUniform, 1, false, modelViewProjectionMatrix, 0);


			GLES32.glBindVertexArray(vao_Circle[0]);

				/********** Position **********/
				GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Circle_Position[0]);

					ByteBuffer circlePosition_ByteBuffer = ByteBuffer.allocateDirect(circle_position.length * 4);
					circlePosition_ByteBuffer.order(ByteOrder.nativeOrder());
					FloatBuffer circlePosition_FloatBuffer = circlePosition_ByteBuffer.asFloatBuffer();
					circlePosition_FloatBuffer.put(circle_position);
					circlePosition_FloatBuffer.position(0);

					GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
									circle_position.length * 4,
									circlePosition_FloatBuffer,
									GLES32.GL_DYNAMIC_DRAW);

				GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);



				/********** Color **********/
				GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Circle_Color[0]);

					ByteBuffer circleColor_ByteBuffer = ByteBuffer.allocateDirect(circle_color.length * 4);
					circleColor_ByteBuffer.order(ByteOrder.nativeOrder());
					FloatBuffer circleColor_FloatBuffer = circleColor_ByteBuffer.asFloatBuffer();
					circleColor_FloatBuffer.put(circle_color);
					circleColor_FloatBuffer.position(0);

					GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
									circle_color.length * 4,
									circleColor_FloatBuffer,
									GLES32.GL_DYNAMIC_DRAW);


				GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);


				/********** Draw ***********/
				GLES32.glDrawArrays(GLES32.GL_POINTS, 0, CIRCLE_POINTS);
			GLES32.glBindVertexArray(0);

		}






		GLES32.glUseProgram(0);

		requestRender();
	}


	private void Fill_AxisData(float pos[], float col[], char c){
		switch(c){
			case 'R':
				pos[0] = 0.0f;
				pos[1] = 1.0f;
				pos[2] = 0.0f;

				pos[3] = 0.0f;
				pos[4] = -1.0f;
				pos[5] = 0.0f;

				col[0] = 1.0f;	//R
				col[1] = 0.0f;	//G
				col[2] = 0.0f;	//B
				
				col[3] = 1.0f;
				col[4] = 0.0f;
				col[5] = 0.0f;
			break;

			case 'B':
				pos[0] = -1.0f;
				pos[1] = 0.0f;
				pos[2] = 0.0f;

				pos[3] = 1.0f;
				pos[4] = 0.0f;
				pos[5] = 0.0f;

				col[0] = 0.0f;	//R
				col[1] = 1.0f;	//G
				col[2] = 0.0f;	//B
				
				col[3] = 0.0f;
				col[4] = 1.0f;
				col[5] = 0.0f;
				
			break;
		}
	}

	private void Fill_LinesData(float pos[], int iLines){

		float x = 1.0f / iLines;
		float val = x;

		for(int i = 0; i < iLines * 4 * 3 * 2; i = i + 24){

			//Vertical Right
			pos[i] = val;
			pos[i + 1] = 1.0f;
			pos[i + 2] = 0.0f;

			pos[i + 3] = val;
			pos[i + 4] = -1.0f;
			pos[i + 5] = 0.0f;

			//Vertical Left;
			pos[i + 6] = -val;
			pos[i + 7] = 1.0f;
			pos[i + 8] = 0.0f;

			pos[i + 9] = -val;
			pos[i + 10] = -1.0f;
			pos[i + 11] = 0.0f;


			//Horizontal Top
			pos[i + 12] = -1.0f;
			pos[i + 13] = val;
			pos[i + 14] = 0.0f;

			pos[i + 15] = 1.0f;
			pos[i + 16] = val;
			pos[i + 17] = 0.0f;

			//Horizontal Bottom;
			pos[i + 18] = -1.0f;
			pos[i + 19] = -val;
			pos[i + 20] = 0.0f;

			pos[i + 21] = 1.0f;
			pos[i + 22] = -val;
			pos[i + 23] = 0.0f;


			val = val + x;
			System.out.println("RTR: Value: " + i);
		}
	}


	private void Calculation(float arr[]){
		float a, b, c;
		float s;

		//Distance Formula
		a = (float)Math.sqrt(Math.pow((arr[6] - arr[3]), 2) + Math.pow((arr[7] - arr[4]), 2));
		b = (float)Math.sqrt(Math.pow((arr[6] - arr[0]), 2) + Math.pow((arr[7] - arr[1]), 2));
		c = (float)Math.sqrt(Math.pow((arr[3] - arr[0]), 2) + Math.pow((arr[4] - arr[1]), 2));

		s = (a + b + c) / 2;

		Incircle_Radius = (float)(Math.sqrt(s * (s - a) * (s - b) * (s - c)) / s);

		Incircle_Center[0] = (a * arr[0] + b * arr[3] + c * arr[6]) / (a+ b+ c);
		Incircle_Center[1] = (a * arr[1] + b * arr[4] + c * arr[7]) / (a+ b+ c);
		Incircle_Center[2] = 0.0f;

	}


	private void Fill_CircleData(float arr[], float arrColor[], int iFlag){
	
		//memset(arr, 0, sizeof(float) * CIRCLE_POINTS * 3);

		if(iFlag == 1){
			//InCircle
			for(int i = 0; i < CIRCLE_POINTS; i = i + 3){
				float x = (float)(2.0f * Math.PI * i / CIRCLE_POINTS);
				arr[i] = (float)(Incircle_Radius * Math.cos(x)) + Incircle_Center[0];
				arr[i + 1] = (float)(Incircle_Radius * Math.sin(x)) + Incircle_Center[1];
				arr[i + 2] = 0.0f;


				arrColor[i] = 1.0f;		//R
				arrColor[i + 1] = 1.0f;	//G
				arrColor[i + 2] = 0.0f;	//B
			}
			

		}
		else{
			//Outer Circle
			for(int i = 0; i < CIRCLE_POINTS; i = i + 3){
				float x = (float)(2.0f * Math.PI * i / CIRCLE_POINTS);
				arr[i] = (float)(1.0f * Math.cos(x));
				arr[i + 1] = (float)(1.0f * Math.sin(x));
				arr[i + 2] = 0.0f;

				arrColor[i] = 1.0f;		//R
				arrColor[i + 1] = 1.0f;	//G
				arrColor[i + 2] = 0.0f; 	//B

			}
		}

	}

}

