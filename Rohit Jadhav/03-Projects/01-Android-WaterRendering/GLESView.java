package com.rohit_r_jadhav.hello_world;

//For OpenGL
import android.opengl.GLSurfaceView;
import android.opengl.GLES32;

import javax.microedition.khronos.opengles.GL10;
import javax.microedition.khronos.egl.EGLConfig;

//Like Vmath
import android.opengl.Matrix;


//For glBufferData() and FloatBuffer
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.IntBuffer;
import java.lang.Math;
import java.util.Random;

import android.media.MediaPlayer;




//For Event 
import android.view.MotionEvent;
import android.view.GestureDetector;
import android.view.GestureDetector.OnGestureListener;
import android.view.GestureDetector.OnDoubleTapListener;
import android.content.Context;

public class GLESView extends GLSurfaceView implements GLSurfaceView.Renderer, OnGestureListener, OnDoubleTapListener{

	private GestureDetector gestureDetector;
	private Context context;
	boolean bLights = true;
	final int GRID_TRIANGLES = 1;
	final int GRID_LINES = 2;
	int iGridType = GRID_TRIANGLES;
	MediaPlayer mediaPlayer;

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

	/********** Methods from OnDoubleTapListener Interface **********/
	@Override
	public boolean onDoubleTap(MotionEvent e){
		if(iGridType == GRID_TRIANGLES)
			iGridType = GRID_LINES;
		else
			iGridType = GRID_TRIANGLES; 

		/*MediaPlayer mediaPlayer = MediaPlayer.create(context, R.raw.sound2);
		mediaPlayer.start();*/
		return(true);
	}

	@Override
	public boolean onDoubleTapEvent(MotionEvent e){
		return(true);
	}

	@Override
	public boolean onSingleTapConfirmed(MotionEvent e){
		if(bLights == false)
			bLights = true;
		else
			bLights = false;
		return(true);
	}

	/********** Methods from OnGestureListener Interface **********/
	@Override
	public boolean onDown(MotionEvent e){
		return(true);
	}

	@Override
	public boolean onSingleTapUp(MotionEvent e){
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
		System.out.println("RTR: Vendor: "+ vendor);
		System.out.println("RTR: Renderer: "+ renderer);

		mediaPlayer = MediaPlayer.create(context, R.raw.final_sound2);
		mediaPlayer.start();
		initialize();
	}

	@Override 
	public void onSurfaceChanged(GL10 unused, int width, int height){
		resize(width, height);
	}

	@Override
	public void onDrawFrame(GL10 unused){
		update();
		display();
	}


	//Grid
	final int GRID_WIDTH = 128;
	final int GRID_HEIGHT = 128;
	final float GRID_AMPLITUDE = 4.0f;


	//For Shader
	private int vertexShaderObject;
	private int fragmentShaderObject;
	private int shaderProgramObject;

	//For Matrix
	private int[] vao_Grid = new int[1];
	private int[] vbo_Grid_Position = new int[1];
	private int[] vbo_Grid_Angle = new int[1];
	private int[] vbo_Grid_Normal = new int[1];
	private int[] vbo_Grid_Element = new int[1];
	private int[] vbo_Grid_CommonPoints = new int[1];

	/********** Position and Color **********/
	float grid_Position[] = new float[3 * GRID_WIDTH * GRID_HEIGHT];
	float grid_Color[] = new float[3 * GRID_WIDTH * GRID_HEIGHT];
	float grid_Angle[] = new float[GRID_WIDTH * GRID_HEIGHT];
	float grid_Normal[] = new float[3 * GRID_WIDTH * GRID_HEIGHT];
	float grid_Texcoord[] = new float[2 * GRID_WIDTH * GRID_HEIGHT];
	int  grid_Index[] = new int[6 * (GRID_WIDTH - 1) * (GRID_HEIGHT - 1)];
	float grid_CommonPoints[] = new float[GRID_WIDTH * GRID_HEIGHT];

	//For Projection
	private float perspectiveProjectionMatrix[] = new float[4 * 4];

	//For Uniform
	private int modelMatrixUniform;
	private int viewMatrixUniform;
	private int projectionMatrixUniform;
	private int speedUniform;

	//Per Fragment
	private int light1_la_Uniform;
	private int light1_ld_Uniform;
	private int light1_ls_Uniform;
	private int light1_PositionUniform;

	private int light2_la_Uniform;
	private int light2_ld_Uniform;
	private int light2_ls_Uniform;
	private int light2_PositionUniform;

	private int light3_la_Uniform;
	private int light3_ld_Uniform;
	private int light3_ls_Uniform;
	private int light3_PositionUniform;

	private int light4_la_Uniform;
	private int light4_ld_Uniform;
	private int light4_ls_Uniform;
	private int light4_PositionUniform;

	
	private int ka_Uniform;
	private int kd_Uniform;
	private int ks_Uniform;
	private int shininessUniform;
	private int LKeyPressUniform;


	//For Texture
	private int texture_ocean[] = new int[1];

	//For Lights
	float light1_Ambient[] = new float[]{0.0f, 0.0f, 0.0f, 0.0f}; 		//241, 218, 164
	float light1_Diffuse[] = new float[] {0.9450f, 0.85490f, 0.64313f, 1.0f};
	float light1_Specular[] = new float[]{0.9450f, 0.85490f, 0.64313f, 1.0f};
	float light1_Position[] = new float[]{0.0f, -5.0f, 128.0f, 1.0f};	//{0.0f, 64.0f, 128.0f, 1.0f};

	

	float light2_Ambient[] = new float[]{0.0f, 0.0f, 0.0f, 0.0f};
	float light2_Diffuse[] = new float[] {0.9450f, 0.85490f, 0.64313f, 1.0f};
	float light2_Specular[] = new float[]{0.9450f, 0.85490f, 0.64313f, 1.0f};
	float light2_Position[] = new float[]{-8.0f, -40.0f, -64.0f, 1.0f};	//{-32.0f, 64.0f, -64.0f, 1.0f};

	float light3_Ambient[] = new float[]{0.0f, 0.0f, 0.0f, 0.0f};
	float light3_Diffuse[] = new float[] {0.9450f, 0.85490f, 0.64313f, 1.0f};
	float light3_Specular[] = new float[]{0.9450f, 0.85490f, 0.64313f, 1.0f};
	float light3_Position[] = new float[]{8.0f, -40.0f, -64.0f, 1.0f};	//{32.0f, 64.0f, -64.0f, 1.0f};


	

	float light4_Ambient[] = new float[]{0.0f, 0.0f, 0.0f, 0.0f};
	float light4_Diffuse[] = new float[] {0.9450f, 0.85490f, 0.64313f, 1.0f};
	float light4_Specular[] = new float[]{0.9450f, 0.85490f, 0.64313f, 1.0f};
	float light4_Position[] = new float[]{0.0f, -5.0f, 128.0f, 1.0f};	//{0.0f, 64.0f, 128.0f, 1.0f};






	//For Material
	float materialAmbient[] = new float[]{0.0f, 0.00f, 0.0f, 1.0f};
	float materialDiffuse[] = new float[] {0.0f, 0.0f, 0.150f, 1.0f};		//0.0f, 0.1f, 0.4f, 1.0f
	float materialSpecular[] = new float[]{0.9450f, 0.85490f, 0.64313f, 1.0f};	//79, 105, 136 // moonlight blues
	float materialShininess = 0.25f * 128.0f;


	//For Sequence
	int iSequence = 1;
	float sceneCounter = 75.0f;


	private void initialize(){

		/********** Vertex Shader **********/
		vertexShaderObject = GLES32.glCreateShader(GLES32.GL_VERTEX_SHADER);

		final String vertexShaderSourceCode = String.format(
				"#version 320 es" +
				"\n" +
				"in vec4 vPosition;" +
				"in vec3 vNormal;" +
				"in float vAngle;" +
				"in float vCommonPoints;" +

				"uniform float u_y_speed;" +
				"uniform mat4 u_projection_matrix;" +
				"uniform mat4 u_view_matrix;" +
				"uniform mat4 u_model_matrix;" +
				"uniform float u_speed;" +
				
				"uniform vec4 u_light1_position;" +
				"uniform vec4 u_light2_position;" +
				"uniform vec4 u_light3_position;" +
				"uniform vec4 u_light4_position;" +

				"out vec3 light1_Direction_VS;" +
				"out vec3 light2_Direction_VS;" +
				"out vec3 light3_Direction_VS;" +
				"out vec3 light4_Direction_VS;" +

				"out vec3 tNormal_VS;" +
				"out vec3 viewer_VS;" +


				"void main(void)" +
				"{" + 
					"vec4 vert = vPosition;" +
					"vert.y = vert.y+ cos(vAngle + u_speed) * 4.0;" +

					"vec4 eyeCoordinate = u_view_matrix * u_model_matrix * vPosition;" +
					"light1_Direction_VS = vec3(u_light1_position - eyeCoordinate);" +
					"light2_Direction_VS = vec3(u_light2_position - eyeCoordinate);" +
					"light3_Direction_VS = vec3(u_light3_position - eyeCoordinate);" +
					"light4_Direction_VS = vec3(u_light4_position - eyeCoordinate);" +
					
						

					"mat3 normalMatrix = mat3(u_view_matrix * u_model_matrix);" +
					"vec3 avgNormal = vNormal;" +
					"avgNormal.x = avgNormal.x / vCommonPoints;" +
					"avgNormal.y = avgNormal.y / vCommonPoints;" +
					"avgNormal.z = avgNormal.z / vCommonPoints;" +					


					"tNormal_VS = vec3(normalMatrix * avgNormal);" +
					"viewer_VS = vec3(-eyeCoordinate.xyz);" +


					"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vert;" +
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


				"in vec3 light1_Direction_VS;" +
				"in vec3 light2_Direction_VS;" +
				"in vec3 light3_Direction_VS;" +
				"in vec3 light4_Direction_VS;" +

				"in vec3 tNormal_VS;" +
				"in vec3 viewer_VS;" +
				
				"uniform vec3 u_la_light1;" +
				"uniform vec3 u_ld_light1;" +
				"uniform vec3 u_ls_light1;" +

				"uniform vec3 u_la_light2;" +
				"uniform vec3 u_ld_light2;" +
				"uniform vec3 u_ls_light2;" +

				"uniform vec3 u_la_light3;" +
				"uniform vec3 u_ld_light3;" +
				"uniform vec3 u_ls_light3;" +

				"uniform vec3 u_la_light4;" +
				"uniform vec3 u_ld_light4;" +
				"uniform vec3 u_ls_light4;" +


				"uniform vec3 u_ka;" +
				"uniform vec3 u_kd;" +
				"uniform vec3 u_ks;" +
				"uniform float u_shininess;" +
				"uniform int u_LKeyPress;" +
				"uniform sampler2D u_sampler;" +

				"out vec4 FragColor;" +
				"void main(void)" +
				"{" +			

					"vec3 phongLight;" +

					"if(u_LKeyPress == 1)" +
					"{" +
						"vec3 normalizeLight1_Direction = normalize(light1_Direction_VS);" +
						"vec3 normalizeTNormal = normalize(tNormal_VS);" +						
						"float S1_Dot_N = max(dot(normalizeLight1_Direction, normalizeTNormal), 0.0);" +

						"vec3 normalizeLight2_Direction = normalize(light2_Direction_VS);" +
						"float S2_Dot_N = max(dot(normalizeLight2_Direction, normalizeTNormal), 0.0);" +

						"vec3 normalizeLight3_Direction = normalize(light3_Direction_VS);" +
						"float S3_Dot_N = max(dot(normalizeLight3_Direction, normalizeTNormal), 0.0);" +

						"vec3 normalizeLight4_Direction = normalize(light4_Direction_VS);" +
						"float S4_Dot_N = max(dot(normalizeLight4_Direction, normalizeTNormal), 0.0);" +



						"vec3 normalizeViewer = normalize(viewer_VS);" +

						"vec3 light1_Reflection = reflect(-normalizeLight1_Direction, normalizeTNormal);" +
						"float R1_Dot_V = max(dot(light1_Reflection, normalizeViewer), 0.0);" +


						"vec3 light2_Reflection = reflect(-normalizeLight2_Direction, normalizeTNormal);" +
						"float R2_Dot_V = max(dot(light2_Reflection, normalizeViewer), 0.0);" +


						"vec3 light3_Reflection = reflect(-normalizeLight3_Direction, normalizeTNormal);" +
						"float R3_Dot_V = max(dot(light3_Reflection, normalizeViewer), 0.0);" +


						"vec3 light4_Reflection = reflect(-normalizeLight4_Direction, normalizeTNormal);" +
						"float R4_Dot_V = max(dot(light4_Reflection, normalizeViewer), 0.0);" +


						"vec3 light1_ambient = u_la_light1 * u_ka;" +
						"vec3 light1_diffuse = u_ld_light1 * u_kd * S1_Dot_N;"  +
						"vec3 light1_specular = u_ls_light1 * u_ks * pow(R1_Dot_V, u_shininess);" +
						"vec3 light1 = light1_ambient + light1_diffuse + light1_specular;"+


						"vec3 light2_ambient = u_la_light2 * u_ka;" +
						"vec3 light2_diffuse = u_ld_light2 * u_kd * S2_Dot_N;"  +
						"vec3 light2_specular = u_ls_light2 * u_ks * pow(R2_Dot_V, u_shininess);" +
						"vec3 light2 = light2_ambient + light2_diffuse + light2_specular;"+


						"vec3 light3_ambient = u_la_light3 * u_ka;" +
						"vec3 light3_diffuse = u_ld_light3 * u_kd * S3_Dot_N;"  +
						"vec3 light3_specular = u_ls_light3 * u_ks * pow(R3_Dot_V, u_shininess);" +
						"vec3 light3 = light3_ambient + light3_diffuse + light3_specular;"+


						"vec3 light4_ambient = u_la_light4 * u_ka;" +
						"vec3 light4_diffuse = u_ld_light4 * u_kd * S4_Dot_N;"  +
						"vec3 light4_specular = u_ls_light4 * u_ks * pow(R4_Dot_V, u_shininess);" +
						"vec3 light4 = light4_ambient + light4_diffuse + light4_specular;"+



						"phongLight = light1 + light2 + light3 + light4;" +
					"}" +
					"else" +
					"{" +
						"phongLight = vec3(1.0, 1.0, 1.0);" +
					"}" +

					"FragColor = vec4(phongLight, 1.0);" +
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
		GLES32.glBindAttribLocation(shaderProgramObject, GLESMacros.AMC_ATTRIBUTE_NORMAL, "vNormal");
		GLES32.glBindAttribLocation(shaderProgramObject, GLESMacros.AMC_ATTRIBUTE_ANGLE, "vAngle");
		GLES32.glBindAttribLocation(shaderProgramObject, GLESMacros.AMC_ATTRIBUTE_COMMON_POINTS, "vCommonPoints");
				

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

		modelMatrixUniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_model_matrix");
		viewMatrixUniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_view_matrix");
		projectionMatrixUniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_projection_matrix");

		speedUniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_speed");
		
		light1_la_Uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_la_light1");
		light1_ld_Uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_ld_light1");
		light1_ls_Uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_ls_light1");
		light1_PositionUniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_light1_position");


		light2_la_Uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_la_light2");
		light2_ld_Uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_ld_light2");
		light2_ls_Uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_ls_light2");
		light2_PositionUniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_light2_position");


		light3_la_Uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_la_light3");
		light3_ld_Uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_ld_light3");
		light3_ls_Uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_ls_light3");
		light3_PositionUniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_light3_position");



		light4_la_Uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_la_light4");
		light4_ld_Uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_ld_light4");
		light4_ls_Uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_ls_light4");
		light4_PositionUniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_light4_position");



		ka_Uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_ka");
		kd_Uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_kd");
		ks_Uniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_ks");
		shininessUniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_shininess");
		LKeyPressUniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_LKeyPress");
		





		GetGridData(grid_Position, grid_Color, grid_Index, grid_Normal, grid_Angle, grid_Texcoord, grid_CommonPoints);




		/********** Grid **********/
		GLES32.glGenVertexArrays(1, vao_Grid, 0);
		GLES32.glBindVertexArray(vao_Grid[0]);

			/********** Position **********/
			GLES32.glGenBuffers(1, vbo_Grid_Position, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Grid_Position[0]);


				/********** For glBufferData() ***********/
				ByteBuffer gridPosition_ByteBuffer = ByteBuffer.allocateDirect(grid_Position.length * 4);
				gridPosition_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer gridPosition_FloatBuffer = gridPosition_ByteBuffer.asFloatBuffer();
				gridPosition_FloatBuffer.put(grid_Position);
				gridPosition_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER, 
							grid_Position.length * 4,
							gridPosition_FloatBuffer,
							GLES32.GL_DYNAMIC_DRAW);

			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_POSITION,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);

			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);


		

			/********** Angle **********/
			GLES32.glGenBuffers(1, vbo_Grid_Angle, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Grid_Angle[0]);

				/********* For glBufferData **********/
				ByteBuffer gridAngle_ByteBuffer = ByteBuffer.allocateDirect(grid_Angle.length * 4);
				gridAngle_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer gridAngle_FloatBuffer = gridAngle_ByteBuffer.asFloatBuffer();
				gridAngle_FloatBuffer.put(grid_Angle);
				gridAngle_FloatBuffer.position(0);
				

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
								grid_Angle.length * 4,
								gridAngle_FloatBuffer,
								GLES32.GL_STATIC_DRAW);

			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_ANGLE,
									1,
									GLES32.GL_FLOAT,
									false,
									0, 0);

			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_ANGLE);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);



			/********** Normals **********/
			GLES32.glGenBuffers(1, vbo_Grid_Normal, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Grid_Normal[0]);

				ByteBuffer gridNormal_ByteBuffer = ByteBuffer.allocateDirect(grid_Normal.length * 4);
				gridNormal_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer gridNormal_FloatBuffer = gridNormal_ByteBuffer.asFloatBuffer();
				gridNormal_FloatBuffer.put(grid_Normal);
				gridNormal_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							grid_Normal.length * 4,
							gridNormal_FloatBuffer,
							GLES32.GL_DYNAMIC_DRAW);
			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_NORMAL,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_NORMAL);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);



			/********** CommonPoints **********/
			GLES32.glGenBuffers(1, vbo_Grid_CommonPoints, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Grid_CommonPoints[0]);


				/********** For glBufferData() ***********/
				ByteBuffer gridCommonPoints_ByteBuffer = ByteBuffer.allocateDirect(grid_CommonPoints.length * 4);
				gridCommonPoints_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer gridCommonPoints_FloatBuffer = gridCommonPoints_ByteBuffer.asFloatBuffer();
				gridCommonPoints_FloatBuffer.put(grid_CommonPoints);
				gridCommonPoints_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER, 
							grid_CommonPoints.length * 4,
							gridCommonPoints_FloatBuffer,
							GLES32.GL_STATIC_DRAW);

			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_COMMON_POINTS,
									1,
									GLES32.GL_FLOAT,
									false,
									0, 0);

			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_COMMON_POINTS);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);



			/********** Elements **********/
			GLES32.glGenBuffers(1, vbo_Grid_Element, 0);
			GLES32.glBindBuffer(GLES32.GL_ELEMENT_ARRAY_BUFFER, vbo_Grid_Element[0]);

				ByteBuffer gridElement_ByteBuffer = ByteBuffer.allocateDirect(grid_Index.length * 4);
				gridElement_ByteBuffer.order(ByteOrder.nativeOrder());
				IntBuffer gridElement_IntBuffer = gridElement_ByteBuffer.asIntBuffer();
				gridElement_IntBuffer.put(grid_Index);
				gridElement_IntBuffer.position(0);


			GLES32.glBufferData(GLES32.GL_ELEMENT_ARRAY_BUFFER,
							grid_Index.length * 4,
							gridElement_IntBuffer,
							GLES32.GL_STATIC_DRAW);

		GLES32.glBindVertexArray(0);


		GLES32.glEnable(GLES32.GL_DEPTH_TEST);
		GLES32.glDepthFunc(GLES32.GL_LEQUAL);


		GLES32.glClearColor(0.0f, 0.0f, 0.08f, 0.0f);
	}




	Random random = new Random();
	int seed;

	private void GetGridData(float pos[], float col[], int index[], float normal[], float angle[], float tex[], float commonPoints[]){

		float max_width = 256.0f;
		float max_height = 128.0f;
		int vertexPointer = 0;	
		float scaleX = max_width / GRID_WIDTH;
		float scaleZ = max_height / GRID_HEIGHT;
		float X = -(max_width / 2);
		float Z = -(max_height / 2);

		float fAngle = 0.0f;

		seed = random.nextInt(100000000);

		int iCommon = 1;

		for(int i = 0; i < GRID_HEIGHT; i++){
			X = -(max_width / 2);
			for(int j = 0; j < GRID_WIDTH ; j++){

				pos[(i * 3 * GRID_WIDTH) + (j * 3) + 0] = X;
				pos[(i * 3 * GRID_WIDTH) + (j * 3) + 1] = getInterpolatedNoise(j/ 6.0f, i /6.0f) * GRID_AMPLITUDE;
				pos[(i * 3 * GRID_WIDTH) + (j * 3) + 2] = Z;

				normal[(i * 3 * GRID_WIDTH) + (j * 3) + 0] = 0.0f;
				normal[(i * 3 * GRID_WIDTH) + (j * 3) + 1] = 0.0f;
				normal[(i * 3 * GRID_WIDTH) + (j * 3) + 2] = 0.0f;

				angle[(i * GRID_WIDTH) + j] = fAngle;

				col[(i * 3 * GRID_WIDTH) + (j * 3) + 0] = (float)(156.0f / 256.0f);
				col[(i * 3 * GRID_WIDTH) + (j * 3) + 1] = (float)(211.0f / 256.0f);
				col[(i * 3 * GRID_WIDTH) + (j * 3) + 2] = (float)(219.0f / 256.0f); 


				tex[(i * 2 * GRID_WIDTH) + (j * 2) + 0] = X;
				tex[(i * 2 * GRID_WIDTH) + (j * 2) + 1] = Z;



				//For Common Triangle Count
				if((i == 0 && j == 0) || (i == GRID_HEIGHT - 1 && j == GRID_WIDTH -1))
					iCommon = 1;
				else if((i == GRID_HEIGHT - 1 &&  j == 0) || (j == GRID_WIDTH - 1 && i == 0))
					iCommon = 2;
				else if(((i == 0 || i == GRID_HEIGHT - 1) && (j >  0 && j < GRID_WIDTH - 1)) ||
					 ((j == 0 || j == GRID_WIDTH - 1) && (i > 0 && i < GRID_HEIGHT - 1)))
					iCommon = 3;
				else 
					iCommon = 6;


				commonPoints[(i * GRID_WIDTH) + j] = iCommon;

				//vertexPointer++;
				X = X + scaleX;
				fAngle = fAngle + 0.05f;

			}
			Z = Z + scaleZ;
		}
		

		int iPosition = 0;

		for(int z = 0; z < GRID_HEIGHT - 1; z++){
			for(int x = 0; x < GRID_WIDTH - 1; x++){

				int topLeft = (z * GRID_WIDTH) + x;
				int topRight = topLeft + 1;
				int bottomLeft = ((z + 1) * GRID_WIDTH) + x;
				int bottomRight = bottomLeft + 1;

				index[iPosition] = topLeft;
				index[iPosition + 1] = bottomLeft;
				index[iPosition + 2] = topRight;

				index[iPosition + 3] = topRight;
				index[iPosition + 4] = bottomLeft;
				index[iPosition + 5] = bottomRight;

				iPosition = iPosition + 6;
			}
		}



	}

	private float findNormal(float p1[], float p2[], float p3[], int iWhichPoint, int iXYZ){

		float U[] = new float[3];
		float V[] = new float[3];

		switch(iWhichPoint){
			case 1:
				U[0] = p2[0] - p1[0];
				U[1] = p2[1] - p1[1];
				U[2] = p2[2] - p1[2];

				V[0] = p3[0] - p1[0];
				V[1] = p3[1] - p1[1];
				V[2] = p3[2] - p1[2];

				break;

			case 2:
				U[0] = p3[0] - p2[0];
				U[1] = p3[1] - p2[1];
				U[2] = p3[2] - p2[2];

				V[0] = p1[0] - p2[0];
				V[1] = p1[1] - p2[1];
				V[2] = p1[2] - p2[2];

				break;

			case 3:
				U[0] = p1[0] - p3[0];
				U[1] = p1[1] - p3[1];
				U[2] = p1[2] - p3[2];

				V[0] = p2[0] - p3[0];
				V[1] = p2[1] - p3[1];
				V[2] = p2[2] - p3[2];

			 	break;
		}


		//Now We find Crossproduct of Them for respective Coordinate;
		switch(iXYZ){

			//X
			case 0:
				return((U[1] * V[2]) - (U[2] * V[1]));

			//Y
			case 1:
				return((U[2] * V[0]) - (U[0] * V[2]));

			//Z
			case 2:
				return((U[0] * V[1]) - (U[1] * V[0]));
		}

		return(0.0f);
	}


	private float getNoise(int x, int z){
		random.setSeed(x * 49544 + z * 324566 + seed);
		return(random.nextFloat() * 2.0f - 1.0f);
	}

	private float getSmoothNoise(int x, int z){
		float corner = (getNoise(x - 1, z - 1) + getNoise(x + 1, z - 1) + getNoise(x - 1, z + 1) + getNoise(x + 1, z + 1)) / 16.0f;
		float side = (getNoise(x, z - 1) + getNoise(x, z + 1) + getNoise(x - 1, z) + getNoise(x + 1, z)) / 8.0f;
		float center = getNoise(x, z) / 4.0f;
		return(corner + side + center);
	}

	private float getInterpolatedNoise(float x, float z){
		int intX = (int)x;
		int intZ = (int)z;
		float fracX = x - intX;
		float fracZ = z - intZ;

		float v1 = getSmoothNoise(intX, intZ);
		float v2 = getSmoothNoise(intX + 1, intZ);
		float v3 = getSmoothNoise(intX, intZ + 1);
		float v4 = getSmoothNoise(intX + 1, intZ + 1);

		float in1 = interpolation(v1, v2, fracX);
		float in2 = interpolation(v3, v4, fracX);
		float in3 = interpolation(in1, in2, fracZ);

		return(in3);

	}


	private float interpolation(float a, float b, float blendFactor){
		float theta = blendFactor * (float)Math.PI;
		float f = (float)((1.0f - Math.cos(theta)) * 0.5f);
		return(a * (1.0f - f) + b * f);
	}



	private void uninitialize(){

		if(mediaPlayer != null){
			mediaPlayer.release();
			mediaPlayer = null;
		}
		
		if(vbo_Grid_Element[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Grid_Element, 0);
			vbo_Grid_Element[0] = 0;
		}

		if(vbo_Grid_Angle[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Grid_Angle, 0);
			vbo_Grid_Angle[0] = 0;
		}

		if(vbo_Grid_Normal[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Grid_Normal, 0);
			vbo_Grid_Normal[0] = 0;
		}

		if(vbo_Grid_Position[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Grid_Position, 0);
			vbo_Grid_Position[0] = 0;
		}

		if(vao_Grid[0] != 0){
			GLES32.glDeleteVertexArrays(1, vao_Grid, 0);
			vao_Grid[0] = 0;
		}

		if(shaderProgramObject != 0){

			int iShaderCount[] = new int[1];
			int iShaderNo;

			GLES32.glUseProgram(shaderProgramObject);

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

		Matrix.setIdentityM(perspectiveProjectionMatrix, 0);

		Matrix.perspectiveM(perspectiveProjectionMatrix, 0,
						45.0f,
						(float)width / (float)height,
						0.1f,
						128.0f);
	}

	private float speed = 0.0f;
	float yTrans = 0.0f;
	float fireTranslate = 0.0f;
	float camY = 0.0f;
	private void display(){

		float translateMatrix[] = new float[4 * 4];
		float rotateMatrix[] = new float[4 * 4];
		float modelMatrix[] = new float[4 * 4];
		float viewMatrix[] = new float[4*4];



		GLES32.glClear(GLES32.GL_COLOR_BUFFER_BIT | GLES32.GL_DEPTH_BUFFER_BIT);

		GLES32.glUseProgram(shaderProgramObject);


			switch(iSequence){

				case 1:

					//With only Two light
					/********** Matrix **********/
					Matrix.setIdentityM(translateMatrix, 0);
					Matrix.setIdentityM(rotateMatrix, 0);
					Matrix.setIdentityM(modelMatrix, 0);
					Matrix.setIdentityM(viewMatrix, 0);

					Matrix.translateM(translateMatrix, 0, 0.0f, -75.0f + yTrans, 0.0f);		//-75.0f + yTrans
					Matrix.multiplyMM(modelMatrix, 0, modelMatrix, 0, translateMatrix, 0);
					
					Matrix.setLookAtM(viewMatrix, 0,
									0.0f , 16.0f, 64.0f , 
									0.0f , 0.0f, 0.0f,
									0.0f, 1.0f, 0.0f);



					GLES32.glUniformMatrix4fv(modelMatrixUniform, 1, false, modelMatrix, 0);
					GLES32.glUniformMatrix4fv(viewMatrixUniform, 1, false, viewMatrix, 0);
					GLES32.glUniformMatrix4fv(projectionMatrixUniform, 1, false, perspectiveProjectionMatrix, 0);
					GLES32.glUniform1f(speedUniform, speed);

					speed = speed + 0.04f;
					if(speed > 360.0f)
						speed = 0.0f;

					

					yTrans += 0.07f;
					if(yTrans > 75.0f)
						yTrans = 75.0f;


					if(bLights == true){

							/*myFireRaise(fireTranslate, 1);
							fireTranslate = fireTranslate + 0.1f;
							if(fireTranslate > 64.0f)
								fireTranslate = 64.0f;*/

							GLES32.glUniform1i(LKeyPressUniform, 1);		


							GLES32.glUniform3fv(light1_la_Uniform, 1, light1_Ambient, 0);
							GLES32.glUniform3fv(light1_ld_Uniform, 1, light1_Diffuse, 0);
							GLES32.glUniform3fv(light1_ls_Uniform, 1, light1_Specular, 0);
							GLES32.glUniform4fv(light1_PositionUniform, 1, light1_Position, 0);


							GLES32.glUniform3fv(light4_la_Uniform, 1, light4_Ambient, 0);
							GLES32.glUniform3fv(light4_ld_Uniform, 1, light4_Diffuse, 0);
							GLES32.glUniform3fv(light4_ls_Uniform, 1, light4_Specular, 0);
							GLES32.glUniform4fv(light4_PositionUniform, 1, light4_Position, 0);

							GLES32.glUniform3fv(ka_Uniform, 1, materialAmbient, 0);
							GLES32.glUniform3fv(kd_Uniform, 1, materialDiffuse, 0);
							GLES32.glUniform3fv(ks_Uniform, 1, materialSpecular, 0);
							GLES32.glUniform1f(shininessUniform, materialShininess);

					}
					else
						GLES32.glUniform1i(LKeyPressUniform, 0);

					sceneCounter = sceneCounter - 0.07f;
					if(sceneCounter < 0.0f){
						sceneCounter = 50.0f;
						iSequence = 2;
						fireTranslate = 0.5f;
						System.out.println("RTR: Sceae 2 Start!!\n");
					}

					break;


				case 2:
					//With Emmerging Two light
					Matrix.setIdentityM(translateMatrix, 0);
					Matrix.setIdentityM(rotateMatrix, 0);
					Matrix.setIdentityM(modelMatrix, 0);
					Matrix.setIdentityM(viewMatrix, 0);

					Matrix.translateM(translateMatrix, 0, 0.0f, 0.0f, 0.0f);
					Matrix.multiplyMM(modelMatrix, 0, modelMatrix, 0, translateMatrix, 0);
				
					Matrix.setLookAtM(viewMatrix, 0,
									0.0f , 16.0f, 64.0f, 
									0.0f , 0.0f, 0.0f,
									0.0f, 1.0f, 0.0f);



					GLES32.glUniformMatrix4fv(modelMatrixUniform, 1, false, modelMatrix, 0);
					GLES32.glUniformMatrix4fv(viewMatrixUniform, 1, false, viewMatrix, 0);
					GLES32.glUniformMatrix4fv(projectionMatrixUniform, 1, false, perspectiveProjectionMatrix, 0);
					GLES32.glUniform1f(speedUniform, speed);

					speed = speed + 0.04f;
					if(speed > 360.0f)
						speed = 0.0f;

					

					if(bLights == true){

							myFireRaise(fireTranslate, 2);
							//fireTranslate = fireTranslate + 0.01f;
					
							System.out.println("RTR: " + light2_Position[1] + " " + light3_Position[1] + " "+ fireTranslate);

							GLES32.glUniform1i(LKeyPressUniform, 1);		


							/*GLES32.glUniform3fv(light1_la_Uniform, 1, light1_Ambient, 0);
							GLES32.glUniform3fv(light1_ld_Uniform, 1, light1_Diffuse, 0);
							GLES32.glUniform3fv(light1_ls_Uniform, 1, light1_Specular, 0);
							GLES32.glUniform4fv(light1_PositionUniform, 1, light1_Position, 0);*/

							
							GLES32.glUniform3fv(light2_la_Uniform, 1, light2_Ambient, 0);
							GLES32.glUniform3fv(light2_ld_Uniform, 1, light2_Diffuse, 0);
							GLES32.glUniform3fv(light2_ls_Uniform, 1, light2_Specular, 0);
							GLES32.glUniform4fv(light2_PositionUniform, 1, light2_Position, 0);

							GLES32.glUniform3fv(light3_la_Uniform, 1, light3_Ambient, 0);
							GLES32.glUniform3fv(light3_ld_Uniform, 1, light3_Diffuse, 0);
							GLES32.glUniform3fv(light3_ls_Uniform, 1, light3_Specular, 0);
							GLES32.glUniform4fv(light3_PositionUniform, 1, light3_Position, 0);

							/*GLES32.glUniform3fv(light4_la_Uniform, 1, light4_Ambient, 0);
							GLES32.glUniform3fv(light4_ld_Uniform, 1, light4_Diffuse, 0);
							GLES32.glUniform3fv(light4_ls_Uniform, 1, light4_Specular, 0);
							GLES32.glUniform4fv(light4_PositionUniform, 1, light4_Position, 0);*/

							GLES32.glUniform3fv(ka_Uniform, 1, materialAmbient, 0);
							GLES32.glUniform3fv(kd_Uniform, 1, materialDiffuse, 0);
							GLES32.glUniform3fv(ks_Uniform, 1, materialSpecular, 0);
							GLES32.glUniform1f(shininessUniform, materialShininess);

					}
					else
						GLES32.glUniform1i(LKeyPressUniform, 0);

					sceneCounter = sceneCounter - 0.07f;
					if(sceneCounter < 0.0f){
						sceneCounter = 75.0f;
						System.out.println("RTR: " + light2_Position[1] + " " + light3_Position[1] + " "+ fireTranslate);
						iSequence = 3;
						fireTranslate = 0.5f;
						System.out.println("RTR: Sceae 3 Start!!\n");
					}
					break;


				case 3:
					//End All lights are off
					Matrix.setIdentityM(translateMatrix, 0);
					Matrix.setIdentityM(rotateMatrix, 0);
					Matrix.setIdentityM(modelMatrix, 0);
					Matrix.setIdentityM(viewMatrix, 0);

					Matrix.translateM(translateMatrix, 0, 0.0f, 0.0f, 0.0f);
					Matrix.multiplyMM(modelMatrix, 0, modelMatrix, 0, translateMatrix, 0);
				


					Matrix.setLookAtM(viewMatrix, 0,
									0.0f , 16.0f, 64.0f + camY , 
									0.0f , 0.0f + camY, 0.0f,
									0.0f, 1.0f, 0.0f);

					camY = camY + 0.07f;

					GLES32.glUniformMatrix4fv(modelMatrixUniform, 1, false, modelMatrix, 0);
					GLES32.glUniformMatrix4fv(viewMatrixUniform, 1, false, viewMatrix, 0);
					GLES32.glUniformMatrix4fv(projectionMatrixUniform, 1, false, perspectiveProjectionMatrix, 0);
					GLES32.glUniform1f(speedUniform, speed);

					speed = speed + 0.04f;
					if(speed > 360.0f)
						speed = 0.0f;

					

					if(bLights == true){

							myFireRaise(fireTranslate, 3);
							//fireTranslate = fireTranslate + 0.01f;
							/*if(fireTranslate > 64.0f)
								fireTranslate = 64.0f;*/

							GLES32.glUniform1i(LKeyPressUniform, 1);		


							/*GLES32.glUniform3fv(light1_la_Uniform, 1, light1_Ambient, 0);
							GLES32.glUniform3fv(light1_ld_Uniform, 1, light1_Diffuse, 0);
							GLES32.glUniform3fv(light1_ls_Uniform, 1, light1_Specular, 0);
							GLES32.glUniform4fv(light1_PositionUniform, 1, light1_Position, 0);*/

							
							GLES32.glUniform3fv(light2_la_Uniform, 1, light2_Ambient, 0);
							GLES32.glUniform3fv(light2_ld_Uniform, 1, light2_Diffuse, 0);
							GLES32.glUniform3fv(light2_ls_Uniform, 1, light2_Specular, 0);
							GLES32.glUniform4fv(light2_PositionUniform, 1, light2_Position, 0);

							GLES32.glUniform3fv(light3_la_Uniform, 1, light3_Ambient, 0);
							GLES32.glUniform3fv(light3_ld_Uniform, 1, light3_Diffuse, 0);
							GLES32.glUniform3fv(light3_ls_Uniform, 1, light3_Specular, 0);
							GLES32.glUniform4fv(light3_PositionUniform, 1, light3_Position, 0);

							/*GLES32.glUniform3fv(light4_la_Uniform, 1, light4_Ambient, 0);
							GLES32.glUniform3fv(light4_ld_Uniform, 1, light4_Diffuse, 0);
							GLES32.glUniform3fv(light4_ls_Uniform, 1, light4_Specular, 0);
							GLES32.glUniform4fv(light4_PositionUniform, 1, light4_Position, 0);*/

							GLES32.glUniform3fv(ka_Uniform, 1, materialAmbient, 0);
							GLES32.glUniform3fv(kd_Uniform, 1, materialDiffuse, 0);
							GLES32.glUniform3fv(ks_Uniform, 1, materialSpecular, 0);
							GLES32.glUniform1f(shininessUniform, materialShininess);

					}
					else
						GLES32.glUniform1i(LKeyPressUniform, 0);

					sceneCounter = sceneCounter - 0.07f;
					if(sceneCounter < 0.0f){
						sceneCounter = 0.0f;
						iSequence = 4;

					}
					break;


			}


			GLES32.glBindVertexArray(vao_Grid[0]);


					GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Grid_Position[0]);
						ByteBuffer gridPosition_ByteBuffer = ByteBuffer.allocateDirect(grid_Position.length * 4);
						gridPosition_ByteBuffer.order(ByteOrder.nativeOrder());
						FloatBuffer gridPosition_FloatBuffer = gridPosition_ByteBuffer.asFloatBuffer();
						gridPosition_FloatBuffer.put(grid_Position);
						gridPosition_FloatBuffer.position(0);


					GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER, 
									grid_Position.length * 4,
									gridPosition_FloatBuffer,
									GLES32.GL_DYNAMIC_DRAW);

					GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);


					GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Grid_Normal[0]);

						ByteBuffer gridNormal_ByteBuffer = ByteBuffer.allocateDirect(grid_Normal.length * 4);
						gridNormal_ByteBuffer.order(ByteOrder.nativeOrder());
						FloatBuffer gridNormal_FloatBuffer = gridNormal_ByteBuffer.asFloatBuffer();
						gridNormal_FloatBuffer.put(grid_Normal);
						gridNormal_FloatBuffer.position(0);

					GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
									grid_Normal.length * 4,
									gridNormal_FloatBuffer,
									GLES32.GL_DYNAMIC_DRAW);


					GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);


					GLES32.glBindBuffer(GLES32.GL_ELEMENT_ARRAY_BUFFER, vbo_Grid_Element[0]);
					if(iGridType == GRID_TRIANGLES){
						GLES32.glDrawElements(GLES32.GL_TRIANGLES, GRID_WIDTH * GRID_HEIGHT * 6, 
										GLES32.GL_UNSIGNED_INT, 0);
					}
					else{
						GLES32.glDrawElements(GLES32.GL_LINES, GRID_WIDTH * GRID_HEIGHT * 6, 
										GLES32.GL_UNSIGNED_INT, 0);
					}	

			GLES32.glBindVertexArray(0);


		GLES32.glUseProgram(0);

		if(iSequence == 4){
			uninitialize();
			System.exit(0);
		}

		requestRender();

	}


	float fNoiseChange = 0.0f;
	float p1[] = new float[3];
	float p2[] = new float[3];
	float p3[] = new float[3];
	float p4[] = new float[3];

	private void update(){

		for(int i = 0; i < GRID_HEIGHT; i++){
			for(int j = 0; j < GRID_WIDTH; j++){

				grid_Position[(i * 3 * GRID_WIDTH) + (j * 3) + 1] = getInterpolatedNoise((j + fNoiseChange)/ 2.50f, (i + fNoiseChange) /2.50f) * GRID_AMPLITUDE;

				grid_Normal[(i * 3 * GRID_WIDTH) + (j * 3) + 0] = 0.0f;
				grid_Normal[(i * 3 * GRID_WIDTH) + (j * 3) + 1] = 0.0f;
				grid_Normal[(i * 3 * GRID_WIDTH) + (j * 3) + 2] = 0.0f;


			}	
		}



		for(int i = 0; i < GRID_WIDTH - 1; i++){

			for(int j = 0; j < GRID_HEIGHT - 1; j++){


				//XYZ for TopLeft
				p1[0] = grid_Position[(i * 3 * GRID_WIDTH) + (j * 3) + 0];
				p1[1] = grid_Position[(i * 3 * GRID_WIDTH) + (j * 3) + 1];
				p1[2] = grid_Position[(i * 3 * GRID_WIDTH) + (j * 3) + 2];

				//XYZ for BottomLeft
				p2[0] = grid_Position[((i + 1) * 3 * GRID_WIDTH) + (j * 3) + 0];
				p2[1] = grid_Position[((i + 1) * 3 * GRID_WIDTH) + (j * 3) + 1];
				p2[2] = grid_Position[((i + 1)* 3 * GRID_WIDTH) + (j * 3) + 2];

				//XYZ For TopLeft
				p3[0] = grid_Position[(i * 3 * GRID_WIDTH) + ((j + 1) * 3) + 0];
				p3[1] = grid_Position[(i * 3 * GRID_WIDTH) + ((j + 1) * 3) + 1];
				p3[2] = grid_Position[(i* 3 * GRID_WIDTH) + ((j + 1) * 3) + 2];


				//XYZ For BottomRight
				p4[0] = grid_Position[((i + 1) * 3 * GRID_WIDTH) + ((j + 1) * 3) + 0];
				p4[1] = grid_Position[((i + 1) * 3 * GRID_WIDTH) + ((j + 1) * 3) + 1];
				p4[2] = grid_Position[((i + 1)* 3 * GRID_WIDTH) + ((j + 1) * 3) + 2];



				//1ST Triangle

				//Top Left
				grid_Normal[(i * 3 * GRID_WIDTH) + (j * 3) + 0] += findNormal(p1, p2, p3, 1, 0);
				grid_Normal[(i * 3 * GRID_WIDTH) + (j * 3) + 1] += findNormal(p1, p2, p3, 1, 1);
				grid_Normal[(i * 3 * GRID_WIDTH) + (j * 3) + 2] += findNormal(p1, p2, p3, 1, 2) ;

				//Bottom Left
				grid_Normal[((i + 1)* 3 * GRID_WIDTH) + (j * 3) + 0] += findNormal(p1, p2, p3, 2, 0);
				grid_Normal[((i+ 1) * 3 * GRID_WIDTH) + (j * 3) + 1] += findNormal(p1, p2, p3, 2, 1);
				grid_Normal[((i + 1) * 3 * GRID_WIDTH) + (j * 3) + 2] += findNormal(p1, p2, p3, 2, 2);


				//Top Right
				grid_Normal[(i * 3 * GRID_WIDTH) + ((j + 1) * 3) + 0] += findNormal(p1, p2, p3, 3, 0);
				grid_Normal[(i * 3 * GRID_WIDTH) + ((j + 1) * 3) + 1] += findNormal(p1, p2, p3, 3, 1);
				grid_Normal[(i * 3 * GRID_WIDTH) + ((j + 1) * 3) + 2] += findNormal(p1, p2, p3, 3, 2);

				


				//2ND Triangle
				//Bottom Left
				grid_Normal[((i + 1) * 3 * GRID_WIDTH) + (j * 3) + 0] += findNormal(p2, p3, p4, 1, 0);
				grid_Normal[((i + 1) * 3 * GRID_WIDTH) + (j * 3) + 1] += findNormal(p2, p3, p4, 1, 1);
				grid_Normal[((i + 1) * 3 * GRID_WIDTH) + (j * 3) + 2] += findNormal(p2, p3, p4, 1, 2);


				//Top Right
				/*grid_Normal[(i * 3 * GRID_WIDTH) + ((j + 1) * 3) + 0] += findNormal(p2, p3, p4, 2, 0);
				grid_Normal[(i * 3 * GRID_WIDTH) + ((j + 1) * 3) + 1] += findNormal(p2, p3, p4, 2, 1);
				grid_Normal[(i * 3 * GRID_WIDTH) + ((j + 1) * 3) + 2] += findNormal(p2, p3, p4, 2, 2);*/

				//Bottom Right
				grid_Normal[((i + 1) * 3 * GRID_WIDTH) + ((j + 1) * 3) + 0] += findNormal(p2, p3, p4, 3, 0);
				grid_Normal[((i + 1) * 3 * GRID_WIDTH) + ((j + 1) * 3) + 1] += findNormal(p2, p3, p4, 3, 1);
				grid_Normal[((i + 1) * 3 * GRID_WIDTH) + ((j + 1) * 3) + 2] += findNormal(p2, p3, p4, 3, 2);

			}

		}

		fNoiseChange = fNoiseChange + 0.20f;	//0.15
	}


	private void myFireRaise(float yT, int iFlag){
		if(iFlag == 1){
			light1_Position[1] += yT;
			light4_Position[1] += yT;
		}
		else if(iFlag == 2){
			light2_Position[1] += yT;
			light3_Position[1] += yT;
		}
		else if(iFlag == 3){
			light2_Position[1] -= yT;
			light3_Position[1] -= yT;
		}
	}

};


