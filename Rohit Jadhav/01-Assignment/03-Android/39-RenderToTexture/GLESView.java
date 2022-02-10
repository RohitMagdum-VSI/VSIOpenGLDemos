package com.rohit_r_jadhav.render_to_texture;

import android.opengl.GLSurfaceView;
import android.opengl.GLES32;
import android.opengl.Matrix;

import javax.microedition.khronos.opengles.GL10;
import javax.microedition.khronos.egl.EGLConfig;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

//For Model Parsing
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.IOException;
import java.util.StringTokenizer;
import java.util.Vector;

//For Exception
import java.util.NoSuchElementException;


//For Texture Loading
import android.graphics.BitmapFactory;
import android.graphics.Bitmap;
import android.opengl.GLUtils;



import android.view.MotionEvent;
import android.view.GestureDetector;
import android.view.GestureDetector.OnDoubleTapListener;
import android.view.GestureDetector.OnGestureListener;
import android.content.Context;

public class GLESView extends GLSurfaceView implements GLSurfaceView.Renderer, OnDoubleTapListener, OnGestureListener{

	GestureDetector gestureDetector_RRJ;
	Context context_RRJ;
	boolean bLights = false;

	GLESView(Context drawingContext_RRJ){
		super(drawingContext_RRJ);
		context_RRJ = drawingContext_RRJ;

		gestureDetector_RRJ = new GestureDetector(drawingContext_RRJ, this, null, false);
		gestureDetector_RRJ.setOnDoubleTapListener(this);

		setEGLContextClientVersion(3);
		setRenderer(this);
		setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);
	}

	@Override
	public boolean onTouchEvent(MotionEvent e){
		int action = e.getAction();
		if(!gestureDetector_RRJ.onTouchEvent(e))
			super.onTouchEvent(e);
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
		if(bLights == false)
			bLights = true;
		else
			bLights = false;
		return(true);
	}



	/********** Methods from OnGestureListener **********/
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


	/********** Methods from GLSurfaceView.Renderer **********/
	@Override
	public void onSurfaceCreated(GL10 gl, EGLConfig config){

		String version = gl.glGetString(GL10.GL_VERSION);
		String glsl_version = gl.glGetString(GLES32.GL_SHADING_LANGUAGE_VERSION);
		String renderer = gl.glGetString(GL10.GL_RENDERER);
		String vendor = gl.glGetString(GL10.GL_VENDOR);

		System.out.println("RTR: OpenGL Version: " + version);
		System.out.println("RTR: OpenGLSL Verion: " + glsl_version);
		System.out.println("RTR: Renderer: " + renderer);
		System.out.println("RTR: Vendor: "+ vendor);

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



	//For Shader
	private int vertexShaderObject_RRJ;
	private int fragmentShaderObject_RRJ;
	private int shaderProgramObject_RRJ;

	//For Model
	private int vao_Model_RRJ[] = new int[1];
	private int vbo_Model_Position_RRJ[] = new int[1] ;
	private int vbo_Model_Texcoord_RRJ[] = new int[1];
	private int vbo_Model_Normal_RRJ[] = new int[1];
	private float angle_model_RRJ; 

	private Vector model_Vertices_RRJ = new Vector();
	private Vector model_Normal_RRJ = new Vector();
	private Vector model_Texcoord_RRJ = new Vector();

	private Vector model_Sorted_Vertices_RRJ = new Vector();
	private Vector model_Sorted_Normal_RRJ = new Vector();
	private Vector model_Sorted_Texcoord_RRJ = new Vector();


	//For Projection
	private float perspectiveProjectionMatrix_RRJ[] = new float[4*4];

	//For Uniform
	private int modelMatrixUniform_RRJ;
	private int viewMatrixUniform_RRJ;
	private int projectionMatrixUniform_RRJ;
	private int la_Uniform_RRJ;
	private int ld_Uniform_RRJ;
	private int ls_Uniform_RRJ;
	private int lightPositionUniform_RRJ;
	private int ka_Uniform_RRJ;
	private int kd_Uniform_RRJ;
	private int ks_Uniform_RRJ;
	private int shininessUniform_RRJ;
	private int LKeyPressUniform_RRJ;

	//For Lights
	float lightAmbient_RRJ[] = new float[]{0.0f, 0.0f, 0.0f, 0.0f};
	float lightDiffuse_RRJ[] = new float[] {1.0f, 1.0f, 1.0f, 1.0f};
	float lightSpecular_RRJ[] = new float[]{1.0f, 1.0f, 1.0f, 1.0f};
	float lightPosition_RRJ[] = new float[]{100.0f, 100.0f, 100.0f, 1.0f};

	//For Material
	float materialAmbient_RRJ[] = new float[]{0.0f, 0.0f, 0.0f, 0.0f};
	float materialDiffuse_RRJ[] = new float[] {1.0f, 1.0f, 1.0f, 1.0f};
	float materialSpecular_RRJ[] = new float[] {1.0f, 1.0f, 1.0f, 1.0f};
	float materialShininess_RRJ = 128.0f;


	//For Texture
	private int texture_Factory_RRJ[] = new int[1];
	private int samplerUniform_RRJ;




	//For Cube
	private int vao_Cube_RRJ[] = new int[1];
	private int vbo_Cube_Position_RRJ[] = new int[1];
	private int vbo_Cube_TexCoord_RRJ[] = new int[1];
	private int texture_Cube_RRJ[] = new int[1];
	private float angle_cube_RRJ = 360.0f;


	//For Uniform
	private int mvpUniform_RRJ;
	private int choiceUniform_RRJ;



	//For Framebuffer
	private int frameBufferObject_RRJ[] = new int[1];
	private int renderBufferObject_Depth_RRJ[] = new int[1];	
	private int viewPort_FBO_Width_RRJ = 2264;
	private int viewPort_FBO_Height_RRJ = 1080;



	private void initialize(){

		vertexShaderObject_RRJ = GLES32.glCreateShader(GLES32.GL_VERTEX_SHADER);

		final String vertexShaderCode_RRJ = String.format(
				"#version 320 es" +
				"\n" +
				"in vec4 vPosition;" +
				"in vec3 vNormal;" +
				"in vec2 vTex;" +
				"out vec2 outTex;" +

				//For Toggle
				"uniform int u_choice;" +


				//For Cube
				"uniform mat4 u_mvp_matrix;" +


				//For Model
				"uniform mat4 u_model_matrix;" +
				"uniform mat4 u_view_matrix;" +
				"uniform mat4 u_projection_matrix;" +
				"uniform vec4 u_light_position;" +

				"out vec3 lightDirection_VS;" +
				"out vec3 viewerVec_VS;" +
				"out vec3 transformedNormal_VS;" +
				
				"void main(void)" +
				"{" +

					//For Model
					"if(u_choice == 1) { " + 
						"vec4 eyeCoord = u_view_matrix * u_model_matrix * vPosition;" +

						"lightDirection_VS = vec3(u_light_position - eyeCoord);" +
						"mat3 normalMatrix = mat3(u_view_matrix * u_model_matrix);" +
						"transformedNormal_VS = (normalMatrix * vNormal);" +
						"viewerVec_VS = (-eyeCoord.xyz);" +
						"outTex = vTex;" +
						"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" +
					"}" +

					//For Cube
					"else if(u_choice == 2){" +
						"outTex = vTex;" +
						"gl_Position = u_mvp_matrix * vPosition;" +
					"}" +

				"}"  
			);


		GLES32.glShaderSource(vertexShaderObject_RRJ, vertexShaderCode_RRJ);

		GLES32.glCompileShader(vertexShaderObject_RRJ);

		int iShaderCompileStatus_RRJ[] = new int[1];
		int iInfoLogLength_RRJ[] = new int[1];
		String szInfoLog_RRJ = null;

		GLES32.glGetShaderiv(vertexShaderObject_RRJ, GLES32.GL_COMPILE_STATUS, iShaderCompileStatus_RRJ, 0);
		if(iShaderCompileStatus_RRJ[0] == GLES32.GL_FALSE){
			GLES32.glGetShaderiv(vertexShaderObject_RRJ, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength_RRJ, 0);
			if(iInfoLogLength_RRJ[0] > 0){
				szInfoLog_RRJ = GLES32.glGetShaderInfoLog(vertexShaderObject_RRJ);
				System.out.println("RTR: Vertex Shader Compilation Error:"+ szInfoLog_RRJ);
				szInfoLog_RRJ = null;
				uninitialize();
				System.exit(1);
			}
		}




		fragmentShaderObject_RRJ = GLES32.glCreateShader(GLES32.GL_FRAGMENT_SHADER);
		
		final String fragmentShaderCode_RRJ = String.format(
				"#version 320 es" +
				"\n" +
				"precision highp float;" +
				"precision highp int;" +

				"uniform int u_choice;" +

				"in vec2 outTex;" +
				"uniform sampler2D u_sampler;" +

				"in vec3 lightDirection_VS;" +
				"in vec3 transformedNormal_VS;" +
				"in vec3 viewerVec_VS;" +

				"uniform vec3 u_la;" +
				"uniform vec3 u_ld;" +
				"uniform vec3 u_ls;" +

				"uniform vec3 u_ka;" +
				"uniform vec3 u_kd;" +
				"uniform vec3 u_ks;" +
				"uniform float u_shininess;" +
				"uniform int u_LKeyPress;" +

				"out vec4 FragColor;" +

				"void main(void)" +
				"{" +	
					//For Model
					"if(u_choice == 1) {" +
						"if(u_LKeyPress == 1) {" +

							"vec3 normalize_LightDirection = normalize(lightDirection_VS);" +
							"vec3 normalize_TransformedNormal = normalize(transformedNormal_VS);" +
							"float S_Dot_N = max(dot(normalize_LightDirection, normalize_TransformedNormal), 0.0);" +

							"vec3 normalize_ViewerVec = normalize(viewerVec_VS);" +
							"vec3 reflectionVec = reflect(-normalize_LightDirection, normalize_TransformedNormal);" +
							"float R_Dot_V = max(dot(reflectionVec, normalize_ViewerVec), 0.0);" +

							"vec3 ambient = u_la * u_ka;" +
							"vec3 diffuse = u_ld * u_kd * S_Dot_N;" +
							"vec3 specular = u_ls * u_ks * max(pow(R_Dot_V, u_shininess), 0.0);" +
							"vec3 phong_ADS_light = ambient + diffuse + specular;" +
							"vec4 tex = texture(u_sampler, outTex);" +
							"FragColor = tex * vec4(phong_ADS_light, 1.0);" +

						"}" +
						"else {" +
							"FragColor = vec4(1.0, 1.0, 1.0, 1.0);" +
						"}" +

					"}" +

					//For Cube
					"else if(u_choice == 2) { " +
						"FragColor = texture(u_sampler, outTex);" +
					"}" +

				"}"
			);

		GLES32.glShaderSource(fragmentShaderObject_RRJ, fragmentShaderCode_RRJ);
		GLES32.glCompileShader(fragmentShaderObject_RRJ);

		iShaderCompileStatus_RRJ[0] = 0;
		iInfoLogLength_RRJ[0] = 0;
		szInfoLog_RRJ = null;

		GLES32.glGetShaderiv(fragmentShaderObject_RRJ, GLES32.GL_COMPILE_STATUS, iShaderCompileStatus_RRJ, 0);
		if(iShaderCompileStatus_RRJ[0] == GLES32.GL_FALSE){
			GLES32.glGetShaderiv(fragmentShaderObject_RRJ, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength_RRJ, 0);
			if(iInfoLogLength_RRJ[0] > 0){
				szInfoLog_RRJ = GLES32.glGetShaderInfoLog(fragmentShaderObject_RRJ);
				System.out.println("RTR: Fragment Shader Compilation Error:"+ szInfoLog_RRJ);
				szInfoLog_RRJ = null;
				uninitialize();
				System.exit(1);
			}
		}



		shaderProgramObject_RRJ = GLES32.glCreateProgram();

		GLES32.glAttachShader(shaderProgramObject_RRJ, vertexShaderObject_RRJ);
		GLES32.glAttachShader(shaderProgramObject_RRJ, fragmentShaderObject_RRJ);

		GLES32.glBindAttribLocation(shaderProgramObject_RRJ, GLESMacros.AMC_ATTRIBUTE_POSITION, "vPosition");
		GLES32.glBindAttribLocation(shaderProgramObject_RRJ, GLESMacros.AMC_ATTRIBUTE_TEXCOORD0, "vTex");
		GLES32.glBindAttribLocation(shaderProgramObject_RRJ, GLESMacros.AMC_ATTRIBUTE_NORMAL, "vNormal");

		GLES32.glLinkProgram(shaderProgramObject_RRJ);

		int iProgramLinkStatus_RRJ[] = new int[1];
		iInfoLogLength_RRJ[0] = 0;
		szInfoLog_RRJ = null;

		GLES32.glGetProgramiv(shaderProgramObject_RRJ, GLES32.GL_LINK_STATUS, iProgramLinkStatus_RRJ, 0);
		if(iProgramLinkStatus_RRJ[0] == GLES32.GL_FALSE){
			GLES32.glGetProgramiv(shaderProgramObject_RRJ, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength_RRJ, 0);
			if(iInfoLogLength_RRJ[0] > 0){
				szInfoLog_RRJ = GLES32.glGetProgramInfoLog(shaderProgramObject_RRJ);
				System.out.println("RTR: Shader Program Linking Error:" + szInfoLog_RRJ);
				szInfoLog_RRJ = null;
				uninitialize();
				System.exit(1);
			} 
		}



		//For Model
		modelMatrixUniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_model_matrix");
		viewMatrixUniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_view_matrix");
		projectionMatrixUniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_projection_matrix");
		la_Uniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_la");
		ld_Uniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_ld");
		ls_Uniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_ls");
		lightPositionUniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_light_position");

		ka_Uniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_ka");
		kd_Uniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_kd");
		ks_Uniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_ks");
		shininessUniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_shininess");
		LKeyPressUniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_LKeyPress");


		samplerUniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_sampler");

		//For Cube
		mvpUniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_mvp_matrix");
		choiceUniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_choice");

	

		/********** POSITION TEXCOORD NORMAL **********/
		
		LoadModel();

		float model_Pos[]  = new float[model_Sorted_Vertices_RRJ.size()];
		float model_Nor[] = new float[model_Sorted_Normal_RRJ.size()];
		float model_Tex[] = new float[model_Sorted_Texcoord_RRJ.size()];



		for(int i = 0; i < model_Sorted_Vertices_RRJ.size(); i++){

			model_Pos[i + 0] = Float.parseFloat(model_Sorted_Vertices_RRJ.get(i + 0).toString());
			model_Nor[i + 0] = Float.parseFloat(model_Sorted_Normal_RRJ.get(i + 0).toString());
		}

		for(int i = 0; i < model_Sorted_Texcoord_RRJ.size(); i++){
			model_Tex[i] = Float.parseFloat(model_Sorted_Texcoord_RRJ.get(i).toString());
		}


		GLES32.glGenVertexArrays(1, vao_Model_RRJ, 0);
		GLES32.glBindVertexArray(vao_Model_RRJ[0]);

			/********** Position **********/
			GLES32.glGenBuffers(1, vbo_Model_Position_RRJ, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Model_Position_RRJ[0]);

				ByteBuffer modelPosition_ByteBuffer = ByteBuffer.allocateDirect(model_Pos.length * 4);
				modelPosition_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer modelPosition_FloatBuffer = modelPosition_ByteBuffer.asFloatBuffer();
				modelPosition_FloatBuffer.put(model_Pos);
				modelPosition_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							model_Pos.length * 4,
							modelPosition_FloatBuffer,
							GLES32.GL_STATIC_DRAW);
			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_POSITION,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);


			/********** Texcoord **********/
			GLES32.glGenBuffers(1, vbo_Model_Texcoord_RRJ, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Model_Texcoord_RRJ[0]);

				ByteBuffer modelTexcoord_ByteBuffer = ByteBuffer.allocateDirect(model_Tex.length * 4);
				modelTexcoord_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer modelTexcoord_FloatBuffer = modelTexcoord_ByteBuffer.asFloatBuffer();
				modelTexcoord_FloatBuffer.put(model_Tex);
				modelTexcoord_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER, model_Tex.length * 4, modelTexcoord_FloatBuffer, GLES32.GL_STATIC_DRAW);
			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_TEXCOORD0,
				2, GLES32.GL_FLOAT, false, 0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_TEXCOORD0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);



			/********** Normal **********/ 
			GLES32.glGenBuffers(1, vbo_Model_Normal_RRJ, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Model_Normal_RRJ[0]);

				ByteBuffer modelNormal_ByteBuffer = ByteBuffer.allocateDirect(model_Nor.length * 4);
				modelNormal_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer modelNormal_FloatBuffer = modelNormal_ByteBuffer.asFloatBuffer();
				modelNormal_FloatBuffer.put(model_Nor);
				modelNormal_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							model_Nor.length * 4,
							modelNormal_FloatBuffer,
							GLES32.GL_STATIC_DRAW);
			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_NORMAL,
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_NORMAL);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		GLES32.glBindVertexArray(0);

		GLES32.glEnable(GLES32.GL_TEXTURE_2D);
		texture_Factory_RRJ[0] = LoadTexture(R.raw.factory_img);








		/********** CUBE **********/
		float cube_Position[] = new float[]{
			//Top
			1.0f, 1.0f, -1.0f,
			-1.0f, 1.0f, -1.0f,
			-1.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f,
			//Bottom
			1.0f, -1.0f, -1.0f,
			-1.0f, -1.0f, -1.0f,
			-1.0f, -1.0f, 1.0f,
			1.0f, -1.0f, 1.0f,
			//Front
			1.0f, 1.0f, 1.0f,
			-1.0f, 1.0f, 1.0f,
			-1.0f, -1.0f, 1.0f,
			1.0f, -1.0f, 1.0f,
			//Back
			1.0f, 1.0f, -1.0f,
			-1.0f, 1.0f, -1.0f,
			-1.0f, -1.0f, -1.0f,
			1.0f, -1.0f, -1.0f,
			//Right
			1.0f, 1.0f, -1.0f,
			1.0f, 1.0f, 1.0f,
			1.0f, -1.0f, 1.0f,
			1.0f, -1.0f, -1.0f,
			//Left
			-1.0f, 1.0f, 1.0f, 
			-1.0f, 1.0f, -1.0f, 
			-1.0f, -1.0f, -1.0f, 
			-1.0f, -1.0f, 1.0f
		};



		float cube_TexCoord[] = new float[]{
			1.0f, 0.0f,
			0.0f, 0.0f,
			0.0f, 1.0f,
			1.0f, 1.0f,

			1.0f, 0.0f,
			0.0f, 0.0f,
			0.0f, 1.0f,
			1.0f, 1.0f,

			1.0f, 0.0f,
			0.0f, 0.0f,
			0.0f, 1.0f,
			1.0f, 1.0f,

			1.0f, 0.0f,
			0.0f, 0.0f,
			0.0f, 1.0f,
			1.0f, 1.0f,

			1.0f, 0.0f,
			0.0f, 0.0f,
			0.0f, 1.0f,
			1.0f, 1.0f,

			1.0f, 0.0f,
			0.0f, 0.0f,
			0.0f, 1.0f,
			1.0f, 1.0f
		};




		/********** Cube **********/
		GLES32.glGenVertexArrays(1, vao_Cube_RRJ, 0);
		GLES32.glBindVertexArray(vao_Cube_RRJ[0]);

			/********** Position **********/
			GLES32.glGenBuffers(1, vbo_Cube_Position_RRJ, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Cube_Position_RRJ[0]);

				/********** For glBufferData **********/
				ByteBuffer Cube_positionByteBuffer = ByteBuffer.allocateDirect(cube_Position.length * 4);
				Cube_positionByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer Cube_positionBuffer = Cube_positionByteBuffer.asFloatBuffer();
				Cube_positionBuffer.put(cube_Position);
				Cube_positionBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							cube_Position.length * 4,
							Cube_positionBuffer,
							GLES32.GL_STATIC_DRAW);

			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_POSITION,
									3,
									GLES32.GL_FLOAT,
									false,
									0,  0);

			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);


			/********** TexCoord **********/
			GLES32.glGenBuffers(1, vbo_Cube_TexCoord_RRJ, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Cube_TexCoord_RRJ[0]);

				/********** For glBufferData **********/
				ByteBuffer cubeTexCoord_ByteBuffer = ByteBuffer.allocateDirect(cube_TexCoord.length * 4);
				cubeTexCoord_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer cubeTexCoord_FloatBuffer = cubeTexCoord_ByteBuffer.asFloatBuffer();
				cubeTexCoord_FloatBuffer.put(cube_TexCoord);
				cubeTexCoord_FloatBuffer.position(0);


			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER, 
							cube_TexCoord.length * 4,
							cubeTexCoord_FloatBuffer,
							GLES32.GL_STATIC_DRAW);

			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_TEXCOORD0,
									2,
									GLES32.GL_FLOAT,
									false,
									0, 0);

			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_TEXCOORD0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		GLES32.glBindVertexArray(0);





		

		/********** For FRAMEBUFFER **********/
		GLES32.glGenFramebuffers(1, frameBufferObject_RRJ, 0);
		GLES32.glBindFramebuffer(GLES32.GL_FRAMEBUFFER, frameBufferObject_RRJ[0]);

			/********** Texture **********/
			GLES32.glPixelStorei(GLES32.GL_UNPACK_ALIGNMENT, 4);
			GLES32.glGenTextures(1, texture_Cube_RRJ, 0);
			GLES32.glBindTexture(GLES32.GL_TEXTURE_2D, texture_Cube_RRJ[0]);
			GLES32.glTexParameteri(GLES32.GL_TEXTURE_2D, GLES32.GL_TEXTURE_WRAP_S, GLES32.GL_REPEAT);
			GLES32.glTexParameteri(GLES32.GL_TEXTURE_2D, GLES32.GL_TEXTURE_WRAP_T, GLES32.GL_REPEAT);
			GLES32.glTexParameteri(GLES32.GL_TEXTURE_2D, GLES32.GL_TEXTURE_MAG_FILTER, GLES32.GL_LINEAR);
			GLES32.glTexParameteri(GLES32.GL_TEXTURE_2D, GLES32.GL_TEXTURE_MIN_FILTER, GLES32.GL_LINEAR);
			
			byte dataForTexture2D[] = new byte[viewPort_FBO_Width_RRJ * viewPort_FBO_Height_RRJ * 4];
			ByteBuffer byteBuffer = ByteBuffer.allocateDirect(dataForTexture2D.length * 4);
			byteBuffer.order(ByteOrder.nativeOrder());
			byteBuffer.put(dataForTexture2D);
			byteBuffer.position(0);

			Bitmap bitmap = Bitmap.createBitmap(viewPort_FBO_Width_RRJ, viewPort_FBO_Height_RRJ, Bitmap.Config.ARGB_8888);
			bitmap.copyPixelsFromBuffer(byteBuffer);

			GLUtils.texImage2D(GLES32.GL_TEXTURE_2D, 0, bitmap, 0);

			GLES32.glBindTexture(GLES32.GL_TEXTURE_2D, 0);
			GLES32.glFramebufferTexture2D(GLES32.GL_FRAMEBUFFER, GLES32.GL_COLOR_ATTACHMENT0, GLES32.GL_TEXTURE_2D, texture_Cube_RRJ[0], 0);


			/********** Depth **********/
			GLES32.glGenRenderbuffers(1, renderBufferObject_Depth_RRJ, 0);
			GLES32.glBindRenderbuffer(GLES32.GL_RENDERBUFFER, renderBufferObject_Depth_RRJ[0]);
			GLES32.glRenderbufferStorage(GLES32.GL_RENDERBUFFER, 
				GLES32.GL_DEPTH_COMPONENT16, viewPort_FBO_Width_RRJ, viewPort_FBO_Height_RRJ);
			GLES32.glBindRenderbuffer(GLES32.GL_RENDERBUFFER, 0);

			GLES32.glFramebufferRenderbuffer(GLES32.GL_FRAMEBUFFER, GLES32.GL_DEPTH_ATTACHMENT, GLES32.GL_RENDERBUFFER, renderBufferObject_Depth_RRJ[0]);


			/********** Checking **********/
			if(GLES32.glCheckFramebufferStatus(GLES32.GL_FRAMEBUFFER) != GLES32.GL_FRAMEBUFFER_COMPLETE){
				System.out.println("RTR: ERROR: glCheckFramebufferStatus()");
				uninitialize();
				System.exit(1);
			}
			else{
				System.out.println("RTR: FrameBuffer Complete");
			}

		GLES32.glBindFramebuffer(GLES32.GL_FRAMEBUFFER, 0);



		GLES32.glEnable(GLES32.GL_DEPTH_TEST);
		GLES32.glDepthFunc(GLES32.GL_LEQUAL);

		GLES32.glClearColor(0.0f, 0.0f, 0.10f, 1.0f);

	}

	private void uninitialize(){


		if(renderBufferObject_Depth_RRJ[0] != 0){
			GLES32.glDeleteRenderbuffers(1, renderBufferObject_Depth_RRJ, 0);
			renderBufferObject_Depth_RRJ[0] = 0;
		}

		if(frameBufferObject_RRJ[0] != 0){
			GLES32.glDeleteFramebuffers(1, frameBufferObject_RRJ, 0);
			frameBufferObject_RRJ[0] = 0;

		}

		if(texture_Cube_RRJ[0] != 0){
			GLES32.glDeleteTextures(1, texture_Cube_RRJ, 0);
			texture_Cube_RRJ[0] = 0;
		}

		if(vbo_Cube_TexCoord_RRJ[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Cube_TexCoord_RRJ, 0);
			vbo_Cube_TexCoord_RRJ[0] = 0;
		}

		if(vbo_Cube_Position_RRJ[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Cube_Position_RRJ, 0);
			vbo_Cube_Position_RRJ[0] = 0;
		}

		if(vao_Cube_RRJ[0] != 0){
			GLES32.glDeleteVertexArrays(1, vao_Cube_RRJ, 0);
			vao_Cube_RRJ[0] = 0;
		}


		if(texture_Factory_RRJ[0] != 0){
			GLES32.glDeleteTextures(1, texture_Factory_RRJ, 0);
			texture_Factory_RRJ[0] = 0;
		}

		if(vbo_Model_Normal_RRJ[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Model_Normal_RRJ, 0);
			vbo_Model_Normal_RRJ[0] = 0;
		}


		if(vbo_Model_Texcoord_RRJ[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Model_Texcoord_RRJ, 0);
			vbo_Model_Texcoord_RRJ[0] = 0;
		}

		if(vbo_Model_Position_RRJ[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Model_Position_RRJ, 0);
			vbo_Model_Position_RRJ[0] = 0;
		}

		if(vao_Model_RRJ[0] != 0){
			GLES32.glDeleteVertexArrays(1, vao_Model_RRJ, 0);
			vao_Model_RRJ[0] = 0;
		}


		if(shaderProgramObject_RRJ != 0){

			GLES32.glUseProgram(shaderProgramObject_RRJ);

				/*int iShaderCount[] = new int[1];
				int iShaderNo = 0;
				GLES32.glGetProgramiv(shaderProgramObject_RRJ, GLES32.GL_ATTACHED_SHADERS, iShaderCount, 0);
				System.out.println("RTR: ShaderCount: " + iShaderCount[0]);
				int iShaders[] = new int[iShaderCount[0]];
				GLES32.glGetAttachedShaders(shaderProgramObject_RRJ, iShaderCount[0], iShaderCount, 0, iShaders, 0);
				for(iShaderNo =0; iShaderNo < iShaderCount[0]; iShaderNo++){
					GLES32.glDetachShader(shaderProgramObject_RRJ, iShaders[iShaderNo]);
					GLES32.glDeleteShader(iShaders[iShaderNo]);
					iShaders[iShaderNo] = 0;
				}*/

				if(fragmentShaderObject_RRJ != 0){
					GLES32.glDetachShader(shaderProgramObject_RRJ, fragmentShaderObject_RRJ);
					GLES32.glDeleteShader(fragmentShaderObject_RRJ);
					fragmentShaderObject_RRJ= 0;
				}

				if(vertexShaderObject_RRJ != 0){
					GLES32.glDetachShader(shaderProgramObject_RRJ, vertexShaderObject_RRJ);
					GLES32.glDeleteShader(vertexShaderObject_RRJ);
					vertexShaderObject_RRJ = 0;
				}

			GLES32.glUseProgram(0);
			GLES32.glDeleteProgram(shaderProgramObject_RRJ);
			shaderProgramObject_RRJ = 0; 
		}
	}


	private void resize(int width, int height){
		if(height == 0)
			height = 1;

		GLES32.glViewport(0, 0, width, height);

		Matrix.setIdentityM(perspectiveProjectionMatrix_RRJ, 0);

		Matrix.perspectiveM(perspectiveProjectionMatrix_RRJ, 0,
						45.0f,
						(float)width / (float)height,
						0.1f,
						100.0f);
	}



	private void display(){

		float translateMatrix_RRJ[] = new float[4*4];
		float scaleMatrix_RRJ[] = new float[4*4];
		float rotateMatrix_RRJ[] = new float[4*4];
		float modelMatrix_RRJ[] = new float[4*4];
		float viewMatrix_RRJ[] = new float[4*4];
		float modelViewMatrix_RRJ[] = new float[4*4];
		float modelViewProjectionMatrix_RRJ[] = new float[4*4];




		//Model
		GLES32.glBindFramebuffer(GLES32.GL_FRAMEBUFFER, frameBufferObject_RRJ[0]);
			GLES32.glClearColor(0.0f, 0.0f, 0.10f, 1.0f);
			GLES32.glClear(GLES32.GL_COLOR_BUFFER_BIT | GLES32.GL_DEPTH_BUFFER_BIT);

			GLES32.glUseProgram(shaderProgramObject_RRJ);

				Matrix.setIdentityM(translateMatrix_RRJ, 0);
				Matrix.setIdentityM(scaleMatrix_RRJ, 0);
				Matrix.setIdentityM(rotateMatrix_RRJ, 0);
				Matrix.setIdentityM(modelMatrix_RRJ, 0);
				Matrix.setIdentityM(viewMatrix_RRJ, 0);

				Matrix.translateM(translateMatrix_RRJ, 0, 0.0f, -1.0f, -5.50f);
				Matrix.scaleM(scaleMatrix_RRJ, 0, 0.01f, 0.01f, 0.01f);
				Matrix.rotateM(rotateMatrix_RRJ, 0, angle_model_RRJ, 0.0f, 1.0f, 0.0f);
				Matrix.multiplyMM(modelMatrix_RRJ, 0, modelMatrix_RRJ, 0, translateMatrix_RRJ, 0);
				Matrix.multiplyMM(modelMatrix_RRJ, 0, modelMatrix_RRJ, 0, scaleMatrix_RRJ, 0);
				Matrix.multiplyMM(modelMatrix_RRJ, 0, modelMatrix_RRJ, 0, rotateMatrix_RRJ, 0);

				GLES32.glUniformMatrix4fv(modelMatrixUniform_RRJ, 1, false, modelMatrix_RRJ, 0);
				GLES32.glUniformMatrix4fv(viewMatrixUniform_RRJ, 1, false, viewMatrix_RRJ, 0);
				GLES32.glUniformMatrix4fv(projectionMatrixUniform_RRJ, 1, false, perspectiveProjectionMatrix_RRJ, 0);

				if(bLights == true){

						GLES32.glUniform1i(LKeyPressUniform_RRJ, 1);

						GLES32.glUniform3fv(la_Uniform_RRJ, 1, lightAmbient_RRJ, 0);
						GLES32.glUniform3fv(ld_Uniform_RRJ, 1, lightDiffuse_RRJ, 0);
						GLES32.glUniform3fv(ls_Uniform_RRJ, 1, lightSpecular_RRJ, 0);
						GLES32.glUniform4fv(lightPositionUniform_RRJ, 1, lightPosition_RRJ, 0);

						GLES32.glUniform3fv(ka_Uniform_RRJ, 1, materialAmbient_RRJ, 0);
						GLES32.glUniform3fv(kd_Uniform_RRJ, 1, materialDiffuse_RRJ, 0);
						GLES32.glUniform3fv(ks_Uniform_RRJ, 1, materialSpecular_RRJ, 0);
						GLES32.glUniform1f(shininessUniform_RRJ, materialShininess_RRJ);

				}
				else
					GLES32.glUniform1i(LKeyPressUniform_RRJ, 0);

			GLES32.glUniform1i(choiceUniform_RRJ, 1);

			GLES32.glActiveTexture(GLES32.GL_TEXTURE0);
			GLES32.glBindTexture(GLES32.GL_TEXTURE_2D, texture_Factory_RRJ[0]);
			GLES32.glUniform1i(samplerUniform_RRJ, 0);


			GLES32.glBindVertexArray(vao_Model_RRJ[0]);
				GLES32.glDrawArrays(GLES32.GL_TRIANGLES, 0, model_Sorted_Vertices_RRJ.size() / 3);
			GLES32.glBindVertexArray(0);

			GLES32.glUseProgram(0);

		GLES32.glBindFramebuffer(GLES32.GL_FRAMEBUFFER, 0);



		GLES32.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		GLES32.glClear(GLES32.GL_COLOR_BUFFER_BIT | GLES32.GL_DEPTH_BUFFER_BIT);
		

		GLES32.glUseProgram(shaderProgramObject_RRJ);

			/********** Cube **********/
			Matrix.setIdentityM(translateMatrix_RRJ, 0);
			Matrix.setIdentityM(rotateMatrix_RRJ, 0);
			Matrix.setIdentityM(scaleMatrix_RRJ, 0);
			Matrix.setIdentityM(modelViewMatrix_RRJ, 0);
			Matrix.setIdentityM(modelViewProjectionMatrix_RRJ, 0);

			Matrix.translateM(translateMatrix_RRJ, 0, 0.0f, 0.0f, -5.50f);
			Matrix.rotateM(rotateMatrix_RRJ, 0, angle_cube_RRJ, 1.0f, 1.0f, 1.0f);
			Matrix.scaleM(scaleMatrix_RRJ, 0, 1.0f, 1.0f, 1.0f);
			Matrix.multiplyMM(modelViewMatrix_RRJ, 0, modelViewMatrix_RRJ, 0, translateMatrix_RRJ, 0);
			Matrix.multiplyMM(modelViewMatrix_RRJ, 0, modelViewMatrix_RRJ, 0, rotateMatrix_RRJ, 0);
			Matrix.multiplyMM(modelViewMatrix_RRJ, 0, modelViewMatrix_RRJ, 0, scaleMatrix_RRJ, 0);
			Matrix.multiplyMM(modelViewProjectionMatrix_RRJ, 0, perspectiveProjectionMatrix_RRJ, 0, modelViewMatrix_RRJ, 0);

			GLES32.glUniformMatrix4fv(mvpUniform_RRJ, 1, false, modelViewProjectionMatrix_RRJ, 0);

			GLES32.glUniform1i(choiceUniform_RRJ, 2);


			GLES32.glActiveTexture(GLES32.GL_TEXTURE0);
			GLES32.glBindTexture(GLES32.GL_TEXTURE_2D, texture_Cube_RRJ[0]);
			GLES32.glUniform1i(samplerUniform_RRJ, 0);

			GLES32.glBindVertexArray(vao_Cube_RRJ[0]);
				GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 0, 4);
				GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 4, 4);
				GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 8, 4);
				GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 12, 4);
				GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 16, 4);
				GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 20, 4);
			GLES32.glBindVertexArray(0);
			
		GLES32.glUseProgram(0);



		requestRender();

	}

	private void update(){
		angle_model_RRJ = angle_model_RRJ + .50f;
		if(angle_model_RRJ > 360.0f)
			angle_model_RRJ = 0.0f;

		angle_cube_RRJ = angle_cube_RRJ - 1.0f;
		if(angle_cube_RRJ < 0.0f)
			angle_cube_RRJ = 360.0f;
	}


	private int LoadTexture(int imageResourceID){

		int texture[] = new int[1];

		BitmapFactory.Options options = new BitmapFactory.Options();

		options.inScaled = false;

		Bitmap bitmap = BitmapFactory.decodeResource(context_RRJ.getResources(), 
			imageResourceID, 
			options);

		GLES32.glPixelStorei(GLES32.GL_UNPACK_ALIGNMENT, 4);
		GLES32.glGenTextures(1, texture, 0);
		GLES32.glBindTexture(GLES32.GL_TEXTURE_2D, texture[0]);
		GLES32.glTexParameteri(GLES32.GL_TEXTURE_2D, GLES32.GL_TEXTURE_MAG_FILTER, GLES32.GL_LINEAR);
		GLES32.glTexParameteri(GLES32.GL_TEXTURE_2D, GLES32.GL_TEXTURE_MIN_FILTER, GLES32.GL_LINEAR_MIPMAP_LINEAR);
		GLUtils.texImage2D(GLES32.GL_TEXTURE_2D, 0, bitmap, 0);
		GLES32.glGenerateMipmap(GLES32.GL_TEXTURE_2D);
		GLES32.glBindTexture(GLES32.GL_TEXTURE_2D, 0);

		return(texture[0]);


	}


	private void LoadModel(){


	
		InputStream inputStream_RRJ = context_RRJ.getResources().openRawResource(R.raw.factory);

		InputStreamReader inputStreamReader_RRJ = new InputStreamReader(inputStream_RRJ);

		BufferedReader bufferedReader_RRJ = new BufferedReader(inputStreamReader_RRJ);

		String str;
		String str_FaceIndices;
		StringTokenizer stringTokenizer_RRJ;
		StringTokenizer stringTokenizer_Face_RRJ;


		try {
			while((str = bufferedReader_RRJ.readLine()) != null){


				String firstToken_RRJ;
				try {
					stringTokenizer_RRJ = new StringTokenizer(str, " ");
					firstToken_RRJ = stringTokenizer_RRJ.nextToken();
				}
				catch(NoSuchElementException e){
					continue;
				}


				if(firstToken_RRJ.equalsIgnoreCase("v")) {

					float x = Float.parseFloat(stringTokenizer_RRJ.nextToken());
					float y = Float.parseFloat(stringTokenizer_RRJ.nextToken());
					float z = Float.parseFloat(stringTokenizer_RRJ.nextToken());

					model_Vertices_RRJ.add(x);
					model_Vertices_RRJ.add(y);
					model_Vertices_RRJ.add(z);

				}
				else if(firstToken_RRJ.equalsIgnoreCase("vt")){

					
					float u = Float.parseFloat(stringTokenizer_RRJ.nextToken());
					float v = Float.parseFloat(stringTokenizer_RRJ.nextToken());

					model_Texcoord_RRJ.add(u);
					model_Texcoord_RRJ.add(v);

				}
				else if(firstToken_RRJ.equalsIgnoreCase("vn")){


					float x = Float.parseFloat(stringTokenizer_RRJ.nextToken());
					float y = Float.parseFloat(stringTokenizer_RRJ.nextToken());
					float z = Float.parseFloat(stringTokenizer_RRJ.nextToken());

					model_Normal_RRJ.add(x);
					model_Normal_RRJ.add(y);
					model_Normal_RRJ.add(z);

				}
				else if(firstToken_RRJ.equalsIgnoreCase("f")){


					for(int i = 0; i < 3; i++){

						str_FaceIndices = stringTokenizer_RRJ.nextToken();
						stringTokenizer_Face_RRJ = new StringTokenizer(str_FaceIndices, "/");

						int vi = Integer.parseInt(stringTokenizer_Face_RRJ.nextToken()) - 1;
						int vt = Integer.parseInt(stringTokenizer_Face_RRJ.nextToken()) - 1;
						int vn = Integer.parseInt(stringTokenizer_Face_RRJ.nextToken()) - 1;


						//For Sorted Vertices
						float x = Float.parseFloat(model_Vertices_RRJ.get((3 * vi) + 0).toString());
						float y = Float.parseFloat(model_Vertices_RRJ.get((3 * vi) + 1).toString());
						float z = Float.parseFloat(model_Vertices_RRJ.get((3 * vi) + 2).toString());

						model_Sorted_Vertices_RRJ.add(x);
						model_Sorted_Vertices_RRJ.add(y);
						model_Sorted_Vertices_RRJ.add(z);



						//For Sorted Texcoord
						float u = Float.parseFloat(model_Texcoord_RRJ.get((2 * vt) + 0).toString());
						float v = Float.parseFloat(model_Texcoord_RRJ.get((2 * vt) + 1).toString());

						model_Sorted_Texcoord_RRJ.add(u);
						model_Sorted_Texcoord_RRJ.add(1.0f - v);


						//For Sorted Normal's
						x = Float.parseFloat(model_Normal_RRJ.get((3 * vn) + 0).toString());
						y = Float.parseFloat(model_Normal_RRJ.get((3 * vn) + 1).toString());
						z = Float.parseFloat(model_Normal_RRJ.get((3 * vn) + 2).toString());

						model_Sorted_Normal_RRJ.add(x);
						model_Sorted_Normal_RRJ.add(y);
						model_Sorted_Normal_RRJ.add(z);

					}

				}

			}


		}	
		catch(IOException ioException){
			System.out.println("Load: ");
		}
	}

}
