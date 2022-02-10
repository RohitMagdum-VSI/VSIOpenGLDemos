package com.rohit_r_jadhav.checkerboard;

import android.opengl.GLSurfaceView;
import android.opengl.GLES32;
import android.opengl.Matrix;

import javax.microedition.khronos.opengles.GL10;
import javax.microedition.khronos.egl.EGLConfig;

import android.graphics.Bitmap;
import android.opengl.GLUtils;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import android.view.MotionEvent;
import android.view.GestureDetector;
import android.view.GestureDetector.OnDoubleTapListener;
import android.view.GestureDetector.OnGestureListener;
import android.content.Context;

public class GLESView extends GLSurfaceView implements GLSurfaceView.Renderer, OnDoubleTapListener, OnGestureListener{

	GestureDetector gestureDetector;
	Context context;

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


	/********** Methods From GLSurfaceView.Renderer **********/
	@Override
	public void onSurfaceCreated(GL10 gl, EGLConfig config){

		String version = gl.glGetString(GL10.GL_VERSION);
		String glsl_version = gl.glGetString(GLES32.GL_SHADING_LANGUAGE_VERSION);
		String renderer = gl.glGetString(GL10.GL_RENDERER);
		String vendor = gl.glGetString(GL10.GL_VENDOR);

		System.out.println("RTR: OpenGL Version: " + version);
		System.out.println("RTR: OpenGLSL Version: " + glsl_version);
		System.out.println("RTR: Renderer: "+ renderer);
		System.out.println("RTR: Vendor: " + vendor);

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




	//For Rectangle
	private int vao_Rect[] = new int[1];
	private int vbo_Rect_Position[] = new int[1];
	private int vbo_Rect_TexCoord[] = new int[1];

	//For Checkerboard
	private final int CHECKIMAGE_WIDTH = 64;
	private final int CHECKIMAGE_HEIGHT = 64;
	private byte CheckImageData[] = new byte[CHECKIMAGE_WIDTH * CHECKIMAGE_HEIGHT * 4];
	private int texture_checkerboard[] = new int[1];

	//For Projection
	private float perspectiveProjectionMatrix[] = new float[4*4];

	//For Uniform
	private int mvpUniform;
	private int samplerUniform;

	//For Shaders
	private int vertexShaderObject;
	private int fragmentShaderObject;
	private int shaderProgramObject;

	private void initialize(){

		vertexShaderObject = GLES32.glCreateShader(GLES32.GL_VERTEX_SHADER);

		final String vertexShaderCode = String.format(
				"#version 320 es" +
				"\n" +
				"in vec4 vPosition;" +
				"in vec2 vTexCoord;" +
				"out vec2 outTexCoord;" +
				"uniform mat4 u_mvp_matrix;" +
				"void main() {" +
					"gl_Position = u_mvp_matrix * vPosition;" +
					"outTexCoord = vTexCoord;" +
				"}" 
			);

		GLES32.glShaderSource(vertexShaderObject, vertexShaderCode);

		GLES32.glCompileShader(vertexShaderObject);

		int iShaderCompileStatus[] = new int[1];
		int iInfoLogLength[] = new int[1];
		String szInfoLog = null;

		GLES32.glGetShaderiv(vertexShaderObject, GLES32.GL_COMPILE_STATUS, iShaderCompileStatus, 0);
		if(iShaderCompileStatus[0] == GLES32.GL_FALSE){
			GLES32.glGetShaderiv(vertexShaderObject, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if(iInfoLogLength[0] > 0){
				szInfoLog = GLES32.glGetShaderInfoLog(vertexShaderObject);
				System.out.println("RTR: Vertex Shader Error: \n" + szInfoLog);
				szInfoLog = null;
				uninitialize();
				System.exit(1);
			}
		}



		fragmentShaderObject = GLES32.glCreateShader(GLES32.GL_FRAGMENT_SHADER);

		final String fragmentShaderCode = String.format(
				"#version 320 es" +
				"\n" +
				"precision highp float;" +
				"precision highp sampler2D;" +
				"in vec2 outTexCoord;" +
				"out vec4 FragColor;" +
				"uniform sampler2D u_sampler;" +
				"void main(void) { " +
					"FragColor = texture(u_sampler, outTexCoord);" +
				"}" 
			);

		GLES32.glShaderSource(fragmentShaderObject, fragmentShaderCode);

		GLES32.glCompileShader(fragmentShaderObject);

		iShaderCompileStatus[0] = 0;
		iInfoLogLength[0] = 0;
		szInfoLog = null;

		GLES32.glGetShaderiv(fragmentShaderObject, GLES32.GL_COMPILE_STATUS, iShaderCompileStatus, 0);
		if(iShaderCompileStatus[0] == GLES32.GL_FALSE){
			GLES32.glGetShaderiv(fragmentShaderObject, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if(iInfoLogLength[0] > 0){
				szInfoLog = GLES32.glGetShaderInfoLog(fragmentShaderObject);
				System.out.println("RTR: Fragment Shader Compilation Error: \n" + szInfoLog);
				szInfoLog = null;
				uninitialize();
				System.exit(1);
			}
		}


		shaderProgramObject = GLES32.glCreateProgram();

		GLES32.glAttachShader(shaderProgramObject, vertexShaderObject);
		GLES32.glAttachShader(shaderProgramObject, fragmentShaderObject);

		GLES32.glBindAttribLocation(shaderProgramObject, GLESMacros.AMC_ATTRIBUTE_POSITION, "vPosition");
		GLES32.glBindAttribLocation(shaderProgramObject, GLESMacros.AMC_ATTRIBUTE_TEXCOORD0, "vTexCoord");

		GLES32.glLinkProgram(shaderProgramObject);

		int iProgramLinkStatus[] = new int[1];
		iInfoLogLength[0] = 0;
		szInfoLog = null;

		GLES32.glGetProgramiv(shaderProgramObject, GLES32.GL_LINK_STATUS, iProgramLinkStatus, 0);
		if(iProgramLinkStatus[0] == GLES32.GL_FALSE){
			GLES32.glGetProgramiv(shaderProgramObject, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if(iInfoLogLength[0] > 0){
				szInfoLog = GLES32.glGetProgramInfoLog(shaderProgramObject);
				System.out.println("RTR: Shader Program Linking Error: \n" + szInfoLog);
				szInfoLog = null;
				uninitialize();
				System.exit(0);
			}
		}


		mvpUniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_mvp_matrix");
		samplerUniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_sampler");



		/*********** Rectangle Position and TexCoord **********/
		float rect_TexCoord[] = new float[]{
			0.0f, 1.0f,
			0.0f, 0.0f,
			1.0f, 0.0f,
			1.0f, 1.0f
		};



		/********** Rectangle **********/
		GLES32.glGenVertexArrays(1, vao_Rect, 0);
		GLES32.glBindVertexArray(vao_Rect[0]);

			/********** Position **********/
			GLES32.glGenBuffers(1, vbo_Rect_Position, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Rect_Position[0]);
			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
								3 * 4 * 4,
								null,
								GLES32.GL_DYNAMIC_DRAW);
			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_POSITION,
									3,
									GLES32.GL_FLOAT,
									false, 
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);


			/********** TexCoord **********/
			GLES32.glGenBuffers(1, vbo_Rect_TexCoord, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Rect_TexCoord[0]);

				ByteBuffer rectTexCoord_ByteBuffer = ByteBuffer.allocateDirect(rect_TexCoord.length * 4);
				rectTexCoord_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer rectTexCoord_FloatBuffer = rectTexCoord_ByteBuffer.asFloatBuffer();
				rectTexCoord_FloatBuffer.put(rect_TexCoord);
				rectTexCoord_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							rect_TexCoord.length *  4,
							rectTexCoord_FloatBuffer,
							GLES32.GL_STATIC_DRAW);
			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_TEXCOORD0,
									2,
									GLES32.GL_FLOAT,
									false, 
									0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_TEXCOORD0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		GLES32.glBindVertexArray(0);


		GLES32.glEnable(GLES32.GL_DEPTH_TEST);
		GLES32.glDepthFunc(GLES32.GL_LEQUAL);

			GLES32.glEnable(GLES32.GL_TEXTURE_2D);
			texture_checkerboard[0] = loadTexture();

			System.out.println("RTR: Texture: " + texture_checkerboard[0]);

		GLES32.glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	}


	private void uninitialize(){

		if(texture_checkerboard[0] != 0){
			GLES32.glDeleteTextures(1, texture_checkerboard, 0);
			texture_checkerboard[0] = 0;
		}

		if(vbo_Rect_TexCoord[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Rect_TexCoord, 0);
			vbo_Rect_TexCoord[0] = 0;
		}

		if(vbo_Rect_Position[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Rect_Position, 0);
			vbo_Rect_Position[0] = 0;
		}

		if(vao_Rect[0] != 0){
			GLES32.glDeleteVertexArrays(1, vao_Rect, 0);
			vao_Rect[0] = 0;
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


	private int loadTexture(){

		int texture[] = new int[1];

		makeCheckImage();

		ByteBuffer byteBuffer = ByteBuffer.allocateDirect(CheckImageData.length * 4);
		byteBuffer.order(ByteOrder.nativeOrder());
		byteBuffer.put(CheckImageData);
		byteBuffer.position(0);

		Bitmap bitmap = Bitmap.createBitmap(CHECKIMAGE_WIDTH, CHECKIMAGE_HEIGHT, Bitmap.Config.ARGB_8888);

		bitmap.copyPixelsFromBuffer(byteBuffer);

		GLES32.glPixelStorei(GLES32.GL_UNPACK_ALIGNMENT, 1);
		GLES32.glGenTextures(1, texture, 0);
		GLES32.glBindTexture(GLES32.GL_TEXTURE_2D, texture[0]);

			GLES32.glTexParameteri(GLES32.GL_TEXTURE_2D, GLES32.GL_TEXTURE_WRAP_S, GLES32.GL_REPEAT);
			GLES32.glTexParameteri(GLES32.GL_TEXTURE_2D, GLES32.GL_TEXTURE_WRAP_T, GLES32.GL_REPEAT);
			GLES32.glTexParameteri(GLES32.GL_TEXTURE_2D, GLES32.GL_TEXTURE_MAG_FILTER, GLES32.GL_NEAREST);
			GLES32.glTexParameteri(GLES32.GL_TEXTURE_2D, GLES32.GL_TEXTURE_MIN_FILTER, GLES32.GL_NEAREST);

			GLUtils.texImage2D(GLES32.GL_TEXTURE_2D, 0, bitmap, 0);

		GLES32.glBindTexture(GLES32.GL_TEXTURE_2D, 0);
		return(texture[0]);
	}


	private void makeCheckImage(){

		int c;

		for(int i = 0; i < CHECKIMAGE_HEIGHT; i++){
			for(int j = 0; j < CHECKIMAGE_WIDTH; j++){
				c = (((i & 0x8) ^ (j & 0x8)) * 255);
				CheckImageData[(i * 64 + j) * 4 + 0] = (byte)c;
				CheckImageData[(i * 64 + j) * 4 + 1] = (byte)c;
				CheckImageData[(i * 64 + j) * 4 + 2] = (byte)c;
				CheckImageData[(i * 64 + j) * 4 + 3] = (byte)255;
			}
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
						100.0f);
	}


	private void display(){

		float translateMatrix[] = new float[4*4];
		float modelViewMatrix[] = new float[4*4];
		float modelViewProjectionMatrix[] = new float[4*4];
		float rect_Position[] = new float[4 * 3];

		GLES32.glClear(GLES32.GL_COLOR_BUFFER_BIT | GLES32.GL_DEPTH_BUFFER_BIT);

		GLES32.glUseProgram(shaderProgramObject);

		for(int i = 0; i < 2; i++){

			if(i == 0){

				rect_Position[0] = -2.0f;
				rect_Position[1] = -1.0f;
				rect_Position[2] = 0.0f;

				rect_Position[3] = -2.0f;
				rect_Position[4] = 1.0f;
				rect_Position[5] = 0.0f;

				rect_Position[6] = 0.0f;
				rect_Position[7] = 1.0f;
				rect_Position[8] = 0.0f;

				rect_Position[9] = 0.0f;
				rect_Position[10] = -1.0f;
				rect_Position[11] = 0.0f;
			}
			else if(i == 1){

				rect_Position[0] = 1.0f;
				rect_Position[1] = -1.0f;
				rect_Position[2] = 0.0f;

				rect_Position[3] = 1.0f;
				rect_Position[4] = 1.0f;
				rect_Position[5] = 0.0f;

				rect_Position[6] = 2.4142f;
				rect_Position[7] = 1.0f;
				rect_Position[8] = -1.4142f;

				rect_Position[9] = 2.4142f;
				rect_Position[10] = -1.0f;
				rect_Position[11] = -1.4142f;
			}

			Matrix.setIdentityM(translateMatrix, 0);
			Matrix.setIdentityM(modelViewMatrix, 0);
			Matrix.setIdentityM(modelViewProjectionMatrix, 0);

			Matrix.translateM(translateMatrix, 0, 0.0f, 0.0f, -3.0f);
			Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, translateMatrix, 0);
			Matrix.multiplyMM(modelViewProjectionMatrix, 0, perspectiveProjectionMatrix, 0, modelViewMatrix, 0);

			GLES32.glUniformMatrix4fv(mvpUniform, 1, false, modelViewProjectionMatrix, 0);

			GLES32.glActiveTexture(GLES32.GL_TEXTURE0);
			GLES32.glBindTexture(GLES32.GL_TEXTURE_2D, texture_checkerboard[0]);
			GLES32.glUniform1i(samplerUniform, 0);

			GLES32.glBindVertexArray(vao_Rect[0]);

				GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Rect_Position[0]);

					ByteBuffer rectPosition_ByteBuffer = ByteBuffer.allocateDirect(rect_Position.length * 4);
					rectPosition_ByteBuffer.order(ByteOrder.nativeOrder());
					FloatBuffer rectPosition_FloatBuffer = rectPosition_ByteBuffer.asFloatBuffer();
					rectPosition_FloatBuffer.put(rect_Position);
					rectPosition_FloatBuffer.position(0);

				GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER, 
								rect_Position.length * 4,
								rectPosition_FloatBuffer,
								GLES32.GL_DYNAMIC_DRAW);

				GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

				GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 0, 4);

			GLES32.glBindVertexArray(0);
		}

		GLES32.glUseProgram(0);

		requestRender();
	}


}

