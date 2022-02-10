package com.rohit_r_jadhav.tessillation_shader;

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

//For Event 
import android.view.MotionEvent;
import android.view.GestureDetector;
import android.view.GestureDetector.OnGestureListener;
import android.view.GestureDetector.OnDoubleTapListener;
import android.content.Context;

public class GLESView extends GLSurfaceView implements GLSurfaceView.Renderer, OnGestureListener, OnDoubleTapListener{

	private GestureDetector gestureDetector_RRJ;
	private Context context_RRJ;

	//For Shader
	private int vertexShaderObject_RRJ;
	private int tessellationControlShaderObject_RRJ;
	private int tessellationEvaluationShaderObject_RRJ;
	private int fragmentShaderObject_RRJ;
	private int shaderProgramObject_RRJ;
	


	//For Triangle
	private int[] vao_Line_RRJ = new int[1];
	private int[] vbo_Line_Position = new int[1];

	//For Projection
	private float perspectiveProjectionMatrix_RRJ[] = new float[4 * 4];

	//For Uniform
	private int mvpUniform_RRJ;
	private int numberOfSegmentsUniform_RRJ;
	private int numberOfStripsUniform_RRJ;
	private int lineColorUniform_RRJ;

	private int numberOfLineSegments_RRJ = 1;
	private int iMaxSegmentFlag_RRJ = 0;
	private int MAXSEG_RRJ = 14;



	GLESView(Context drawingContext){

		super(drawingContext);
		context_RRJ = drawingContext;

		gestureDetector_RRJ = new GestureDetector(drawingContext, this, null, false);
		gestureDetector_RRJ.setOnDoubleTapListener(this);

		setEGLContextClientVersion(3);
		setRenderer(this);
		setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);
	
	}


	@Override
	public boolean onTouchEvent(MotionEvent event){
		int eventaction = event.getAction();
		if(!gestureDetector_RRJ.onTouchEvent(event))
			super.onTouchEvent(event);
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

		if(iMaxSegmentFlag_RRJ == 0){
			numberOfLineSegments_RRJ++;
			if(numberOfLineSegments_RRJ > MAXSEG_RRJ)
				iMaxSegmentFlag_RRJ = 1;
		}
		else{
			numberOfLineSegments_RRJ--;
			if(numberOfLineSegments_RRJ < 2)
				iMaxSegmentFlag_RRJ = 0;	
		}

		System.out.println("RTR: " + numberOfLineSegments_RRJ + " flags: " + iMaxSegmentFlag_RRJ);
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
		String version_RRJ = gl.glGetString(GL10.GL_VERSION);
		String glsl_version_RRJ = gl.glGetString(GLES32.GL_SHADING_LANGUAGE_VERSION);
		String vendor_RRJ = gl.glGetString(GL10.GL_VENDOR);
		String renderer_RRJ = gl.glGetString(GL10.GL_RENDERER);

		System.out.println("RTR: OpenGL Version: " + version_RRJ);
		System.out.println("RTR: OpenGLSL Version: " + glsl_version_RRJ);
		System.out.println("RTR: Vendor: "+ vendor_RRJ);
		System.out.println("RTR: Renderer: "+ renderer_RRJ);

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


	/********** Now OpenGL Starts **********/

	private void initialize(){

		/********** Vertex Shader **********/
		vertexShaderObject_RRJ = GLES32.glCreateShader(GLES32.GL_VERTEX_SHADER);

		final String vertexShaderSourceCode_RRJ = String.format(
				"#version 320 es" +
				"\n" +
				"in vec2 vPosition;" +
				"void main(void)" +
				"{" + 
					"gl_Position = vec4(vPosition, 0.0, 1.0);"  +
					
				"}" 
			);

		GLES32.glShaderSource(vertexShaderObject_RRJ, vertexShaderSourceCode_RRJ);

		GLES32.glCompileShader(vertexShaderObject_RRJ);

		int iShaderCompileStatus_RRJ[] = new int[1];
		int iInfoLogLength_RRJ[] = new int[1];
		String szInfoLog_RRJ = null;

		GLES32.glGetShaderiv(vertexShaderObject_RRJ, GLES32.GL_COMPILE_STATUS, iShaderCompileStatus_RRJ, 0);
		if(iShaderCompileStatus_RRJ[0] == GLES32.GL_FALSE){
			GLES32.glGetShaderiv(vertexShaderObject_RRJ, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength_RRJ, 0);
			if(iInfoLogLength_RRJ[0] > 0){

				szInfoLog_RRJ = GLES32.glGetShaderInfoLog(vertexShaderObject_RRJ);
				System.out.println("RTR: Vertex Shader Compilation Error: " + szInfoLog_RRJ);

				uninitialize();
				System.exit(0);
			
			}
		}


	

		/********** Tessellation Control Shader **********/
		tessellationControlShaderObject_RRJ = GLES32.glCreateShader(GLES32.GL_TESS_CONTROL_SHADER);
		final String tessellationControlShaderSourceCode_RRJ = String.format(
				"#version 320 es" +
				"\n" +
				"layout(vertices=4)out;" +
				"uniform int u_numberOfSegments;" +
				"uniform int u_numberOfStrips;" +
				"void main(void) {" +
					"gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;" +
					"gl_TessLevelOuter[0] = float(u_numberOfStrips);" +
					"gl_TessLevelOuter[1] = float(u_numberOfSegments);" +
				"}"
			);

		GLES32.glShaderSource(tessellationControlShaderObject_RRJ, tessellationControlShaderSourceCode_RRJ);
		GLES32.glCompileShader(tessellationControlShaderObject_RRJ);


		iShaderCompileStatus_RRJ[0] = 0;
		iInfoLogLength_RRJ[0] = 0;
		szInfoLog_RRJ = null;

		GLES32.glGetShaderiv(tessellationControlShaderObject_RRJ, GLES32.GL_COMPILE_STATUS, iShaderCompileStatus_RRJ, 0);
		if(iShaderCompileStatus_RRJ[0] == GLES32.GL_FALSE){	
			GLES32.glGetShaderiv(tessellationControlShaderObject_RRJ, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength_RRJ, 0);
			if(iInfoLogLength_RRJ[0] > 0){

				szInfoLog_RRJ = GLES32.glGetShaderInfoLog(tessellationControlShaderObject_RRJ);
				System.out.println("RTR: Tessellation Control Shader Compilation Error: " + szInfoLog_RRJ);

				uninitialize();
				System.exit(0);

			}

		}





		/********** Tessellation Evaluation Shader **********/
		tessellationEvaluationShaderObject_RRJ = GLES32.glCreateShader(GLES32.GL_TESS_EVALUATION_SHADER);
		final String tessellationEvaluationShaderSourceCode_RRJ = String.format(
				"#version 320 es" +
				"\n" +

				"layout(isolines)in;" +
				"uniform mat4 u_mvp_matrix;" +

				"void main(void) {" +
					
					"float u = gl_TessCoord.x;" +

					"vec3 p0 = gl_in[0].gl_Position.xyz;" +
					"vec3 p1 = gl_in[1].gl_Position.xyz;" +
					"vec3 p2 = gl_in[2].gl_Position.xyz;" +
					"vec3 p3 = gl_in[3].gl_Position.xyz;" +

					"float b0 = (1.0 - u) * (1.0 - u) * (1.0 - u);" +
					"float b1 = 3.0 * u * (1.0 - u) * (1.0 - u);" +
					"float b2 = 3.0 * u * u * (1.0 - u);" +
					"float b3 = u * u * u;" +

					"vec3 p = p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3;" +
					"gl_Position = u_mvp_matrix * vec4(p, 1.0);" +

				"}" 

			);



		GLES32.glShaderSource(tessellationEvaluationShaderObject_RRJ, tessellationEvaluationShaderSourceCode_RRJ);
		GLES32.glCompileShader(tessellationEvaluationShaderObject_RRJ);


		iShaderCompileStatus_RRJ[0] = 0;
		iInfoLogLength_RRJ[0] = 0;
		szInfoLog_RRJ = null;

		GLES32.glGetShaderiv(tessellationEvaluationShaderObject_RRJ, GLES32.GL_COMPILE_STATUS, iShaderCompileStatus_RRJ, 0);
		if(iShaderCompileStatus_RRJ[0] == GLES32.GL_FALSE){	
			GLES32.glGetShaderiv(tessellationEvaluationShaderObject_RRJ, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength_RRJ, 0);
			if(iInfoLogLength_RRJ[0] > 0){

				szInfoLog_RRJ = GLES32.glGetShaderInfoLog(tessellationEvaluationShaderObject_RRJ);
				System.out.println("RTR: Tessellation Evaluation Shader Compilation Error: " + szInfoLog_RRJ);

				uninitialize();
				System.exit(0);

			}

		}





		/********** Fragment Shader **********/
		fragmentShaderObject_RRJ = GLES32.glCreateShader(GLES32.GL_FRAGMENT_SHADER);

		final String fragmentShaderSourceCode_RRJ = String.format(
				"#version 320 es" +
				"\n" +
				"precision highp float;" +
				"uniform vec4 u_lineColor;" +
				"out vec4 FragColor;" +
				"void main(void)" +
				"{" +
					"FragColor = u_lineColor;" +
				"}"
			);

		GLES32.glShaderSource(fragmentShaderObject_RRJ, fragmentShaderSourceCode_RRJ);

		GLES32.glCompileShader(fragmentShaderObject_RRJ);

		GLES32.glGetShaderiv(fragmentShaderObject_RRJ, GLES32.GL_COMPILE_STATUS, iShaderCompileStatus_RRJ, 0);
		if(iShaderCompileStatus_RRJ[0] == GLES32.GL_FALSE){
			GLES32.glGetShaderiv(fragmentShaderObject_RRJ, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength_RRJ, 0);
			if(iInfoLogLength_RRJ[0] > 0){
				szInfoLog_RRJ = GLES32.glGetShaderInfoLog(fragmentShaderObject_RRJ);
				System.out.println("RTR: Fragment Shader Compilation Error: "+ szInfoLog_RRJ);
				uninitialize();
				System.exit(0);
			}
		}


		/********** Shader Program Object **********/
		shaderProgramObject_RRJ = GLES32.glCreateProgram();

		GLES32.glAttachShader(shaderProgramObject_RRJ, vertexShaderObject_RRJ);
		GLES32.glAttachShader(shaderProgramObject_RRJ, tessellationControlShaderObject_RRJ);
		GLES32.glAttachShader(shaderProgramObject_RRJ, tessellationEvaluationShaderObject_RRJ);
		GLES32.glAttachShader(shaderProgramObject_RRJ, fragmentShaderObject_RRJ);

		GLES32.glBindAttribLocation(shaderProgramObject_RRJ, GLESMacros.AMC_ATTRIBUTE_POSITION, "vPosition");

		GLES32.glLinkProgram(shaderProgramObject_RRJ);

		int iProgramLinkStatus_RRJ[] = new int[1];
		iInfoLogLength_RRJ[0] = 0;
		szInfoLog_RRJ = null;

		GLES32.glGetProgramiv(shaderProgramObject_RRJ, GLES32.GL_LINK_STATUS, iProgramLinkStatus_RRJ, 0);
		if(iProgramLinkStatus_RRJ[0] == GLES32.GL_FALSE){
			GLES32.glGetProgramiv(shaderProgramObject_RRJ, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength_RRJ, 0);
			if(iInfoLogLength_RRJ[0] > 0){
				szInfoLog_RRJ = GLES32.glGetProgramInfoLog(shaderProgramObject_RRJ);
				System.out.println("RTR: Shader Program Linking Error: "+ szInfoLog_RRJ);
				uninitialize();
				System.exit(0);
			}
		}



		mvpUniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_mvp_matrix");
		numberOfSegmentsUniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_numberOfSegments");
		numberOfStripsUniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_numberOfStrips");
		lineColorUniform_RRJ = GLES32.glGetUniformLocation(shaderProgramObject_RRJ, "u_lineColor");




		/********** Position and Color **********/
		final float line_Pos_RRJ[] = new float[]{
			-1.0f, -1.0f, 
			-0.5f, 1.0f, 
			0.5f, -1.0f, 
			1.0f, 1.0f
		};
			

		/********** Triangle **********/
		GLES32.glGenVertexArrays(1, vao_Line_RRJ, 0);
		GLES32.glBindVertexArray(vao_Line_RRJ[0]);

			/********** Position **********/
			GLES32.glGenBuffers(1, vbo_Line_Position, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Line_Position[0]);

				/********** For glBufferData() ***********/
				//Allocate ByteBuffer
				ByteBuffer byteBuffer_Position_RRJ = ByteBuffer.allocateDirect(line_Pos_RRJ.length * 4);
				//Set native ByteOrder
				byteBuffer_Position_RRJ.order(ByteOrder.nativeOrder());
				//Float Buffer
				FloatBuffer positionBuffer_RRJ = byteBuffer_Position_RRJ.asFloatBuffer();
				//Put Data into Float Buffer
				positionBuffer_RRJ.put(line_Pos_RRJ);
				//Set Position to 0
				positionBuffer_RRJ.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER, 
								line_Pos_RRJ.length * 4,
								positionBuffer_RRJ,
								GLES32.GL_STATIC_DRAW);

			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_POSITION,
										2,
										GLES32.GL_FLOAT,
										false,
										0, 0);
			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);


			
		GLES32.glBindVertexArray(0);


		GLES32.glLineWidth(10.0f);

		numberOfLineSegments_RRJ = 1;

		GLES32.glEnable(GLES32.GL_DEPTH_TEST);
		GLES32.glDepthFunc(GLES32.GL_LEQUAL);

		GLES32.glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	}




	private void uninitialize(){

		

		if(vbo_Line_Position[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Line_Position, 0);
			vbo_Line_Position[0] = 0;
		}

		if(vao_Line_RRJ[0] != 0){
			GLES32.glDeleteVertexArrays(1, vao_Line_RRJ, 0);
			vao_Line_RRJ[0] = 0;
		}

		if(shaderProgramObject_RRJ != 0){

			int iShaderCount[] = new int[1];
			int iShaderNo;

			GLES32.glUseProgram(shaderProgramObject_RRJ);

					/*GLES32.glGetProgramiv(shaderProgramObject_RRJ, GLES32.GL_ATTACHED_SHADERS, iShaderCount, 0);
					int iShaders[] = new int[iShaderCount[0]];
					GLES32.glGetAttachedShaders(shaderProgramObject_RRJ, iShaderCount[0],
																iShaderCount, 0,
																iShaders, 0);
					
					for(iShaderNo = 0; iShaderNo < iShaderCount[0]; iShaderNo++){
							GLES32.glDetachShader(shaderProgramObject_RRJ, iShaders[iShaderNo]);
							GLES32.glDeleteShader(iShaders[iShaderNo]);
							iShaders[iShaderNo] = 0;
					}*/



					if(fragmentShaderObject_RRJ != 0){
						GLES32.glDetachShader(shaderProgramObject_RRJ, fragmentShaderObject_RRJ);
						GLES32.glDeleteShader(fragmentShaderObject_RRJ);
						fragmentShaderObject_RRJ = 0;
					}

					if(tessellationEvaluationShaderObject_RRJ != 0){
						GLES32.glDetachShader(shaderProgramObject_RRJ, tessellationEvaluationShaderObject_RRJ);
						GLES32.glDeleteShader(tessellationEvaluationShaderObject_RRJ);
						tessellationEvaluationShaderObject_RRJ = 0;
					}

					if(tessellationControlShaderObject_RRJ !=0){
						GLES32.glDetachShader(shaderProgramObject_RRJ, tessellationControlShaderObject_RRJ);
						GLES32.glDeleteShader(tessellationControlShaderObject_RRJ);
						tessellationControlShaderObject_RRJ = 0;
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

		float translateMatrix_RRJ[] = new float[4 * 4];
		float modelViewMatrix_RRJ[] = new float[4 * 4];
		float modelViewProjectionMatrix_RRJ[] = new float[4 * 4];

		GLES32.glClear(GLES32.GL_COLOR_BUFFER_BIT | GLES32.GL_DEPTH_BUFFER_BIT);

		

		GLES32.glUseProgram(shaderProgramObject_RRJ);

			/********** Lines **********/
			Matrix.setIdentityM(translateMatrix_RRJ, 0);
			Matrix.setIdentityM(modelViewMatrix_RRJ, 0);
			Matrix.setIdentityM(modelViewProjectionMatrix_RRJ, 0);

			Matrix.translateM(translateMatrix_RRJ, 0, 0.0f, 0.0f, -4.0f);
			Matrix.multiplyMM(modelViewMatrix_RRJ, 0, modelViewMatrix_RRJ, 0, translateMatrix_RRJ, 0);
			Matrix.multiplyMM(modelViewProjectionMatrix_RRJ, 0, perspectiveProjectionMatrix_RRJ, 0, modelViewMatrix_RRJ, 0);


			

			GLES32.glUniformMatrix4fv(mvpUniform_RRJ, 1, false, modelViewProjectionMatrix_RRJ, 0);	
			GLES32.glUniform1i(numberOfSegmentsUniform_RRJ, numberOfLineSegments_RRJ);
			GLES32.glUniform1i(numberOfStripsUniform_RRJ, 1);

			if(iMaxSegmentFlag_RRJ == 0)
				GLES32.glUniform4f(lineColorUniform_RRJ, 1.0f, 1.0f, 0.0f, 1.0f);
			else 
				GLES32.glUniform4f(lineColorUniform_RRJ, 0.0f, 1.0f, 0.0f, 1.0f);

			GLES32.glBindVertexArray(vao_Line_RRJ[0]);

				GLES32.glPatchParameteri(GLES32.GL_PATCH_VERTICES, 4);
				GLES32.glDrawArrays(GLES32.GL_PATCHES, 0, 4);

			GLES32.glBindVertexArray(0);	

		GLES32.glUseProgram(0);

		requestRender();

	}


};

