package com.astromedicomp.gestures;

import android.app.Activity;
import android.os.Bundle;
import android.graphics.Color;
import android.view.Window;
import android.view.WindowManager;
import android.content.pm.ActivityInfo;

public class MainActivity extends Activity {

	private MyView myView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {

	//This is done to remove ActionBar
	requestWindowFeature(Window.FEATURE_NO_TITLE);

	//This is done to make Window Fullscreen
	getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,WindowManager.LayoutParams.FLAG_FULLSCREEN);

     super.onCreate(savedInstanceState);
     //setContentView(R.layout.activity_main);
	
	//Force activity window orientation to landscape
	MainActivity.this.setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);

	//Setting Background Color Black
	getWindow().getDecorView().setBackgroundColor(Color.rgb(0,0,0));
	
	myView = new MyView(this);

	setContentView(myView);
    }
}
