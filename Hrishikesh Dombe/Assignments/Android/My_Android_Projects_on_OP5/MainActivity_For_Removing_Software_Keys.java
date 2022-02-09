package com.geronime.eventhandling;

import android.app.Activity;
import android.os.Bundle;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.content.pm.ActivityInfo;
import android.graphics.Color;

public class MainActivity extends Activity {

	private MyView myView;
	
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);        
		
		int iBaseSystemUiVisibility = View.SYSTEM_UI_FLAG_HIDE_NAVIGATION | View.SYSTEM_UI_FLAG_IMMERSIVE | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY;
		
		// Get rid of the action bar/title bar		
		this.requestWindowFeature(Window.FEATURE_NO_TITLE);
		
		// This is required to make full screen
		this.getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);
		
		// Force activity window orientation to landscape
		MainActivity.this.setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
		
		// Set the background color to black
		getWindow().getDecorView().setBackgroundColor(Color.rgb(0, 0, 0));
		
		// Hide the soft-keys (Android 4.0 and above)
		getWindow().getDecorView().setSystemUiVisibility(iBaseSystemUiVisibility);			

		myView = new MyView(this);
		
		// Set myView as content view of the activity
		setContentView(myView);
    }
	
	@Override
	protected void onPause(){
		super.onPause();
	}
	
	@Override
	protected void onResume(){
		super.onResume();
	}
}
