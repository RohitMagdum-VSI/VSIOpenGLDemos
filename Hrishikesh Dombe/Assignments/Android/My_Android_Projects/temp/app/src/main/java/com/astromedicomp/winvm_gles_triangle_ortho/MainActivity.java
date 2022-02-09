package com.astromedicomp.winvm_gles_triangle_ortho;

// default supplied packages by android SDK
import android.app.Activity;
import android.os.Bundle;

// later added packages
import android.view.Window; // for "Window" class
import android.view.WindowManager; // for "WindowManager" class
import android.content.pm.ActivityInfo; // for "ActivityInfo" class

public class MainActivity extends Activity
{
    private GLESView glesView;

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        
        // this is done to get rid of ActionBar
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        
        // this is done to make Fullscreen
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);

        // force activity window orientation to Landscape
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);

        glesView=new GLESView(this);
        
        // set view as content view of the activity
        setContentView(glesView);
    }
    
    @Override
    protected void onPause()
    {
        super.onPause();
    }
    
    @Override
    protected void onResume()
    {
        super.onResume();
    }
}
