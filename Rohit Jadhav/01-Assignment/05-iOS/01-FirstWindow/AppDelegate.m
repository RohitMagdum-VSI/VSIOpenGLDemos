//
//  AppDelegate.m
//  01-FirstWindow
//
//  Created by user160249 on 3/22/20.
//

/*#import "AppDelegate.h"

@interface AppDelegate ()

@end*/

#import "AppDelegate.h"
#import "ViewController.h"
#import "MyView.h"


@implementation AppDelegate{
    @private
        UIWindow *mainWindow_RRJ;
        ViewController *mainViewController_RRJ;
        MyView *myView_RRJ;
}

-(BOOL)application:(UIApplication*)application didFinishLaunchingWithOptions:(NSDictionary*)launchOptions{

    CGRect screenBounds_RRJ = [[UIScreen mainScreen]bounds];

    mainWindow_RRJ = [[UIWindow alloc] initWithFrame:screenBounds_RRJ];

    mainViewController_RRJ = [[ViewController alloc] init];

    [mainWindow_RRJ setRootViewController: mainViewController_RRJ];

    myView_RRJ = [[MyView alloc] initWithFrame:screenBounds_RRJ];

    [mainViewController_RRJ setView:myView_RRJ];

    [myView_RRJ release];

    [mainWindow_RRJ addSubview:[mainViewController_RRJ view]];

    [mainWindow_RRJ makeKeyAndVisible];

    return(YES);
}

-(void)applicationWillResignActive:(UIApplication*)application{
    //Send when the application is about to move from active to inactive state

    //Happens when and Incoming Call or SMS message

    //This method is use to pause ongoing task disable timers, invalidate graphics rendering callbacks,
    //game should use this method to pause the game
}

-(void)applicationDidEnterBackground:(UIApplication*)application{
    /*Here we store all the essential counter and timer and active states so that
    
    when we get back to running state this is used to restore the state of our app

    and if out app support background execution the we use this method as applicationWillTerminate()

    */
}

-(void)applicationDidBecomeActive:(UIApplication*)application{
    /*
    restore the data if app is in background state and now become active

    and refresh all the user interface

    */
}


-(void)applicationWillTerminate:(UIApplication*)application{
    /*
    calls when application is about to terminate */
}


-(void)dealloc{

    [myView_RRJ release];

    [mainViewController_RRJ release];

    [mainWindow_RRJ release];

    [super dealloc];
}

@end



