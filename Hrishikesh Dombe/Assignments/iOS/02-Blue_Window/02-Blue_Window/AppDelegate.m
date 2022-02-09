//
//  AppDelegate.m
//  02-Blue_Window
//
//  Created by Samarth Mabrukar on 28/06/18.
//  Copyright © 2018 Hrishikiesh Dombe. All rights reserved.
//

#import "AppDelegate.h"
#import "ViewController.h"
#import "GLESView.h"

@implementation AppDelegate
{
@private
    UIWindow *mainWindow;
    ViewController *mainViewController;
    GLESView *myView;
}

-(BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions
{
    printf("In didFinishLaunchingWithOptions");
    CGRect screenBounds=[[UIScreen mainScreen]bounds];
    
    mainWindow=[[UIWindow alloc]initWithFrame:screenBounds];
    
    printf("After initWithFrame");
    
    mainViewController=[[ViewController alloc]init];
    
    [mainWindow setRootViewController:mainViewController];
    
    myView =[[GLESView alloc] initWithFrame:screenBounds];
    
    printf("After initWithFrame:myview");
    
    [mainViewController setView:myView];
    
    [myView release];
    
    [mainWindow addSubview:[mainViewController view]];
    
    printf("After addSubview");
    
    [mainWindow makeKeyAndVisible];
    
    printf("After makeKeyAndVisible");
    
    [myView startAnimation];
    
    return(YES);
}


- (void)applicationWillResignActive:(UIApplication *)application {
    // Sent when the application is about to move from active to inactive state. This can occur for certain types of temporary interruptions (such as an incoming phone call or SMS message) or when the user quits the application and it begins the transition to the background state.
    // Use this method to pause ongoing tasks, disable timers, and invalidate graphics rendering callbacks. Games should use this method to pause the game.
    [myView stopAnimation];
}


- (void)applicationDidEnterBackground:(UIApplication *)application {
    // Use this method to release shared resources, save user data, invalidate timers, and store enough application state information to restore your application to its current state in case it is terminated later.
    // If your application supports background execution, this method is called instead of applicationWillTerminate: when the user quits.
}


- (void)applicationWillEnterForeground:(UIApplication *)application {
    // Called as part of the transition from the background to the active state; here you can undo many of the changes made on entering the background.
}


- (void)applicationDidBecomeActive:(UIApplication *)application {
    // Restart any tasks that were paused (or not yet started) while the application was inactive. If the application was previously in the background, optionally refresh the user interface.
    [myView startAnimation];
}


- (void)applicationWillTerminate:(UIApplication *)application {
    // Called when the application is about to terminate. Save data if appropriate. See also applicationDidEnterBackground:.
    [myView stopAnimation];
}

- (void)dealloc
{
    [myView release];
    [mainViewController release];
    [mainWindow release];
    [super dealloc];
}

@end
