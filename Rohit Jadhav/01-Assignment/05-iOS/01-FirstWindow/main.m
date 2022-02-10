//
//  main.m
//  01-FirstWindow
//
//  Created by user160249 on 3/22/20.
//

#import <UIKit/UIKit.h>
#import "AppDelegate.h"

int main(int argc, char * argv[]) {
    NSString * appDelegateClassName = nil;
    appDelegateClassName = NSStringFromClass([AppDelegate class]);
    
    NSAutoreleasePool *pool_RRJ = [[NSAutoreleasePool alloc]init];
    int  ret_RRJ;
    ret_RRJ = UIApplicationMain(argc, argv, nil, appDelegateClassName);
    [pool_RRJ release];
    return(ret_RRJ);
    
   /* @autoreleasepool {
        // Setup code that might create autoreleased objects goes here.
        appDelegateClassName = NSStringFromClass([AppDelegate class]);
    }*/
    //return UIApplicationMain(argc, argv, nil, appDelegateClassName);
}
