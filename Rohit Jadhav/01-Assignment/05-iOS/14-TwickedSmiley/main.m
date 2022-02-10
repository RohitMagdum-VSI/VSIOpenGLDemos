//
//  main.m
//  14-TwickedSmiley
//
//  Created by user160249 on 3/27/20.
//

#import <UIKit/UIKit.h>
#import "AppDelegate.h"

int main(int argc, char * argv[]) {
    NSString * appDelegateClassName;
    appDelegateClassName = NSStringFromClass([AppDelegate class]);
    NSAutoreleasePool *pool_RRJ = [[NSAutoreleasePool alloc]init];
    int ret_RRJ = UIApplicationMain(argc, argv, nil, appDelegateClassName);
    [pool_RRJ release];
    return(ret_RRJ);
}
