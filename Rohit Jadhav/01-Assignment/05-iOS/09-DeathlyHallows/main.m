//
//  main.m
//  09-DeathlyHallows
//
//  Created by user160249 on 3/23/20.
//

#import <UIKit/UIKit.h>
#import "AppDelegate.h"

int main(int argc, char * argv[]) {
    NSString * appDelegateClassName;
    int ret_RRJ;
    NSAutoreleasePool *pool_RRJ = [[NSAutoreleasePool alloc] init];
    appDelegateClassName = NSStringFromClass([AppDelegate class]);
    ret_RRJ = UIApplicationMain(argc, argv, nil, appDelegateClassName);
    [pool_RRJ release];
    return(ret_RRJ);
}
