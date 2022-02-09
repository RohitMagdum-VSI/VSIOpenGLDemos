//Headers
#import <Foundation/Foundation.h>
#import <Cocoa/Cocoa.h>
#include"AppDelegate.h"
#include"MyView.h"

//	Entry point function
int main(int argc, char *argv[])
{
	NSAutoreleasePool *pPool = [[NSAutoreleasePool alloc]init];
	
	NSApp = [NSApplication sharedApplication];
	
	[NSApp setDelegate:[[AppDelegate alloc]init]];
	
	[NSApp run];
	
	[pPool release];
	
	return 0;
}
