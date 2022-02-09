Rect when made local it always initialize to default value everytime it comes in WndProc.
So either make it "static Rect rc" or declare Rect as Global