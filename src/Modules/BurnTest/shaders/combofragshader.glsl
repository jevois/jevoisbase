// This is a combination of some of the shaders used in DemoGPU

precision mediump float;

varying vec2 tcoord;
uniform sampler2D tex;
uniform vec2 texelsize;

#define cas3(val0, val1) tmp = val0; val0 = min(tmp, val1); val1 = max(tmp, val1);

void main(void) 
{
// sobel
	float tm1m1 = texture2D(tex, tcoord+vec2(-1,-1) * texelsize).r;
	float tm10 = texture2D(tex, tcoord+vec2(-1,0) * texelsize).r;
	float tm1p1 = texture2D(tex, tcoord+vec2(-1,1) * texelsize).r;
	float tp1m1 = texture2D(tex, tcoord+vec2(1,-1) * texelsize).r;
	float tp10 = texture2D(tex, tcoord+vec2(1,0) * texelsize).r;
	float tp1p1 = texture2D(tex, tcoord+vec2(1,1) * texelsize).r;
	float t0m1 = texture2D(tex, tcoord+vec2(0,-1) * texelsize).r;
	float t0p1 = texture2D(tex, tcoord+vec2(0,1) * texelsize).r;
	float t00 = texture2D(tex, tcoord).r;

	float xdiff = -1.0 * tm1m1 + -2.0 * tm10 + -1.0 * tm1p1 + 1.0 * tp1m1 + 2.0 * tp10 + 1.0 * tp1p1;
	float ydiff = -1.0 * tm1m1 + -2.0 * t0m1 + -1.0 * tp1m1 + 1.0 * tm1p1 + 2.0 * t0p1 + 1.0 * tp1p1;
	float tot = sqrt(xdiff * xdiff + ydiff * ydiff);

	float col = clamp(tot, 0.0, 1.0);

	gl_FragColor.r = col;

// median
        float tmp;
	cas3(tm10, tm1p1);
	cas3(tp10, tp1p1);
	cas3(t0p1, t00);
	cas3(tm1m1, tm10);
	cas3(tp1m1, tp10);
	cas3(t0m1, t0p1);
	cas3(tm10, tm1p1);
	cas3(tp10, tp1p1);
	cas3(t0p1, t00);
	cas3(tp1m1, t0m1);
	cas3(tp10, t0p1);
	cas3(tp1p1, t00);
	cas3(tm1m1, tp1m1);
	cas3(tm10, tp10);
	cas3(tm1p1, tp1p1);
	cas3(tp1m1, t0m1);
	cas3(tp10, t0p1);
	cas3(tm10, tp1m1);
	cas3(tm1p1, t0m1);
	cas3(tm1p1, tp1m1);
	cas3(tp10, t0m1);
	cas3(tp1m1, tp10);
	gl_FragColor.g = tp10;

// erode
   	float e1 = min(tm1m1, tm10);
	float e2 = min(tm1p1, tp1m1);
	float e3 = min(tp10, tp1p1);
	float e4 = min(t0m1, t0p1);
	float e5 = min(t00, e1);
	float e6 = min(e2, e3);
	float e7 = min(e4, e5);
	gl_FragColor.b = min(e6, e7);
	
// dilate
   	e1 = max(tm1m1, tm10);
	e2 = max(tm1p1, tp1m1);
	e3 = max(tp10, tp1p1);
	e4 = max(t0m1, t0p1);
	e5 = max(t00, e1);
	e6 = max(e2, e3);
	e7 = max(e4, e5);
	gl_FragColor.a = max(e6, e7);
}
