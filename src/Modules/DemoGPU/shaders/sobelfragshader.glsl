precision mediump float;

varying vec2 tcoord;
uniform sampler2D tex;
uniform vec2 texelsize;

void main(void) 
{
	vec4 tm1m1 = texture2D(tex,tcoord+vec2(-1,-1)*texelsize);
	vec4 tm10 = texture2D(tex,tcoord+vec2(-1,0)*texelsize);
	vec4 tm1p1 = texture2D(tex,tcoord+vec2(-1,1)*texelsize);
	vec4 tp1m1 = texture2D(tex,tcoord+vec2(1,-1)*texelsize);
	vec4 tp10 = texture2D(tex,tcoord+vec2(1,0)*texelsize);
	vec4 tp1p1 = texture2D(tex,tcoord+vec2(1,1)*texelsize);
	vec4 t0m1 = texture2D(tex,tcoord+vec2(0,-1)*texelsize);
	vec4 t0p1 = texture2D(tex,tcoord+vec2(0,1)*texelsize);

	vec4 xdiff = -1.0*tm1m1 + -2.0*tm10 + -1.0*tm1p1 + 1.0*tp1m1 + 2.0*tp10 + 1.0*tp1p1;
	vec4 ydiff = -1.0*tm1m1 + -2.0*t0m1 + -1.0*tp1m1 + 1.0*tm1p1 + 2.0*t0p1 + 1.0*tp1p1;
	vec4 tot = sqrt(xdiff*xdiff+ydiff*ydiff);

	vec4 col = tot;
	col.a = 1.0;

	//vec4 comp = vec4(greaterThan(col,vec4(0.1)));
	//col = col * comp;

    gl_FragColor = clamp(col,vec4(0),vec4(1));
}
