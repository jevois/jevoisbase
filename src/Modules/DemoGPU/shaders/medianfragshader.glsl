/*
Chris cummings - adapted from the image processing libary in the copyright notice below
*/

/*
  Copyright (c) 2011 Michael Zucchi

  This file is part of socles, an OpenCL image processing library.

  socles is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  socles is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with socles.  If not, see <http://www.gnu.org/licenses/>.
*/

// 175
precision mediump float;

#define cas3(val0, val1) { vec4 tmp = val0; vec4 comp = vec4(greaterThan(val0,val1)); val0 = mix(tmp,val1,comp); val1 = mix(val1,tmp,comp);  }

//val0 = mix(tmp,val0,comp);
//val1 = mix(val0,tmp,comp); 
//a = select(b, a, c); b = select(x, b, c);

varying vec2 tcoord;
uniform sampler2D tex;
uniform vec2 texelsize;

void main(void) 
{
	vec4 s0 = texture2D(tex,tcoord+vec2(-1,-1)*texelsize);
	vec4 s1 = texture2D(tex,tcoord+vec2(-1,0)*texelsize);
	vec4 s2 = texture2D(tex,tcoord+vec2(-1,1)*texelsize);
	vec4 s3 = texture2D(tex,tcoord+vec2(1,-1)*texelsize);
	vec4 s4 = texture2D(tex,tcoord+vec2(1,0)*texelsize);
	vec4 s5 = texture2D(tex,tcoord+vec2(1,1)*texelsize);
	vec4 s6 = texture2D(tex,tcoord+vec2(0,-1)*texelsize);
	vec4 s7 = texture2D(tex,tcoord+vec2(0,1)*texelsize);
	vec4 s8 = texture2D(tex,tcoord+vec2(0,0)*texelsize);
	
	// stage0
	cas3(s1, s2);
	cas3(s4, s5);
	cas3(s7, s8);
	
	// 1
	cas3(s0, s1);
	cas3(s3, s4);
	cas3(s6, s7);
	
	// 2
	cas3(s1, s2);
	cas3(s4, s5);
	cas3(s7, s8);
	
	// 3/4
	cas3(s3, s6);
	cas3(s4, s7);
	cas3(s5, s8);
	cas3(s0, s3);
	
	cas3(s1, s4);
	cas3(s2, s5);
	cas3(s3, s6);
	
	//      cas3(s5, s8); // not needed for median
	cas3(s4, s7);
	cas3(s1, s3);
	
	cas3(s2, s6);
	//      cas3(s5, s7); // not needed for median
	cas3(s2, s3);
	cas3(s4, s6);
	
	cas3(s3, s4);
	//      cas3(s5, s6); // not needed for median
	
    gl_FragColor = s4;
}

