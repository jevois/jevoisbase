#!/bin/sh

# This script only works on Prof Itti's computer, do not use it!

/bin/rm -rf /lab/jevois/basedoc/*

cd doc/html
tar cvf - . | ( cd /lab/jevois/basedoc; tar xf - )

