# Get the following contribs:

######### tiny-cnn convolutional neural networks:
git clone https://github.com/nyanp/tiny-cnn.git

# no config required

########## zbar barcode / QR-code decoder:
git clone https://github.com/ZBar/ZBar.git

# make sure you have all the tools described in ZBar/HACKING
cd ZBar
autoreconf --install
cp Makefile.am Makefile.in
cp java/Makefile.am java/Makefile.in
./configure --without-x --without-xshm --without-xv --without-jpeg --without-imagemagick --without-graphicsmagick --without-npapi --without-gtk --without-python --without-qt --without-java --enable-video=no
# That's all we need for jevois, no need to compile, we just wanted ZBar/include/config.h


######### To contribute your changes: run ./undate-patch.sh and commit the modified patches.
