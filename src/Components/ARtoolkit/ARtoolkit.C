#include <jevoisbase/Components/ARtoolkit/ARtoolkit.H>
using namespace std;
using namespace cv;


ARtoolkit::~ARtoolkit(){}
void ARtoolkit::postInit(){}
void ARtoolkit::postUninit(){}
void ARtoolkit::manualinit(){
    //initialize ARtoolkit
    ARParam         cparam;
    string markerConfigDataFilename;
    string CPARA_NAME = "not a file";
    switch(artoolkit::dictionary::get()){
       case artoolkit::Dict::AR_MATRIX_CODE_3x3 : markerConfigDataFilename = "markers0.dat";break;
       case artoolkit::Dict::AR_MATRIX_CODE_3x3_HAMMING63 :  markerConfigDataFilename = "markers1.dat";break;
       case artoolkit::Dict::AR_MATRIX_CODE_3x3_PARITY65 :  markerConfigDataFilename = "markers2.dat";break;
       default :  markerConfigDataFilename = "markers2.dat";
    }
    cout << "the carmeradatafile name is " << markerConfigDataFilename << endl;
    AR_PIXEL_FORMAT pixFormat;
    int xsize = artoolkit::xsize::get();
    int ysize = artoolkit::ysize::get();
    arParamChangeSize( &cparam, xsize, ysize, &cparam );
    //cout<<"xsize="<<xsize<<endl;
    //cout<<"ysize="<<ysize<<endl;
    //cout << "CPARA_NAME = "<<CPARA_NAME<<endl;
    if(xsize == 1280 && ysize == 1024){
      CPARA_NAME = "camera_para1280x1024.dat";
    }else if(xsize == 640 && ysize == 480){
      //cout<<"!@#!@$@#!@# I should enter here"<<endl;
      CPARA_NAME = "camera_para640x480.dat";
    }else if(xsize == 352 && ysize == 288){
      CPARA_NAME = "camera_para352x288.dat";
    }else if(xsize == 320 && ysize == 240){
      CPARA_NAME = "camera_para320x240.dat";
    }else if(xsize == 176 && ysize == 144){
      CPARA_NAME = "camera_para176x144.dat";
    }else if(xsize == 160 && ysize == 120){
      CPARA_NAME = "camera_para160x120.dat";
    }
    //cout << "CPARA_NAME = "<<CPARA_NAME<<endl;

    //ARLOG("*** Camera Parameter ***\n");
    
    //ARLOGi("Camera Parameter Name '%s'.\n", CPARA_NAME);
    if( arParamLoad(absolutePath(CPARA_NAME).c_str(), 1, &cparam) < 0 ) {
        ARLOGe("Camera parameter load error !!\n");
        exit(0);
    }
    //arParamDisp( &cparam );
    /*if( arParamLoad(CPARA_NAME.c_str(), 1, &cparam) < 0 ) {
        ARLOGe("Camera parameter load error !!\n");
        exit(0);
    }*/
    if ((gCparamLT = arParamLTCreate(&cparam, AR_PARAM_LT_DEFAULT_OFFSET)) == NULL) {
        ARLOGe("Error: arParamLTCreate.\n");
        exit(-1);
    }

    if( (arHandle=arCreateHandle(gCparamLT)) == NULL ) {
        ARLOGe("Error: arCreateHandle.\n");
        exit(0);
    }



    if( (ar3DHandle=ar3DCreateHandle(&cparam)) == NULL ) {
        ARLOGe("Error: ar3DCreateHandle.\n");
        exit(0);
    }
    //here the input is RGB so pixFormat = 1 listed by AR_PIXEL_FORMAT;
    pixFormat = AR_PIXEL_FORMAT_BGR;
    //I can use the RGB_565 avoid transformation
    //pixFormat = AR_PIXEL_FORMAT_RGB_565
    if( arSetPixelFormat(arHandle, pixFormat) < 0 ) {
        ARLOGe("Error: arSetPixelFormat.\n");
        exit(0);
    }
    if( (arPattHandle=arPattCreateHandle()) == NULL ) {
        ARLOGe("Error: arPattCreateHandle.\n");
        exit(0);
    }
    newMarkers(absolutePath(markerConfigDataFilename).c_str(), arPattHandle, &markersSquare, &markersSquareCount, &gARPattDetectionMode);
    //ARLOGi("markersSquareCount Marker count = %d\n", markersSquareCount);
    arPattAttach( arHandle, arPattHandle );
    arSetPatternDetectionMode(arHandle, AR_MATRIX_CODE_DETECTION);
    //arSetMatrixCodeType(arHandle, AR_MATRIX_CODE_3x3_HAMMING63);
    //arSetMatrixCodeType(arHandle, AR_MATRIX_CODE_3x3_PARITY65);
    switch(artoolkit::dictionary::get()){
       case artoolkit::Dict::AR_MATRIX_CODE_3x3 : arSetMatrixCodeType(arHandle, AR_MATRIX_CODE_3x3);break;
       case artoolkit::Dict::AR_MATRIX_CODE_3x3_HAMMING63 : arSetMatrixCodeType(arHandle, AR_MATRIX_CODE_3x3_HAMMING63);;break;
       case artoolkit::Dict::AR_MATRIX_CODE_3x3_PARITY65 : arSetMatrixCodeType(arHandle, AR_MATRIX_CODE_3x3_PARITY65); break;
       default : arSetMatrixCodeType(arHandle, AR_MATRIX_CODE_3x3_PARITY65);
    }
    //arSetMatrixCodeType(arHandle, matrixCodeType);

}
void ARtoolkit::detectMarkers(cv::Mat image){
  int useContPoseEstimation = artoolkit::useContPoseEstimation::get();
  ARUint8* dataPtr = image.data;
  ARMarkerInfo   *markerInfo;
  int             markerNum;
  ARdouble        err;
  
  if( arDetectMarker(arHandle, dataPtr) < 0 ) {
    //cout<<"cannot find Marker"<<endl;
  }else{		
    //get Detected Marker
   // cout<<"find Marker#!@#$"<<endl;
    markerNum = arGetMarkerNum( arHandle );
    //ARLOGi("markerNum = %d\n",markerNum);
				//cout<<"find Marker"<<endl;
    markerInfo =  arGetMarker( arHandle );
    for(int i = 0; i < markersSquareCount;i++ ){
      markersSquare[i].validPrev = markersSquare[i].valid;
      int k = -1;
      if (markersSquare[i].patt_type == AR_PATTERN_TYPE_MATRIX) {
        for (int j = 0; j < markerNum; j++) {
          if (markersSquare[i].patt_id == markerInfo[j].id) {
            if (k == -1) {
              if (markerInfo[j].cfPatt >= markersSquare[i].matchingThreshold) k = j; // First marker detected.
            } else if (markerInfo[j].cfPatt > markerInfo[k].cfPatt) k = j; // Higher confidence marker detected.
          }
        }
        
      } 
      if( k != -1){
        markersSquare[i].valid = TRUE;
        //ARLOGi("Marker %d matched pattern %d.\n", i, markerInfo[k].id);					//get the transformation matrix from the markers to camera in camera coordinate
        //rotation matrix + translation matrix
        if (markersSquare[i].validPrev && useContPoseEstimation) {
          err = arGetTransMatSquareCont(ar3DHandle, &(markerInfo[k]), markersSquare[i].trans, markersSquare[i].marker_width, markersSquare[i].trans);
        } else {
          err = arGetTransMatSquare(ar3DHandle, &(markerInfo[k]), markersSquare[i].marker_width, markersSquare[i].trans);
        }
        //error
        //cout<<"error=" << err<<endl;
        //direction of the marker
        //cout<<"direction"<<markerInfo[k].dir<<endl;
        //center of the marker on the screen
        //cout<<"\nmarkerInfo[k].pos[0] = " <<markerInfo[k].pos[0] << "\nmarkerInfo[k].pos[1] = "<<markerInfo[k].pos[1] <<"\n";	     
        circle( image, Point( markerInfo[k].pos[0], markerInfo[k].pos[1] ), 5.0, Scalar( 255, 0, 0 ), 1, 8 );
        int curid1 = markerInfo[k].id;
        //cout << "curid is " << curid << endl;
        string curidstring = "id = "+ std::to_string(curid1);
        //cout << "curidstring is " << curidstring << endl;
        

        for(int i1 = 0; i1 <4;i1++){
          Point *pt1 = new Point(markerInfo[k].vertex[i1][0], markerInfo[k].vertex[i1][1]);
          Point *pt2;
          if(i1<=2){
            pt2 = new Point(markerInfo[k].vertex[i1+1][0], markerInfo[k].vertex[i1+1][1]);
          }else{
            pt2 = new Point(markerInfo[k].vertex[0][0], markerInfo[k].vertex[0][1]);
          }
          
          line(image, *pt1, *pt2, Scalar(0,255,0), 3);
        }
        putText(image, curidstring, Point( markerInfo[k].pos[0], markerInfo[k].pos[1] ), FONT_HERSHEY_PLAIN, 1.0, Scalar(0,0,255),2);
		
        /*for(int i1=0;i1<3;i1++){
          
          cout<<markersSquare[i].trans[i1][3]<<endl;
          for(int i2 = 0; i2<4;i2++){
          cout<<"marker translational matrix = "<<markersSquare[i].trans[i1][i2]<<endl;  
          }
          
          }*/
        
        //draw(markersSquare[i].trans);
      }else{
        markersSquare[i].valid = FALSE;					
      }
    }
  }
}
