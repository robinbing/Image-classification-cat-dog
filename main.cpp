#include "cv.h"    
#include "highgui.h" 
#include "cxcore.h"
#include <ml.h>    
#include <iostream>    
#include <fstream>    
#include <string>    
#include <vector>    
using namespace std;
using namespace cv;



int main(int argc, char** argv)
{
	vector<string> img_path;  //this variable will record the names of image files  
	vector<int> img_catg;
	int nLine = 0;
	string buf;
	ifstream svm_data("E://forfun/trainlist.txt"); //In .txt, it's a list of all the file names
	unsigned long n;
	
	while (svm_data)//read the train files one by one  
	{
		if (getline(svm_data, buf))
		{
			nLine++;
			if (nLine < 1400)  //this number is the number of cats training images
			{
				img_catg.push_back(1); //give the label 1 denotes cat
				img_path.push_back(buf);  
			}
			else
			{
				img_catg.push_back(0); // give the label 0 denotes dog
				img_path.push_back(buf);  
			}
		}
	}
	svm_data.close();//close file
	
	CvMat *data_mat, *res_mat;
	int nImgNum = nLine;   //the number of the files
	data_mat = cvCreateMat(nImgNum, 1764, CV_32FC1);  //1764 can be obtained by descriptors.size()   
	res_mat = cvCreateMat(nImgNum, 1, CV_32FC1);
	cvSetZero(res_mat);

	IplImage* src;
	IplImage* trainImg = cvCreateImage(cvSize(64, 64), 8, 3);//one image, give the size 64*64
	
	//start to get HOG descriptor
	for (string::size_type i = 0; i != img_path.size(); i++)
	{
		src = cvLoadImage(img_path[i].c_str(), 1);
		if (src == NULL)
		{
			cout << " can not load the image: " << img_path[i].c_str() << endl;
			continue;
		}

		cout << " processing " << img_path[i].c_str() << endl;

		cvResize(src, trainImg);   //read the image       
		HOGDescriptor *hog = new HOGDescriptor(cvSize(64, 64), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);    
		//There are 4 parameters:
		//winSize(64,128): the size of the window 
		//blockSize(16,16): the size of the block
		//blockStride(8,8): the move length of the block
		//cellSize(8,8): the size of the cells
		
		vector<float>descriptors;   
		hog->compute(trainImg, descriptors, Size(1, 1), Size(0, 0));       
		cout << "HOG dims: " << descriptors.size() << endl; 
		n = 0;
		for (vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++)
		{
			cvmSet(data_mat, i, n, *iter);//save the HOG descriptor   
			n++;
		}    
		cvmSet(res_mat, i, 0, img_catg[i]);
		cout << " end processing " << img_path[i].c_str() << " " << img_catg[i] << endl;
	}


	CvSVM svm;    
	CvSVMParams param; 
	CvTermCriteria criteria;
	criteria = cvTermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	param = CvSVMParams(CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.09, 1.0, 10.0, 0.5, 1.0, NULL, criteria);
	
	//train SVM
	svm.train(data_mat, res_mat, NULL, NULL, param);            
	svm.save("E://test/SVM_DATA.xml");

	//  
	IplImage *test;
	vector<string> img_tst_path;
	ifstream img_tst("E://test/testlist.txt");//The same as reading training files, but we dont need to add labels.
	while (img_tst)
	{
		if (getline(img_tst, buf))
		{
			img_tst_path.push_back(buf);
		}
	}
	img_tst.close();



	CvMat *test_hog = cvCreateMat(1, 1764, CV_32FC1);  
	char line[512];
	ofstream predict_txt("E://test/predict.txt");   
	for (string::size_type j = 0; j != img_tst_path.size(); j++)  
	{
		test = cvLoadImage(img_tst_path[j].c_str(), 1);
		if (test == NULL)
		{
			cout << " can not load the image: " << img_tst_path[j].c_str() << endl;
			continue;
		}

		cvZero(trainImg);
		cvResize(test, trainImg);   //read the image      
		HOGDescriptor *hog = new HOGDescriptor(cvSize(64, 64), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);  //具体意思见参考文章1,2       
		vector<float>descriptors;     
		hog->compute(trainImg, descriptors, Size(1, 1), Size(0, 0));       
		cout << "HOG dims: " << descriptors.size() << endl;
		CvMat* SVMtrainMat = cvCreateMat(1, descriptors.size(), CV_32FC1);
		n = 0;
		for (vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++)
		{
			cvmSet(SVMtrainMat, 0, n, *iter);
			n++;
		}

		int ret = svm.predict(SVMtrainMat); 
		std::sprintf(line, "%s %d\r\n", img_tst_path[j].c_str(), ret);
		predict_txt << line;
	}
	predict_txt.close();

  
	cvReleaseMat(&data_mat);
	cvReleaseMat(&res_mat);

	return 0;
}

	