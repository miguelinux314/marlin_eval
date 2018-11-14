#include <dirent.h>

#include <marlinlib/marlin.hpp>

#include <fstream>
#include <map>
#include <queue>
#include <chrono>
#include <memory>
#include <iostream>

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <util/distribution.hpp>

#include <codecs/rle.hpp>
#include <codecs/snappy.hpp>
#include <codecs/nibble.hpp>
#include <codecs/charls.hpp>
#include <codecs/gipfeli.hpp>
#include <codecs/gzip.hpp>
#include <codecs/lzo.hpp>
#include <codecs/zstd.hpp>
#include <codecs/fse.hpp>
#include <codecs/rice.hpp>
#include <codecs/lz4.hpp>
#include <codecs/huf.hpp>
#include <codecs/marlin.hpp>
#include <codecs/marlin2018.hpp>
#include <codecs/marlin2019.hpp>

#include <uSnippets/cache.hpp>
#include <uSnippets/log.hpp>

struct TestTimer {
	timespec c_start, c_end;
	void start() { clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &c_start); };
	void stop () { clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &c_end); };
	double operator()() { return (c_end.tv_sec-c_start.tv_sec) + 1.E-9*(c_end.tv_nsec-c_start.tv_nsec); }
};

static inline std::vector<std::string> getAllFilenames(std::string path, std::vector<std::string> types= {""}) {
	
	std::vector<std::string> r;
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir (path.c_str())) == NULL)
		return r;
	
	while ((ent = readdir (dir)) != NULL)
		for (auto && type : types)
			if (std::string(ent->d_name).size()>=type.size() and std::string(&ent->d_name[std::string(ent->d_name).size()-type.size()])==type)
				r.push_back(path+"/"+ent->d_name);

	closedir (dir);
	std::sort(r.begin(), r.end());
	return r;
}

namespace predictors {
	
	double predictCompressedBlockSize(const std::vector<uint32_t> &histogram) {
		
		double sum = 1e-100;
		for (auto &&p : histogram)
			sum += p;
		
		
		double entropy=0;
		for (auto &&p : histogram)
			if (p)
				entropy += -double(p/sum)*std::log2(double(p/sum));

		return entropy * sum + 32;
	}
	
	
	enum PredictorType { PREDICTOR_TOP, PREDICTOR_LEFT, PREDICTOR_ABC, PREDICTOR_DC, PREDICTOR_TOP_DC } ;


	template<typename T>
	static std::vector<std::vector<uint32_t>> testPredictorHistograms(cv::Mat_<T> orig_img, size_t imageBlockWidth, PredictorType predictorType, int colorPrediction) {

		if (imageBlockWidth == 0) {


			size_t brows = (orig_img.rows+32-1)/32;
			size_t bcols = (orig_img.cols+32-1)/32;	
			cv::Mat_<T> img;
			cv::copyMakeBorder(orig_img, img, 0, brows*32-orig_img.rows, 0, bcols*32-orig_img.cols, cv::BORDER_REPLICATE);
			
			auto h32 = testPredictorHistograms(img, 32, predictorType, colorPrediction);
			auto h16 = testPredictorHistograms(img, 16, predictorType, colorPrediction);
			auto h8  = testPredictorHistograms(img, 8, predictorType, colorPrediction);
							
			
			//std::cerr << brows << " " << bcols << " " << orig_img.channels() << std::endl;
			
			std::vector<std::vector<uint32_t>> ret;
			size_t channels = orig_img.channels();
			for (size_t c=0; c<channels; c++) {
				
				for (size_t i=0; i<brows; i++) {
					for (size_t j=0; j<bcols; j++) {

						std::vector<std::vector<uint32_t>> hist16sum;
						double size16sum = 0;
						for (size_t ii=0; ii<2; ii++) {
							for (size_t jj=0; jj<2; jj++) {
								

								std::vector<std::vector<uint32_t>> hist8sum;
								double size8sum = 0;
								for (size_t iii=0; iii<2; iii++) {
									for (size_t jjj=0; jjj<2; jjj++) {
										
										uSnippets::Assert(((4*i+2*ii+iii)*4*bcols+4*j+2*jj+jjj)*channels+c < h8.size()) << "H8! " << ((4*i+2*ii+iii)*4*bcols+4*j+2*jj+jjj)*channels+c << " " << h8.size();
										auto h = h8[((4*i+2*ii+iii)*4*bcols+4*j+2*jj+jjj)*channels+c];
										double size8 = predictCompressedBlockSize(h);
										size8sum += size8;
										hist8sum.push_back(h);
									}
								}
								
								uSnippets::Assert(((2*i+1*ii)*2*bcols+2*j+1*jj)*channels+c < h16.size()) << "H16!";
								auto h = h16[((2*i+1*ii)*2*bcols+2*j+1*jj)*channels+c];
								double size16 = predictCompressedBlockSize(h);
								if (size16 < size8sum) {
									size16sum += size16;
									hist16sum.push_back(h);
								} else {
									size16sum += size8sum;
									for (auto &&hh:hist8sum)
										hist16sum.push_back(hh);
								}
							}
						}

						uSnippets::Assert((i*bcols+j)*channels+c < h32.size()) << "H32!";
						auto h = h32[(i*bcols+j)*channels+c];
						double size32 = predictCompressedBlockSize(h);
						if (size32 < size16sum) {
							ret.push_back(h);
						} else {
							for (auto &&hh:hist16sum)
								ret.push_back(hh);
						}
					}
				}
			}
			return ret;
		}


		if (predictorType == PREDICTOR_TOP_DC) {
			
			auto p1 = testPredictorHistograms(orig_img, imageBlockWidth, PREDICTOR_TOP, colorPrediction);
			auto p2 = testPredictorHistograms(orig_img, imageBlockWidth, PREDICTOR_DC, colorPrediction);
			for (uint i=0; i<p1.size(); i++)
				if (predictCompressedBlockSize(p2[i]) < predictCompressedBlockSize(p1[i]))
					p1[i] = p2[i];
					
			return p1;
		}

		const size_t bs = imageBlockWidth;

		size_t brows = (orig_img.rows+bs-1)/bs;
		size_t bcols = (orig_img.cols+bs-1)/bs;	
		cv::Mat_<T> img, imgdc;
		cv::copyMakeBorder(orig_img, img, 0, brows*bs-orig_img.rows, 0, bcols*bs-orig_img.cols, cv::BORDER_REPLICATE);
		cv::resize(img, imgdc, cv::Size(brows,bcols));

		cv::Mat_<T> imgpredicted = img.clone();

		switch (predictorType) {
			case PREDICTOR_TOP:
				for (size_t i=0; i<img.rows-bs+1; i+=bs) {
					for (size_t j=0; j<img.cols-bs+1; j+=bs) {
						imgpredicted(i,j) = imgpredicted(i,j) - imgpredicted(i,j);
						for (size_t jj=1; jj<bs; jj++)
							imgpredicted(i,j+jj) = img(i,j+jj) - img(i,j+jj-1);

						for (size_t ii=1; ii<bs; ii++)
							for (size_t jj=0; jj<bs; jj++)
								imgpredicted(i+ii,j+jj) = img(i+ii,j+jj) - img(i+ii-1,j+jj);
					}
				}
				break;
			case PREDICTOR_LEFT:

				for (size_t i=0; i<img.rows-bs+1; i+=bs) {
					for (size_t j=0; j<img.cols-bs+1; j+=bs) {
						imgpredicted(i,j) = imgpredicted(i,j) - imgpredicted(i,j);
						for (size_t ii=1; ii<bs; ii++)
							imgpredicted(i+ii,j) = img(i+ii,j) - img(i+ii-1,j);

						for (size_t ii=0; ii<bs; ii++)
							for (size_t jj=1; jj<bs; jj++)
								imgpredicted(i+ii,j+jj) = img(i+ii,j+jj) - img(i+ii,j+jj-1);
					}
				}
				break;
			case PREDICTOR_ABC:
				for (size_t i=0; i<img.rows-bs+1; i+=bs) {
					for (size_t j=0; j<img.cols-bs+1; j+=bs) {
						imgpredicted(i,j) = imgpredicted(i,j) - imgpredicted(i,j);
						for (size_t jj=1; jj<bs; jj++)
							imgpredicted(i,j+jj) = img(i,j+jj) - img(i,j+jj-1);

						for (size_t ii=1; ii<bs; ii++)
							imgpredicted(i+ii,j) = img(i+ii,j) - img(i+ii-1,j);

						for (size_t ii=1; ii<bs; ii++)
							for (size_t jj=1; jj<bs; jj++)
								imgpredicted(i+ii,j+jj) = img(i+ii,j+jj) - img(i+ii-1,j+jj) - img(i+ii,j+jj-1) + img(i+ii-1,j+jj-1);
					}
				}
				break;
			case PREDICTOR_DC:
				for (size_t i=0; i<img.rows-bs+1; i+=bs) {
					for (size_t j=0; j<img.cols-bs+1; j+=bs) {
						for (size_t ii=0; ii<bs; ii++)
							for (size_t jj=0; jj<bs; jj++)
								imgpredicted(i+ii,j+jj) = img(i+ii,j+jj) - imgdc(i/bs,j/bs);
					}
				}
				break;				
			case PREDICTOR_TOP_DC:
			default:
				uSnippets::Assert(false) << "Unsupported predictor";
		}

		std::vector<std::vector<uint32_t>> ret;
		
		if (orig_img.channels()==3 ) {
			
			cv::Mat3b pred3b = imgpredicted;
			
			if (colorPrediction) {
				for (auto &&p :pred3b) {
					uint8_t old = p[colorPrediction-1];
					p[0] -= old;
					p[1] -= old;
					p[2] -= old;
					p[colorPrediction-1] = old;
				}
			}
			for (size_t i=0; i<pred3b.rows-bs+1; i+=bs) {
				for (size_t j=0; j<pred3b.cols-bs+1; j+=bs) {
					std::vector<uint32_t> br(256,0), bg(256,0), bb(256,0);
					for (size_t ii=0; ii<bs; ii++) {
						for (size_t jj=0; jj<bs; jj++) {
							br[pred3b(i+ii,j+jj)[0]]++;
							bg[pred3b(i+ii,j+jj)[1]]++;
							bb[pred3b(i+ii,j+jj)[2]]++;
						}
					}
					ret.push_back(br);
					ret.push_back(bg);
					ret.push_back(bb);
				}
			}
		} else if (orig_img.channels()==1 ) {
			
			cv::Mat1b pred1b = imgpredicted;
			
			for (size_t i=0; i<pred1b.rows-bs+1; i+=bs) {
				for (size_t j=0; j<pred1b.cols-bs+1; j+=bs) {
					std::vector<uint32_t> h(256,0);
					for (size_t ii=0; ii<bs; ii++) {
						for (size_t jj=0; jj<bs; jj++) {
							h[pred1b(i+ii,j+jj)]++;
						}
					}
					ret.push_back(h);
				}
			}
		}
		
		return ret;
	}

	static std::vector<std::vector<uint32_t>> testPredictorHistograms(cv::Mat orig_img, size_t imageBlockWidth, PredictorType predictorType, int colorPrediction) {
		if (orig_img.channels()==3) return testPredictorHistograms<cv::Vec3b>(orig_img, imageBlockWidth, predictorType, colorPrediction);
		if (orig_img.channels()==1) return testPredictorHistograms<uint8_t>(orig_img, imageBlockWidth, predictorType, colorPrediction);
		uSnippets::Assert(false) << "Type not supported";		
	}
}
	
int main(int argc, char **argv) {
	
	static const int sampleSize = 256;
	
	uSnippets::Assert(argc>1) << "must specify test";
	
	std::vector<std::string> filenames;
	for (int i=2; i<argc; i++)
		for (auto &&f : getAllFilenames(argv[i], {"png", "jpg", "JPEG"}) )
			filenames.push_back(f);

	uSnippets::Assert(!filenames.empty()) << "must specify dataset paths";
			
	//for (auto &&f : filenames) 
	//	std::cout << f << std::endl;
	
	if (std::string(argv[1])=="analyzePredictors") {
		
		std::random_shuffle(filenames.begin(), filenames.end());
		if (filenames.size()>sampleSize) filenames.resize(sampleSize);
		
		uSnippets::Log(0) << "Reading Images";
		std::vector<cv::Mat> images;
		for (auto &f : filenames)
			images.push_back(cv::imread(f,cv::IMREAD_UNCHANGED));

		uSnippets::Log(0) << "Images Read";

		for (auto &&imageBlockWidth : std::vector<size_t>{0, 8,16,32}) {


//			for (auto &&predictorType : std::vector<predictors::PredictorType>{predictors::PREDICTOR_TOP, predictors::PREDICTOR_LEFT, predictors::PREDICTOR_ABC, predictors::PREDICTOR_DC, predictors::PREDICTOR_TOP_DC }) {
			for (auto &&predictorType : std::vector<predictors::PredictorType>{predictors::PREDICTOR_TOP, predictors::PREDICTOR_TOP_DC }) {

//				for (auto &&colorPrediction : std::vector<int>{0,1,2,3}) {
				for (auto &&colorPrediction : std::vector<int>{2}) {
					
					std::vector<std::vector<uint32_t>> histograms;
					std::vector<double> predictedBitsPerPixel;
					for (auto &i : images) {
						
						double predictedSize = 0.;
						for (auto &h : testPredictorHistograms(i.clone(), imageBlockWidth, predictorType, colorPrediction)) {
							histograms.push_back(h);
							predictedSize += predictors::predictCompressedBlockSize(h);
						}
						predictedBitsPerPixel.push_back(predictedSize/(i.rows*i.cols));
					}
					
					cv::Scalar mean, stddev;
					cv::meanStdDev(predictedBitsPerPixel, mean, stddev);
					std::cout << "BS: " << imageBlockWidth << " ";
					std::cout << "Pred: " << int(predictorType) << " ";
					std::cout << "Color: " << int(colorPrediction) << " ";					
					
					std::cout << "mean, stdev: " << mean[0] <<  " (+- " << stddev[0] << ") bits per pixel" << std::endl;

				}
			}
		}
	}

	return 0;
}
