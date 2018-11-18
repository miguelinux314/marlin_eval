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

#include <uSnippets/log.hpp>
//#include <uSnippets/mpng.hpp>


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#include <uSnippets/turbojpeg.hpp>
#pragma GCC diagnostic pop

#include <CharLS/interface.h>
#include <webp/encode.h>
#include <webp/decode.h>

#include <pstreams/pstream.h>

struct TestTimer {
	timespec c_start, c_end;
//	void start() { clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &c_start); };
//	void stop () { clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &c_end); };
	void start() { clock_gettime(CLOCK_MONOTONIC, &c_start); };
	void stop () { clock_gettime(CLOCK_MONOTONIC, &c_end); };
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
				entropy += -double(p)*std::log2(double(p/sum));

		return entropy + (sum>257?32:24);
	}
	
	uint8_t sub(uint8_t a, uint8_t b) { return a-b; }
	cv::Vec3b sub(cv::Vec3b a, cv::Vec3b b) { return cv::Vec3b(a[0]-b[0], a[1]-b[1], a[2]-b[2]); }

	double predictCompressedBlockSize(const std::vector<uint32_t> &histogram, const std::vector<double> &pdf) {
		

		double sum = 1e-100;
		for (auto &&p : histogram)
			sum += p;
		
		double entropy=0;
		for (size_t i=0; i<histogram.size(); i++) {
			if (pdf[i] and -std::log2(pdf[i])<16) {
				entropy += -double(histogram[i])*std::log2(pdf[i]);
			} else {
				entropy += double(histogram[i])*(sum>257?24:16);
			}
		}

		return entropy + (sum>257?32:24);
	}
	
	
	enum PredictorType { PREDICTOR_TOP, PREDICTOR_LEFT, PREDICTOR_ABC, PREDICTOR_DC, PREDICTOR_TOP_DC } ;




	template<typename T>
	static std::vector<std::vector<uint32_t>> testPredictorHistograms(cv::Mat_<T> orig_img, size_t imageBlockWidth, PredictorType predictorType, int colorPrediction) {

		orig_img = orig_img.clone();
		
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
			
			auto p1 = testPredictorHistograms(orig_img.clone(), imageBlockWidth, PREDICTOR_TOP, colorPrediction);
			auto p2 = testPredictorHistograms(orig_img.clone(), imageBlockWidth, PREDICTOR_DC, colorPrediction);
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
		cv::resize(img, imgdc, cv::Size(bcols,brows),0,0,cv::INTER_AREA);
//		uSnippets::Log(0) << imgdc.rows << " " << img.rows/bs << " " << imgdc.cols << " " <<  img.cols/bs;
		
		cv::Mat_<T> imgpredicted = img.clone();

		switch (predictorType) {
			case PREDICTOR_TOP:
				for (size_t i=0; i<img.rows-bs+1; i+=bs) {
					for (size_t j=0; j<img.cols-bs+1; j+=bs) {
						imgpredicted(i,j) = sub(imgpredicted(i,j), imgpredicted(i,j));
						for (size_t jj=1; jj<bs; jj++)
							imgpredicted(i,j+jj) = sub(img(i,j+jj), img(i,j+jj-1));

						for (size_t ii=1; ii<bs; ii++)
							for (size_t jj=0; jj<bs; jj++)
								imgpredicted(i+ii,j+jj) = sub(img(i+ii,j+jj), img(i+ii-1,j+jj));
					}
				}
				break;
			case PREDICTOR_LEFT:

				for (size_t i=0; i<img.rows-bs+1; i+=bs) {
					for (size_t j=0; j<img.cols-bs+1; j+=bs) {
						imgpredicted(i,j) = sub(imgpredicted(i,j), imgpredicted(i,j));
						for (size_t ii=1; ii<bs; ii++)
							imgpredicted(i+ii,j) = sub(img(i+ii,j), img(i+ii-1,j));

						for (size_t ii=0; ii<bs; ii++)
							for (size_t jj=1; jj<bs; jj++)
								imgpredicted(i+ii,j+jj) = sub(img(i+ii,j+jj), img(i+ii,j+jj-1));
					}
				}
				break;
			case PREDICTOR_ABC:
				for (size_t i=0; i<img.rows-bs+1; i+=bs) {
					for (size_t j=0; j<img.cols-bs+1; j+=bs) {
						imgpredicted(i,j) = sub(imgpredicted(i,j), imgpredicted(i,j));
						for (size_t jj=1; jj<bs; jj++)
							imgpredicted(i,j+jj) = sub(img(i,j+jj), img(i,j+jj-1));

						for (size_t ii=1; ii<bs; ii++)
							imgpredicted(i+ii,j) = sub(img(i+ii,j), img(i+ii-1,j));

						for (size_t ii=1; ii<bs; ii++)
							for (size_t jj=1; jj<bs; jj++) // WORNG
								imgpredicted(i+ii,j+jj) = img(i+ii,j+jj) - img(i+ii-1,j+jj) - img(i+ii,j+jj-1) + img(i+ii-1,j+jj-1);
					}
				}
				break;
			case PREDICTOR_DC:
				for (size_t i=0; i<img.rows-bs+1; i+=bs) {
					for (size_t j=0; j<img.cols-bs+1; j+=bs) {
						for (size_t ii=0; ii<bs; ii++)
							for (size_t jj=0; jj<bs; jj++)
								imgpredicted(i+ii,j+jj) = sub(img(i+ii,j+jj), imgdc(i/bs,j/bs));
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
			
			if (colorPrediction && predictorType!=PREDICTOR_TOP_DC) {
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

	cv::Mat1b Color2Planar(cv::Mat img, int colorPrediction) {
		
		if (img.channels()==1) return img;
		
		cv::Mat3b img3b = img;
		cv::Mat1b img1b(img3b.rows*3,img3b.cols);
		
		uint8_t *p0 = &img1b(img3b.rows*0,0);
		uint8_t *p1 = &img1b(img3b.rows*1,0);
		uint8_t *p2 = &img1b(img3b.rows*2,0);
		
		for (int i=0; i<img3b.rows; i++) {
			for (int j=0; j<img3b.cols; j++) {
				*p0++ = img3b(i,j)[0];
				*p1++ = img3b(i,j)[1];
				*p2++ = img3b(i,j)[2];
			}
		}
		
		if (colorPrediction) {
			for (int k=0; k<3; k++) {
				if (k==colorPrediction-1) continue;

				uint8_t *p0 = &img1b(img3b.rows*k,0);
				uint8_t *p1 = &img1b(img3b.rows*(colorPrediction-1),0);
				
				for (int i=0; i<img3b.rows; i++) {
					for (int j=0; j<img3b.cols; j++) {
						*p0++ -= *p1++;
					}
				}
			}
		}
		return img1b;
	}

	cv::Mat3b Planar2Color(cv::Mat1b img1b, int colorPrediction) {

		cv::Mat3b img3b(img1b.rows/3,img1b.cols);

		if (colorPrediction) {
			for (int k=0; k<3; k++) {
				if (k==colorPrediction-1) continue;

				uint8_t *p0 = &img1b(img3b.rows*k,0);
				uint8_t *p1 = &img1b(img3b.rows*(colorPrediction-1),0);
				
				for (int i=0; i<img3b.rows; i++) {
					for (int j=0; j<img3b.cols; j++) {
						*p0++ += *p1++;
					}
				}
			}
		}
		
		uint8_t *p0 = &img1b(img3b.rows*0,0);
		uint8_t *p1 = &img1b(img3b.rows*1,0);
		uint8_t *p2 = &img1b(img3b.rows*2,0);
		
		for (int i=0; i<img3b.rows; i++) {
			for (int j=0; j<img3b.cols; j++) {
				img3b(i,j)[0] = *p0++;
				img3b(i,j)[1] = *p1++;
				img3b(i,j)[2] = *p2++;
			}
		}
		
		return img1b;
	}


	cv::Mat1b Planar2Predicted(cv::Mat1b img, PredictorType predictorType) {
		
		cv::Mat1b predicted;
		
		if (predictorType == PREDICTOR_TOP) {
			
			predicted = img.clone();
			for (int j=1; j<img.cols; j++)
				predicted(0,j) = img(0,j)-img(0,j-1);
				
			for (int i=1; i<img.rows; i++)
				for (int j=0; j<img.cols; j++)
					predicted(i,j) = img(i,j)-img(i-1,j);

		} else if (predictorType == PREDICTOR_LEFT) {
			
			predicted = Planar2Predicted(img.t(), PREDICTOR_TOP).t();
			
		} else if (predictorType == PREDICTOR_ABC) {
			
			predicted = Planar2Predicted(Planar2Predicted(img,PREDICTOR_TOP).t(), PREDICTOR_TOP).t();
		} else throw std::invalid_argument("Unknown Predictor");


		return predicted;
	}

	cv::Mat1b Predicted2Planar(cv::Mat1b img, PredictorType predictorType) {
		
		if (predictorType == PREDICTOR_TOP) {
			
			uint8_t *p = &img(0,0);
			for (int j=1; j<img.cols; j++) {
				uint8_t v=*p;
				p += 1;
				*p += v;
			}
				
			uint8_t *p0 = &img(0,0);
			uint8_t *p1 = &img(1,0);
			bool aligned =
				(reinterpret_cast<std::uintptr_t>(p0) % 16 == 0 ) and
				(reinterpret_cast<std::uintptr_t>(p1) % 16 == 0 ) and
				img.cols % 16 == 0;
			
			for (int i=1; i<img.rows; i++) {

				int j=0;
				if (aligned)
				for (; j<img.cols-15; j+=16) {
					*(__m128i*)p1 = _mm_add_epi8(*(__m128i*)p1, *(__m128i*)p0); 
					p1 += 16;
					p0 += 16;
				}

				for (; j<img.cols-7; j+=8) {
					*(__m64*)p1 = _mm_add_pi8(*(__m64*)p1, *(__m64*)p0); 
					p1 += 8;
					p0 += 8;
				}
				
				for (; j<img.cols; j++)
					*p1++ += *p0++;
			}

		} else if (predictorType == PREDICTOR_LEFT) {
			
			uint8_t *p = &img(0,0);
			for (int i=1; i<img.rows; i++) {
				uint8_t v=*p;
				p += img.cols;
				*p += v;
			}
				
			for (int i=0; i<img.rows; i++) {
				uint8_t *p0 = &img(i,0);
				uint8_t *p1 = &img(i,1);

				int j=1;
				for (; j<img.cols-7; j+=8) {
					*p1++ += *p0++;
					*p1++ += *p0++;
					*p1++ += *p0++;
					*p1++ += *p0++;

					*p1++ += *p0++;
					*p1++ += *p0++;
					*p1++ += *p0++;
					*p1++ += *p0++;
				}
				
				for (; j<img.cols; j++)
					*p1++ += *p0++;
			}
			
		} else if (predictorType == PREDICTOR_ABC) {

			img = Predicted2Planar(img, PREDICTOR_TOP);
			img = Predicted2Planar(img, PREDICTOR_LEFT);			
		} else throw std::invalid_argument("Unknown Predictor");

		return img;
	}


	std::vector<std::vector<uint8_t>> Predicted2Blocks(cv::Mat1b img, int blockSize) {
		
		std::vector<std::vector<uint8_t>> blocks;
		for (int i=0; i<img.rows; i+=blockSize) {
			for (int j=0; j<img.cols; j+=blockSize) {
				std::vector<uint8_t> block;
				for (int ii=0; ii<blockSize and i+ii<img.rows; ii++) 
					for (int jj=0; jj<blockSize and j+jj<img.cols; jj++) 
						block.push_back(img(i+ii,j+jj));
						
				blocks.push_back(block);
			}
		}
		return blocks;
	}

	std::vector<std::vector<uint32_t>> Blocks2Histograms(const std::vector<std::vector<uint8_t>> &blocks) {

		std::vector<std::vector<uint32_t>> ret;
		for (auto &&block : blocks) {
		
			std::vector<uint32_t> h(256,0);
			for (auto &&v:block)
				h[v]++;
			ret.push_back(h);
		}
		return ret;
	}

	std::vector<std::vector<uint32_t>> testPredictorHistograms(cv::Mat img, size_t blockSize, PredictorType predictorType, int colorPrediction) {
		
		cv::Mat1b img1b = Color2Planar(img,colorPrediction);
		return Blocks2Histograms(Predicted2Blocks(Planar2Predicted(img1b,predictorType),blockSize));
	}

}

namespace compressors {

	cv::Mat1b Predicted2Stripped(cv::Mat1b img, int blockSize) {
		
		size_t brows = (img.rows+blockSize-1)/blockSize;
		size_t bcols = (img.cols+blockSize-1)/blockSize;	
		cv::copyMakeBorder(img, img, 0, brows*blockSize-img.rows, 0, bcols*blockSize-img.cols, cv::BORDER_CONSTANT, 0);
		
		
		cv::Mat1b stripped(img.rows, img.cols);
		
		uint8_t *in  = (uint8_t *)&img(0,0);
		for (int i=0; i<img.rows; i+=blockSize) {
			for (int ii=0; ii<blockSize; ii++) {
				uint8_t *out = (uint8_t *)&stripped(i,0);
				out += ii*blockSize;
				for (int j=0; j<img.cols; j+=blockSize) {
					memcpy(out, in, blockSize);
					out += blockSize*blockSize;
					in += blockSize;
				}
			}
		}
		
		return stripped;
	}

	template<typename T>
	cv::Mat1b Stripped2Predicted(cv::Mat1b stripped) {

		cv::Mat1b predicted(stripped.rows, stripped.cols);
		size_t skip = predicted.cols/sizeof(T);

		T *in = (T *)&stripped(0,0);
		for (int i=0; i<predicted.rows; i+=sizeof(T)) {
			for (int j=0; j<predicted.cols; j+=sizeof(T)) {
				T *out  = (T *)&predicted(i,j);
				for (uint ii=0; ii<sizeof(T)/8; ii++) {
					*out = *in++; out += skip;
					*out = *in++; out += skip;
					*out = *in++; out += skip;
					*out = *in++; out += skip;

					*out = *in++; out += skip;
					*out = *in++; out += skip;
					*out = *in++; out += skip;
					*out = *in++; out += skip;
				}
			}
		}
		return predicted;
	}
	
	cv::Mat1b Stripped2Predicted(cv::Mat1b stripped, int blockSize) {
		
		if (blockSize==8) return Stripped2Predicted<uint64_t>(stripped);
		if (blockSize==16) return Stripped2Predicted<__m128i>(stripped);
		if (blockSize==32) return Stripped2Predicted<std::pair<__m128i,__m128i>>(stripped);
		
		cv::Mat1b predicted(stripped.rows, stripped.cols);
		
		uint8_t *out  = (uint8_t *)&predicted(0,0);
		for (int i=0; i<predicted.rows; i+=blockSize) {
			for (int ii=0; ii<blockSize; ii++) {
				uint8_t *in = (uint8_t *)&stripped(i,0);
				in += ii*blockSize;
				for (int j=0; j<predicted.cols; j+=blockSize) {
					memcpy(out, in, blockSize);
					in += blockSize*blockSize;
					out += blockSize;
				}
			}
		}
		
		return predicted;
	}

	struct ICodec {
		
		virtual std::vector<uint8_t> encode(const cv::Mat &) = 0;
		virtual cv::Mat decode(const std::vector<uint8_t> &, const cv::Mat &) = 0;
		virtual std::string name() = 0;
	};

	struct OpenCVCodec : public ICodec {
		
		std::string ext;
		std::vector<int> flags;
		
		OpenCVCodec(std::string ext, std::vector<int> flags =std::vector<int>() ) : ext(ext), flags(flags) {}

		
		virtual std::vector<uint8_t> encode(const cv::Mat &img_) { 

			cv::Mat img = img_;
			std::vector<uint8_t> buf;
			cv::imencode(ext, img, buf, flags);
			return buf;
		}
		
		virtual cv::Mat decode(const std::vector<uint8_t> &buf, const cv::Mat &) {
			return cv::imdecode(buf,cv::IMREAD_COLOR);
		}

		virtual std::string name() { return ext; }
	};
	
	struct EntropyCodec : public ICodec {

		std::shared_ptr<CODEC8> codec;
		
		EntropyCodec(std::shared_ptr<CODEC8> codec) : codec(codec) {}
		
		virtual std::vector<uint8_t> encode(const cv::Mat &img) { 
			
			cv::Mat1b img1b = predictors::Color2Planar(img.clone(), 2);
			cv::Mat1b predicted = predictors::Planar2Predicted(img1b.clone(), predictors::PREDICTOR_ABC);
			cv::Mat1b stripped = compressors::Predicted2Stripped(predicted, 32);

			UncompressedData8 in(stripped.data, stripped.rows*stripped.cols);
			CompressedData8 compressed;
				
			codec->compress(in, compressed);
			
			return std::vector<uint8_t>(compressed);
		}
		
		virtual cv::Mat decode(const std::vector<uint8_t> &buf, const cv::Mat &in) {

			
			CompressedData8 compressed(buf);
			UncompressedData8 uncompressed;
			uncompressed.resize(in.rows*in.cols*4);

			codec->uncompress(compressed, uncompressed);

			return cv::Mat();
		}
		
		virtual std::string name() { return codec->name(); }

	};

	struct TurboJPEG : public ICodec {
		
		virtual std::vector<uint8_t> encode(const cv::Mat &img) { 
			
			std::string str;
			if (img.channels()==1) {
				str = uSnippets::TJ::code(cv::Mat1b(img),100, TJSAMP_444, TJFLAG_FASTUPSAMPLE |TJFLAG_ACCURATEDCT);
			} else {
				str = uSnippets::TJ::code(cv::Mat3b(img),100, TJSAMP_444, TJFLAG_FASTUPSAMPLE |TJFLAG_ACCURATEDCT);
			}
			return std::vector<uint8_t>(str.begin(), str.end());
		}
		
		virtual cv::Mat decode(const std::vector<uint8_t> &buf, const cv::Mat &) {

			return uSnippets::TJ::decode<cv::Vec3b>(std::string(buf.begin(), buf.end()), TJFLAG_FASTUPSAMPLE |TJFLAG_ACCURATEDCT);
		}
		
		virtual std::string name() { return "TurboJPEG"; }
	};


	struct CharLS : public ICodec {
		
		virtual std::vector<uint8_t> encode(const cv::Mat &img_) { 

			cv::Mat1b img = predictors::Color2Planar(img_.clone(), 2);

			JlsParameters info = JlsParameters();
			info.components = img.channels();
			info.bitspersample = 8;
			info.bytesperline = img.cols*img.channels();
			info.width = img.cols;
			info.height = img.rows;


			std::vector<uint8_t> ret(img.rows*img.cols*img.channels()*2);
			size_t compressedLength;
			JpegLsEncode(&ret[0], ret.capacity(), &compressedLength, img.data, img.rows*img.cols*img.channels(), &info);

			ret.resize(compressedLength);
			return ret;
		}
		
		virtual cv::Mat decode(const std::vector<uint8_t> &buf, const cv::Mat &in) {
			
			cv::Mat out = in.clone();
			JpegLsDecode(out.data, out.rows*out.cols*out.channels(), &buf[0], buf.size(), nullptr);
			return out;
			//return in;
		}
		
		virtual std::string name() { return "CharLS"; }
	};

	struct Memcpy : public ICodec {
		
		virtual std::vector<uint8_t> encode(const cv::Mat &img_) { 

			std::vector<uint8_t> ret(img_.rows*img_.cols*img_.channels());
			memcpy(&ret[0], img_.data, ret.size());
			return ret;
		}
		
		virtual cv::Mat decode(const std::vector<uint8_t> &buf, const cv::Mat &in) {

			cv::Mat out = in.clone();
			memcpy(out.data, &buf[0], buf.size());
			return out;
		}
		
		virtual std::string name() { return "Memcpy"; }
	};


	struct WebP : public ICodec {
		
		virtual std::vector<uint8_t> encode(const cv::Mat &img_) { 

			cv::Mat img = img_;
			if (img.channels()==1) cv::cvtColor(img.clone(),img,cv::COLOR_GRAY2BGR);

			uint8_t *out;
			size_t sz = WebPEncodeLosslessBGR(img.data, img.cols, img.rows, img.cols*3, &out);

			std::vector<uint8_t> ret(out,out+sz);
			free(out);
			
			return ret;
		}
		
		virtual cv::Mat decode(const std::vector<uint8_t> &buf, const cv::Mat &in) {
			
			int width, height;
			uint8_t * out = WebPDecodeBGR(&buf[0], buf.size(), &width, &height);
			cv::Mat ret = cv::Mat3b(in.rows,in.cols,(cv::Vec3b *)out).clone();
			free(out);
			
			if (in.channels()==1) cv::cvtColor(ret,ret,cv::COLOR_BGR2GRAY);

			return ret;
		}
		
		virtual std::string name() { return "WebP"; }
	};


	struct CommandLine : public ICodec {
		
		std::string cEncode, cDecode;
		
		CommandLine(std::string cEncode, std::string cDecode) : cEncode(cEncode), cDecode(cDecode) {}
		
		
		virtual std::vector<uint8_t> encode(const cv::Mat &img) { 
			
			cv::imwrite("/tmp/in.bmp", img);
			
			int systemret = system(cEncode.c_str());
			if (systemret) {}
			
			
			std::ifstream iss("/tmp/out.file");
			
			iss.seekg(0, std::ios::end);   
			size_t sz = iss.tellg();
			iss.seekg(0, std::ios::beg);
			
			std::vector<uint8_t> ret(sz);
			iss.read((char *)&ret[0],sz);
			return ret;
		}
		
		virtual cv::Mat decode(const std::vector<uint8_t> &, const cv::Mat &) {
			
			int systemret = system(cDecode.c_str());
			if (systemret) {}

			return cv::Mat();
		}
		
		virtual std::string name() { return cEncode; }
	};
}

std::vector<cv::Mat> loadImages(int argc, char **argv,size_t sampleSize) {

	std::vector<std::string> filenames;
	for (int i=2; i<argc; i++)
		for (auto &&f : getAllFilenames(argv[i], {"png", "jpg", "JPEG"}) )
			filenames.push_back(f);

	uSnippets::Assert(!filenames.empty()) << "must specify dataset paths";				
		
	std::random_shuffle(filenames.begin(), filenames.end());
	if (filenames.size()>sampleSize) filenames.resize(sampleSize);
	
	uSnippets::Log(0) << "Reading Images";
	std::vector<cv::Mat> images;
	for (auto &f : filenames) {
		
		cv::Mat img = cv::imread(f,cv::IMREAD_UNCHANGED);
		uSnippets::Assert(img.rows>=32) << "Img.rows " << img.rows;
		uSnippets::Assert(img.cols>=32) << "Img.cols " << img.rows;
		
		img = img(cv::Rect(0,0,img.cols & 0xFFE0,img.rows & 0xFFE0)).clone();
		
		images.push_back(img);
	}

	uSnippets::Log(0) << "Images Read";	
	
	return images;
}


static std::vector<std::shared_ptr<compressors::ICodec>> getCodecs() {

	std::map<std::string, double> baseConf;
	baseConf["O"] = 2;
	baseConf["K"] = 8;	
	baseConf.emplace("iterations",2);
	baseConf.emplace("numDict",8);
	baseConf.emplace("autoMaxWordSize",7);		

	return std::vector<std::shared_ptr<compressors::ICodec>>{
		std::make_shared<compressors::EntropyCodec>(std::make_shared<Rice>()),
		std::make_shared<compressors::EntropyCodec>(std::make_shared<RLE>()),
//		std::make_shared<compressors::EntropyCodec>(std::make_shared<Snappy>()),
		std::make_shared<compressors::EntropyCodec>(std::make_shared<Nibble>()),
		std::make_shared<compressors::EntropyCodec>(std::make_shared<Lz4>()),
		std::make_shared<compressors::EntropyCodec>(std::make_shared<FiniteStateEntropy>()),
		std::make_shared<compressors::EntropyCodec>(std::make_shared<Gipfeli>()),
//			std::make_shared<compressors::EntropyCodec>(std::make_shared<Gzip>()),
		std::make_shared<compressors::EntropyCodec>(std::make_shared<Lzo>()),
		std::make_shared<compressors::EntropyCodec>(std::make_shared<Huff0>()),
		std::make_shared<compressors::EntropyCodec>(std::make_shared<Zstd>()),
		std::make_shared<compressors::EntropyCodec>(std::make_shared<Marlin2019>(Distribution::Laplace,baseConf)),
//		std::make_shared<compressors::EntropyCodec>(std::make_shared<CharLS>()),
		
//		std::make_shared<compressors::OpenCVCodec>(".jp2",std::vector<int>{}),
//		std::make_shared<compressors::OpenCVCodec>(".png",std::vector<int>{CV_IMWRITE_PNG_COMPRESSION,1,CV_IMWRITE_PNG_STRATEGY, CV_IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY}),
//		std::make_shared<compressors::OpenCVCodec>(".png",std::vector<int>{CV_IMWRITE_PNG_COMPRESSION,1,CV_IMWRITE_PNG_STRATEGY, CV_IMWRITE_PNG_STRATEGY_RLE}),
//		std::make_shared<compressors::OpenCVCodec>(".png",std::vector<int>{CV_IMWRITE_PNG_COMPRESSION,9,CV_IMWRITE_PNG_STRATEGY, CV_IMWRITE_PNG_STRATEGY_DEFAULT}),
//		std::make_shared<compressors::OpenCVCodec>(".png",std::vector<int>{CV_IMWRITE_PNG_COMPRESSION,9,CV_IMWRITE_PNG_STRATEGY, CV_IMWRITE_PNG_STRATEGY_FILTERED}),
//		std::make_shared<compressors::OpenCVCodec>(".jpg",std::vector<int>{CV_IMWRITE_JPEG_QUALITY,100}),
		std::make_shared<compressors::TurboJPEG>(),
		std::make_shared<compressors::CharLS>(),
//		std::make_shared<compressors::WebP>(),
		std::make_shared<compressors::Memcpy>(),
		
//		std::make_shared<compressors::CommandLine>("cp /tmp/in.bmp /tmp/out.file", "cp /tmp/out.file /tmp/out.bmp"),
//		std::make_shared<compressors::CommandLine>("./izc.sh", "./izd.sh"),
//		std::make_shared<compressors::CommandLine>("./flifc.sh", "./flifd.sh"),
	};
}
	
int main(int argc, char **argv) {
	
	uSnippets::Assert(argc>1) << "must specify test";
	
	std::vector<cv::Mat> images = loadImages(argc, argv, 16);
				
	if (std::string(argv[1])=="analyzePredictors") {
		
		for (auto &&imageBlockWidth : std::vector<size_t>{8,16,32}) {


//			for (auto &&predictorType : std::vector<predictors::PredictorType>{predictors::PREDICTOR_TOP, predictors::PREDICTOR_LEFT, predictors::PREDICTOR_ABC, predictors::PREDICTOR_DC, predictors::PREDICTOR_TOP_DC }) {
			for (auto &&predictorType : std::vector<predictors::PredictorType>{predictors::PREDICTOR_TOP, predictors::PREDICTOR_LEFT, predictors::PREDICTOR_ABC}) {

//				for (auto &&colorPrediction : std::vector<int>{0,1,2,3}) {
				for (auto &&colorPrediction : std::vector<int>{0,1,2,3}) {
					
					
					double aggregateTime = 0;
					double aggregatePixels = 0;
					
					std::vector<std::vector<uint32_t>> histograms;
					std::vector<double> predictedBitsPerPixel;
					for (auto &i : images) {
						
						cv::Mat1b img1b = predictors::Color2Planar(i.clone(), colorPrediction);
						cv::Mat1b predicted = predictors::Planar2Predicted(img1b.clone(), predictorType);
						cv::Mat1b stripped = compressors::Predicted2Stripped(predicted, imageBlockWidth);
						TestTimer tt; tt.start();
						cv::Mat1b unstripped = compressors::Stripped2Predicted(stripped, imageBlockWidth);
						tt.stop();
						
						cv::Mat1b recovered = predictors::Predicted2Planar(unstripped, predictorType);
						aggregateTime += tt();
						aggregatePixels += recovered.rows*recovered.cols;
						if (cv::countNonZero(recovered(cv::Rect(0,0,img1b.cols,img1b.rows)) != img1b))
							uSnippets::Log(0) << "Ay ay ay...";
						
						
						
						
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
					
					std::cout << "Predicting speed: " << (aggregatePixels/aggregateTime)/(1<<20) << "MiB/s" << std::endl;

				}
			}
		}
	} else if (std::string(argv[1])=="buildCustomDictionary") {
		
		for (auto &&numDict : std::vector<size_t>{8, 32}) {

			for (auto &&customDict : std::vector<size_t>{0, 8, 16, 32}) {

				std::vector<std::vector<double>> pdfs;
				
				for (auto &&type : std::vector<Distribution::Type>{ Distribution::Gaussian, Distribution::Laplace, Distribution::Exponential, Distribution::Poisson }) {
					
					for (size_t p=0; p<numDict; p++) {
					
						std::vector<double> pdf(256,0.);
						
						size_t nSamples = 10;
						for (double i=0.5/nSamples; i<0.9999999; i+=1./nSamples) {
							auto pdf0 = Distribution::pdfByEntropy(type, (p+i)/double(numDict));
							for (size_t j=0; j<pdf.size(); j++)
								pdf[j] += pdf0[j]/nSamples;
						}
						pdfs.push_back(pdf);
					}
				}


		//		for (auto &&imageBlockWidth : std::vector<size_t>{0, 8,16,32}) {
				for (auto &&imageBlockWidth : std::vector<size_t>{8}) {


		//			for (auto &&predictorType : std::vector<predictors::PredictorType>{predictors::PREDICTOR_TOP, predictors::PREDICTOR_LEFT, predictors::PREDICTOR_ABC, predictors::PREDICTOR_DC, predictors::PREDICTOR_TOP_DC }) {
					for (auto &&predictorType : std::vector<predictors::PredictorType>{predictors::PREDICTOR_TOP }) {

		//				for (auto &&colorPrediction : std::vector<int>{0,1,2,3}) {
						for (auto &&colorPrediction : std::vector<int>{2}) {
							
							std::vector<std::vector<double>> customPdfs;

							if (1) {
								for (auto &i : images) {
									for (auto &h : testPredictorHistograms(i.clone(), imageBlockWidth, predictorType, colorPrediction)) {
										double aim = predictors::predictCompressedBlockSize(h);

										double best = 1e100;
										std::vector<double> bestPdf;
										for (auto &&pdf : pdfs) {
											double sz = predictors::predictCompressedBlockSize(h,pdf);
											if (sz<best) best = sz;
											bestPdf = pdf;

										}
										for (auto &&pdf : customPdfs) {
											double sz = predictors::predictCompressedBlockSize(h,pdf);
											if (sz<best) best = sz;
											bestPdf = pdf;
										}
										if (aim*1.40<best) {
											double sum = 0;
											for (auto p : h) sum += p;
											std::vector<double> pdf;
											for (auto p : h) pdf.push_back(double(p)/sum);
											customPdfs.push_back(pdf);
											uSnippets::Log(1) << int(double(aim/best)*1000)/10. <<  ": " << aim << " " << best << " " << customPdfs.size();
											
											//for (int i=0; i<256; i++)
											//	if (h[i] or int(bestPdf[i]*256))
											//		uSnippets::Log(0) << i << ": " << h[i] << " " << int(bestPdf[i]*256);
									
											//if (aim/best<0.4) exit(-1);
											
											

										}
									}
								}
							}
							
							{
							
								std::vector<std::vector<uint32_t>> histograms;
								std::vector<double> bestBitsPerPixel;
								std::vector<double> predictedBitsPerPixel;
								for (auto &i : images) {
									
									double bestSize = 0.;
									double predictedSize = 0.;
									for (auto &h : testPredictorHistograms(i.clone(), imageBlockWidth, predictorType, colorPrediction)) {
										histograms.push_back(h);
										
										double best = 1e100;
										std::vector<double> bestPdf;
										for (auto &&pdf : pdfs) {
											double sz = predictors::predictCompressedBlockSize(h,pdf);
											if (sz<best) {
												best = sz;
												bestPdf = pdf;
											}
										}
										for (auto &&pdf : customPdfs) {
											double sz = predictors::predictCompressedBlockSize(h,pdf);
											if (sz<best) {
												best = sz;
												bestPdf = pdf;
											}
										}
										predictedSize += best;
										
										double aim = predictors::predictCompressedBlockSize(h);
										
										bestSize += aim;
										
										if (false and aim > 40 and aim*3<best) {
											std::cout << aim << std::endl;
											std::cout << best << std::endl;
											for (int i=0; i<256; i++)
												if (h[i] or int(bestPdf[i]*256))
													uSnippets::Log(0) << i << ": " << h[i] << " " << int(bestPdf[i]*256);
									
											//exit(-1);
										}
									}
									bestBitsPerPixel.push_back(bestSize/(i.rows*i.cols));
									predictedBitsPerPixel.push_back(predictedSize/(i.rows*i.cols));
								}

								cv::Scalar bestMean, bestStddev;
								cv::meanStdDev(bestBitsPerPixel, bestMean, bestStddev);
								
								cv::Scalar predictedMean, predictedStddev;
								cv::meanStdDev(predictedBitsPerPixel, predictedMean, predictedStddev);


								std::cout << "numDict: " << numDict << " ";
								std::cout << "BS: " << imageBlockWidth << " ";
								std::cout << "Pred: " << int(predictorType) << " ";
								std::cout << "Color: " << int(colorPrediction) << " ";					
								
								std::cout << "best: " << bestMean[0] <<  " (+- " << bestStddev[0] << ") ";
								std::cout << "predicted: " << predictedMean[0] <<  " (+- " << predictedStddev[0] << ") bits per pixel" << std::endl;
							}
						}
					}
				}
			}
		}
	} else if (std::string(argv[1])=="testCompressionThoroughput") {

		for (auto codec : getCodecs()) {

			std::vector<double> compressSpeed, uncompressSpeed, compressionRate;

			for (auto &img : images) {
				
				std::vector<uint8_t> buf;
				
				TestTimer compressTimer, uncompressTimer;
				size_t nComp = 1, nUncomp = 1;
				do {
					nComp *= 2;
					buf = codec->encode(img);
					compressTimer.start();
					for (size_t t=0; t<nComp; t++)
						buf = codec->encode(img);
					compressTimer.stop();
				} while (compressTimer()<.01);

				do {
					nUncomp *= 2;
					codec->decode(buf,img);
					uncompressTimer.start();
					for (size_t t=0; t<nUncomp; t++)
						codec->decode(buf,img);
					uncompressTimer.stop();
				} while (uncompressTimer()<.01);

				compressSpeed.push_back(nComp*img.rows*img.cols*img.channels()/  compressTimer());
				uncompressSpeed.push_back(nUncomp*img.rows*img.cols*img.channels()/  uncompressTimer());
				compressionRate.push_back(double(buf.size())/double(img.rows*img.cols*img.channels()));
				
				//uSnippets::Log(0) << nComp << " " << in.nBytes() << " " << compressTimer();
			}

			double meanCompressionRate=0, meanCompressSpeed=0, meanUncompressSpeed=0;
			for (auto &e : compressionRate) meanCompressionRate += e/compressionRate.size(); 
			for (auto &e : compressSpeed) meanCompressSpeed += e/compressSpeed.size(); 
			for (auto &e : uncompressSpeed) meanUncompressSpeed += e/uncompressSpeed.size(); 

			
			std::cout << "" << 1./meanCompressionRate << " " << (meanCompressSpeed/(1<<20)) << " " << codec->name() << " {west}" << std::endl;

			std::cout << "" << 1./meanCompressionRate << " " << (meanUncompressSpeed/(1<<20)) << " " << codec->name() << " {west}" << std::endl;

		}
	
	
	} else if (std::string(argv[1])=="testNetworkAndDisk") {
		
		
		
		struct TestType { std::string name; double bandwidth, seekTime; };

		for (auto &&nCores : std::vector<int>{ 1 }) {
		for (auto &&testType : std::vector<TestType>{ 
//			{"Rotational",200*double(1<<20),0.01}, 
//			{"SSD SATA-600",600*double(1<<20),0.0001}, 
			{"Net 1GB",100*double(1<<20),0.0001}, 
			}) {
		for (auto codec : getCodecs()) {

			typedef std::pair<std::vector<uint8_t>, cv::Mat> Msg;

			std::vector<std::shared_ptr<Msg>> compressedMessages;
			for (auto &img : images)
				compressedMessages.push_back( std::make_shared<Msg>( codec->encode(img), img));
				
			std::mutex mtx;
			typedef std::lock_guard<std::mutex> Lock;

			std::deque< std::shared_ptr<Msg>> processingQueue;
			
			
			uSnippets::Log(0) << "What!";
			size_t imagesDone = 0;
			bool done = false;
			
			std::thread producer([&](){

				auto ip = compressedMessages.begin();
				while (!done) {
					
					if (ip==compressedMessages.end()) ip = compressedMessages.begin();
					auto msgptr = *ip++;
					

					{
						Lock l(mtx);
						if (processingQueue.size()<100)
							processingQueue.push_back(msgptr);
					}
					
					double bandwidthTime = msgptr->first.size()/testType.bandwidth;
					
					//uSnippets::Log(0) << uint64_t(testType.seekTime*1e9) << " " << uint64_t(bandwidthTime*1e9);
					
					std::this_thread::sleep_for(
						std::chrono::nanoseconds(
							uint64_t(std::max(testType.seekTime,bandwidthTime)*1e9)));
				};
			});

			std::vector<std::thread> consumers;
			for (int i=0; i<nCores; i++) consumers.emplace_back([&](){
				
				while (!done) {
					
					std::shared_ptr<Msg> msgptr;
					{
						Lock l(mtx);
						if (not processingQueue.empty()) {
							msgptr = processingQueue.front();
							processingQueue.pop_front();
						}
					}
					if (not msgptr) {
						std::this_thread::sleep_for(std::chrono::milliseconds(1));
						continue;
					}
					
					codec->decode( msgptr->first, msgptr->second );
					imagesDone++;
					std::this_thread::yield();
				};	
			});
			
			std::this_thread::sleep_for(std::chrono::milliseconds(1000));
			done = true;
			producer.join();
			for (auto &&t : consumers) t.join();
			
			uSnippets::Log(1) << codec->name() << ": " << imagesDone << " Images/second";
			
		}
		}
		}
	} 
	
	return 0;
}
