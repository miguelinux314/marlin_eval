#include <dirent.h>
#include <fstream>
#include <map>
#include <queue>
#include <chrono>
#include <memory>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <marlinlib/marlin.hpp>
#include <util/distribution.hpp>
#include <codecs/rle.hpp>
#include <codecs/snappy.hpp>
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

struct TestTimer {
	timespec c_start, c_end;
	void start() { clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &c_start); };
	void stop () { clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &c_end); };
	double operator()() { return (c_end.tv_sec-c_start.tv_sec) + 1.E-9*(c_end.tv_nsec-c_start.tv_nsec); }
};

static inline std::vector<std::string> getAllFilenames(std::string path, std::string type="") {
	
	std::vector<std::string> r;
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir (path.c_str())) == NULL)
		return r;
	
	while ((ent = readdir (dir)) != NULL)
		if (std::string(ent->d_name).size()>=type.size() and std::string(&ent->d_name[std::string(ent->d_name).size()-type.size()])==type)
			r.push_back(path+"/"+ent->d_name);

	closedir (dir);
	std::sort(r.begin(), r.end());
	return r;
}

static inline cv::Mat1b readPGM8(std::string fileName) {
	
	std::ifstream in(fileName);
	
	std::string type; int rows, cols, values;
	in >> type >> cols >> rows >> values;
	in.get();
	cv::Mat1b img(rows, cols);
	in.read((char *)&img(0,0),rows*cols);
	return img;
}

static inline void testCorrectness(std::shared_ptr<CODEC8> codec) {
	
	std::cout << "Testing codec: " << codec->name() << " for correctness" << std::endl;
	
	for (double p=0.1; p<.995; p+=0.1) {
		
		UncompressedData8 in(Distribution::getResiduals(Distribution::pdf(Distribution::Laplace, p),1<<20));
		CompressedData8 compressed;
		UncompressedData8 uncompressed; 
		
		compressed.randomize();
		uncompressed.randomize();
		
		codec->compress(in, compressed);
		codec->uncompress(compressed, uncompressed);
		
		std::vector<uint8_t> inv(in), outv(uncompressed);
		
		if (inv != outv) {
			
			std::cout << "P: " << p << " " << "FAIL!     sizes(" << inv.size() << "," << outv.size() << ")" << std::endl;
			for (size_t i=0; i<10; i++)
				printf("%02X:%02X ", inv[i], outv[i]);
			std::cout << std::endl;
			
			{
				int c = 0;
				for (size_t i=0; i<inv.size(); i++) {
					if (inv[i] != outv[i]) {
						printf("Pos %04X = %02X:%02X\n", uint(i), inv[i], outv[i]);
						if (c++==4) break;
					}
				}
			}
		}
	}
}

static inline void testAgainstP( std::shared_ptr<CODEC8> codec, std::ofstream &tex, size_t testSize = 1<<18) {
	
	std::cout << "Testing codec: " << codec->name() << " against P" << std::endl;

	std::map<double, double> C, D, E;
		
	// Test compression (C) and uncompression (D) speeds
	for (double p=0.1; p<.995; p+=0.1) {

		UncompressedData8 in(Distribution::getResiduals(Distribution::pdf(Distribution::Laplace, p),testSize));
		CompressedData8 compressed;
		UncompressedData8 uncompressed;
		
		compressed.randomize();
		uncompressed.randomize();
		
		TestTimer compressTimer, uncompressTimer;
		size_t nComp = 5, nUncomp = 5;
		do {
			nComp *= 2;
			codec->compress(in, compressed);
			compressTimer.start();
			for (size_t t=0; t<nComp; t++)
				codec->compress(in, compressed);
			compressTimer.stop();
		} while (compressTimer()<.1);


		do {
			nUncomp *= 2;
			codec->uncompress(compressed, uncompressed);
			uncompressTimer.start();
			for (size_t t=0; t<nUncomp; t++)
				codec->uncompress(compressed, uncompressed);
			uncompressTimer.stop();
		} while (uncompressTimer()<.1);

		//std::cerr << "E: " << (nComp*in.nBytes()/  compressTimer())/(1<<20) << std::endl;
		//std::cerr << "D: " << nUncomp*in.nBytes()/uncompressTimer()/(1<<20) << std::endl;
				
		C[p] =   nComp*in.nBytes()/  compressTimer();
		D[p] = nUncomp*in.nBytes()/uncompressTimer();
	}
	
	{ double m=0; for (auto &v : C) m+=v.second; std::cout << "Mean Compression Speed:   " << m/C.size()/(1<<20) << "MB/s" << std::endl; }
	{ double m=0; for (auto &v : D) m+=v.second; std::cout << "Mean Decompression Speed: " << m/D.size()/(1<<20) << "MB/s" << std::endl; }
	
	// Test compression efficiency (E)
	for (double p=0.01; p<1.; p+=0.01) {
		
		UncompressedData8 in(Distribution::getResiduals(Distribution::pdf(Distribution::Laplace, p),1<<20));
		CompressedData8 compressed;
		codec->compress(in, compressed);
		
		E[p]=Distribution::entropy(Distribution::pdf(Distribution::Laplace, p))/(8.*double(compressed.nBytes())/(double(in.nBytes())+1e-100));
	}

	{ double m=0; for (auto &v : E) m+=v.second; std::cout << "Mean Efficiency: " << 100.*m/E.size() << "%" << std::endl; }

	// output tex graph
	if (tex) {
		tex << "\\compfig{" << codec->name() << "}{ " << std::endl;
		tex << "\\addplot coordinates {";
		for (auto &c : C) tex << "("<<c.first*100<<","<<c.second/(1<<30)<<") ";
		tex << "};" << std::endl;
		tex << "\\addplot coordinates {";
		for (auto &c : D) tex << "("<<c.first*100<<","<<c.second/(1<<30)<<") ";
		tex << "};" << std::endl;
		tex << "}{" << std::endl;
		tex << "\\addplot+[line width=2pt,teal, mark=none] coordinates {";
		for (auto &c : E) tex << "("<<c.first*100<<","<<c.second*100<<") ";
		tex << "};" << std::endl;
		tex << "}%" << std::endl;
	}
}

static inline std::pair<std::string,std::string> testOnIndividualFiles(
        std::shared_ptr<CODEC8> codec, std::string dir_path, std::ofstream &csv) {

	std::cout << "Testing codec: " << codec->name() << " against Images" << std::endl;

	std::map<std::string, double> compressSpeed, uncompressSpeed, compressionRate;

    std::cout << std::endl << std::endl;
    std::cout << "====" << dir_path << "X" << codec->name() << std::endl;
    for (auto file : getAllFilenames(dir_path, ".pgm")) {
        std::cout << "~~~~" << file << std::endl;

        cv::Mat1b img = readPGM8(file);
        img = img(cv::Rect(0, 0, img.cols & 0xFF80, img.rows & 0xFF80));
        if (img.cols * img.rows == 0) {
            std::cerr << "empty image " << file << std::endl;
            abort();
        }
        std::cout << file << ":" << img.cols << "," << img.rows << std::endl;

        UncompressedData8 in(img);
        CompressedData8 compressed;
        UncompressedData8 uncompressed;

        TestTimer compressTimer, uncompressTimer;
        double min_execution_seconds = 1;
        size_t nComp = 32, nUncomp = 32; // Minimum repetition count
        do {
            nComp *= 2;
            codec->compress(in, compressed);
            compressTimer.start();
            for (size_t t = 0; t < nComp; t++)
                codec->compress(in, compressed);
            compressTimer.stop();
        } while (compressTimer() < min_execution_seconds);

        do {
            nUncomp *= 2;
            codec->uncompress(compressed, uncompressed);
            uncompressTimer.start();
            for (size_t t = 0; t < nUncomp; t++)
                codec->uncompress(compressed, uncompressed);
            uncompressTimer.stop();
        } while (uncompressTimer() < min_execution_seconds);

        bool compressed_ok;
        if (cv::countNonZero(img != uncompressed.img(img.rows, img.cols)) != 0) {
            std::cerr << "Image uncompressed incorrectly" << std::endl;
            cv::Mat diff = img - uncompressed.img(img.rows, img.cols);
            compressed_ok = false;
        } else {
            compressed_ok = true;
        }

        compressSpeed[file] = nComp * in.nBytes() / compressTimer();
        uncompressSpeed[file] = nUncomp * in.nBytes() / uncompressTimer();
        compressionRate[file] = double(compressed.nBytes()) / double(in.nBytes());

        double minVal;
        double maxVal;
        cv::Point minLoc;
        cv::Point maxLoc;
        minMaxLoc(img, &minVal, &maxVal, &minLoc, &maxLoc);
        csv << codec->name()
            << "," << dir_path
            << "," << file
            << "," << minVal
            << "," << maxVal
            << "," << img.total()
            << "," << compressed.nBytes()
            << "," << compressTimer() / nComp
            << "," << nComp
            << "," << uncompressTimer() / nUncomp
            << "," << nUncomp
            << "," << compressed_ok
            << std::endl;
    }

    if (compressionRate.size() * compressSpeed.size() * uncompressSpeed.size() == 0) {
        std::cerr << "dir_path " << dir_path << " without images?" << std::endl;
        abort();
    }

	double meanCompressionRate=0, meanCompressSpeed=0, meanUncompressSpeed=0;
	for (auto &e : compressionRate) meanCompressionRate += e.second/compressionRate.size(); 
	for (auto &e : compressSpeed) meanCompressSpeed += e.second/compressSpeed.size(); 
	for (auto &e : uncompressSpeed) meanUncompressSpeed += e.second/uncompressSpeed.size(); 

	std::ostringstream enc, dec;
	enc << "" << 1./meanCompressionRate << " " << (meanCompressSpeed/(1<<20)) << " " << codec->name() << " {west}" << std::endl;

	dec << "" << 1./meanCompressionRate << " " << (meanUncompressSpeed/(1<<20)) << " " << codec->name() << " {west}" << std::endl;

	return std::pair<std::string,std::string>(enc.str(), dec.str());
}
using namespace std;
	
int main( int , char *[] ) {

	std::vector<shared_ptr<CODEC8>> C = {
        std::make_shared<Rice>(),
		std::make_shared<RLE>(),
		std::make_shared<Snappy>(),
		std::make_shared<FiniteStateEntropy>(),
		std::make_shared<Gipfeli>(),
		std::make_shared<Lzo>(),
		std::make_shared<Huff0>(),
		std::make_shared<Lz4>(),
		std::make_shared<Zstd>(),
	};

	Distribution::Type distType = Distribution::Laplace;
	for (int K=8; K<=10; K++) {
        for (int O = 0; O <= 4; O++) {
            if ((O != 0 && O != 2)
                || (K != 8 && K != 10)) {
                continue;
            }
            std::map<std::string, double> conf;
            conf["O"] = O;
            conf["K"] = K;
            conf.emplace("minMarlinSymbols", 2);
//            conf.emplace("maxWordSize", 64);
//            conf.emplace("autoMaxWordSize",8);
            conf.emplace("purgeProbabilityThreshold", 0.5 / 4096 / 256);

            C.push_back(std::make_shared<Marlin2019>(distType, conf));
            std::cout << "### Generating " << C[C.size()-1]->name() << std::endl;
        }
    }

	
	for (auto c : C) {
	    testCorrectness(c);
	}

	ofstream csv("benchmark_results.csv");
	csv << "codec_name"
	    <<","<< "directory"
	    <<","<< "file"
        <<","<< "pixel_min"
        <<","<< "pixel_max"
	    <<","<< "pixel_count"
	    <<","<< "compression_bytes"
	    <<","<< "compression_avg_time_s"
	    <<","<< "compression_count"
	    <<","<< "decompression_avg_time_s"
	    <<","<< "decompression_count"
        <<","<< "lossless"
        << std::endl;
	std::vector<std::string> encodeImages, decodeImages;
	for (auto c : C) {
	    std::cout << c->name() << "::" << std::endl;
        for (auto dir_path : {
                              "../test_datasets/rawzor",
                              "../test_datasets/iso_12640_2",
                              "../test_datasets/kodak_photocd",
                              "../test_datasets/mixed_datasets",
                              }) {
            auto res = testOnIndividualFiles(c, dir_path, csv);
            encodeImages.push_back(res.first);
            decodeImages.push_back(res.second);
            std::cout << res.first << res.second;
        }
	}
    csv.close();

	return 0;
}
