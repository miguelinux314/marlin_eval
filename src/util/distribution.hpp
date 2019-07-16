#pragma once
#include <map>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <numeric>

class Distribution {


	static inline std::vector<double> norm1(std::vector<double> pdf) {

		double sum  = std::accumulate(pdf.rbegin(), pdf.rend(), 0.);
		for (auto &v : pdf) v/=sum;
		return pdf;
	}

	static inline std::vector<double> PDFGaussian(size_t N, double b) {

		std::vector<double> pdf(N, 1e-100);
		for (int i=-10*int(N)+1; i<int(10*N); i++)
			pdf[(10*N+i) % N] += std::exp(-double(  i*i)/b );
		return norm1(pdf);
	}

	static inline std::vector<double> PDFLaplace(size_t N, double b) {

		std::vector<double> pdf(N, 1e-100);
		pdf[0] += 1.;
		for (size_t i=1; i<10*N; i++) {
			pdf[      i  % N] += std::exp(-double(  i)/b );
			pdf[(10*N-i) % N] += std::exp(-double(  i)/b );
		}
		return norm1(pdf);
	}

	static inline std::vector<double> PDFExponential(size_t N, double b) {

		std::vector<double> pdf(N, 1e-100);
		pdf[0] += 1.;
		for (size_t i=1; i<10*N; i++)
			pdf[      i  % N] += std::exp(-double(  i)/b );

		return norm1(pdf);
	}

	static inline std::vector<double> PDFPoisson(size_t N, double lambda) {

		std::vector<double> pdf(N, 1e-100);
		double lv = -lambda;
		pdf[0] += std::exp(std::max(std::min(20.,lv),-20.));
		for (size_t i=1; i<10*N; i++) {
			lv = lv + std::log(lambda) - std::log(i);
			pdf[      i  % N] += std::exp(std::max(std::min(20.,lv),-20.));
		}

		return norm1(pdf);
	}
	
	

public:

	enum Type { Gaussian=0, Laplace=1, Exponential=2, Poisson=3 };

	static inline std::vector<double> pdfByType(size_t N, Type type, double var) {

		if (type == Gaussian   ) return PDFGaussian(N,var);
		if (type == Laplace    ) return PDFLaplace(N,var);
		if (type == Exponential) return PDFExponential(N,var);
		if (type == Poisson    ) return PDFPoisson(N,var);
		throw std::runtime_error("Unsupported distribution");		
	}

	static inline std::string typeToName(Type distType) {
	    if (distType == Gaussian) {
	        return "Gaussian";
	    }
        if (distType == Laplace) {
            return "Laplacian";
        }
        if (distType == Exponential) {
            return "Exponential";
        }
        if (distType == Poisson) {
            return "Possion";
        }
        return "UNKNOWN";
	}

	
	static inline double entropy(const std::vector<double> &pdf) {

		double distEntropy=0;
		for (auto &&p : pdf)
			if (p)
				distEntropy += -p*std::log2(p);

		return distEntropy;
	}

	template<typename T> 
	static inline double entropy(const std::map<T,double> &pdf) {

		double distEntropy=0;
		for (auto &&p : pdf)
			if (p.second)
				distEntropy += -p.second*std::log2(p.second);

		return distEntropy;
	}

	template<size_t N>
	static inline double entropy(const std::array<double,N> &pdf) {
		return entropy(std::vector<double>(pdf.begin(), pdf.end()));
	}

	static inline std::vector<double> pdfByEntropy(size_t N, Type type, double h) {

		double b=1<<16;
		// Estimate parameter b from p using dicotomic search
		double stepSize = 1<<15;
		while (stepSize>1E-12) {
			if (h > entropy(pdfByType(N,type,b))/std::log2(N) ) b+=stepSize;
			else b-=stepSize;
			stepSize/=2.;
		}
		
		//std::cerr << "b: " << b << std::endl;

		return pdfByType(N,type,b);
	}

	static inline std::vector<double> pdf(size_t N, Type type, double h) { return pdfByEntropy(N, type,h); }

	static inline std::array<double,256> pdfByEntropy(Type type, double h) {

		auto P = pdfByEntropy(256, type, h);
		std::array<double,256> A;
		for (size_t i=0; i<256; i++) A[i]=P[i];
		return A;
	}

	static inline std::array<double,256> pdf(Type type, double h) { return pdfByEntropy(type,h); }

	static inline std::vector<uint8_t> getResiduals(const std::vector<double> &pdf, size_t S) {

		int8_t cdf[0x10000];
		uint j=0;
		double lim=0;
		for (uint i=0; i<pdf.size(); i++) {
			lim += pdf[i]*0x10000;
			uint ilim = round(lim);
			while (j<ilim)
				cdf[j++]=i;
		}

		std::vector<uint8_t> ret;
		uint32_t rnd =  135154;
		for (size_t i=0; i<S; i++) {
			rnd = 36969 * (rnd & 65535) + (rnd >> 16);
			ret.push_back(cdf[rnd&0xFFFF]);
		}

		return ret;
	}

	static inline std::vector<uint8_t> getResiduals(const std::array<double,256> &pdf, size_t S) {
		return getResiduals(std::vector<double>(pdf.begin(), pdf.end()), S);
	}

};
