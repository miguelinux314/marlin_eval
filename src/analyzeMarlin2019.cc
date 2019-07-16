#include <iostream>
#include <fstream>
#include <util/distribution.hpp>

#include <marlin/inc/marlin.h>

typedef marlin::TMarlinDictionary<uint8_t,uint8_t> MarlinDict;

using namespace std;
int main() {
	
	int skip=2;
	size_t nDistributions = 257;


	std::map<std::string, std::vector<std::vector<double>>> distributions;
	for (size_t i=0; i<nDistributions; i++) {
//		distributions["Exponential"].push_back( Distribution::pdfByEntropy(256,Distribution::Exponential,i/(nDistributions-1.)));
		distributions["Laplace"].push_back( Distribution::pdfByEntropy(256,Distribution::Laplace,i/(nDistributions-1.)));
//		distributions["Gaussian"].push_back( Distribution::pdfByEntropy(256,Distribution::Gaussian,i/(nDistributions-1.)));
		distributions["Poisson"].push_back( Distribution::pdfByEntropy(256,Distribution::Poisson,i/(nDistributions-1.)));
	}

	std::map<std::string, double> baseConf;
	baseConf["O"] = 4;
	baseConf["K"] = 8;	
	baseConf.emplace("iterations",2);

//	conf.emplace("autoMaxWordSize",7);
	baseConf.emplace("minMarlinSymbols",2);
	baseConf.emplace("purbaseConfgeProbabilityThreshold",0.5/4096/128);

	
	ofstream tex("out.tex");
	
	tex << R"ML(
\documentclass{article}
\usepackage[a4paper, margin=1cm]{geometry}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.13}
\usepgfplotslibrary{colorbrewer}
\pgfplotsset{cycle list/Set2}
\begin{document}
)ML";

	bool run_analysis = true;

	// Unless stated differently: key=12, overlap=4

	// Same Dictionary Size, efficiency over H.
	if (run_analysis) {
	
		auto conf = baseConf;
		tex << "\\input{results/ssse1.tex}\n";
		ofstream res("results/ssse1.tex");
		
		auto Dist = distributions["Laplace"];

		res << R"ML(
		\begin{figure}
		\centering
		\begin{tikzpicture} 
		\begin{axis}[
			title="Same Dictionary Size Efficiency", 
			title style={yshift=-1mm},
			height=3cm, width=6cm,
			scale only axis, 
			enlargelimits=false, 
			xmin=0, xmax=100, 
			ymin=80, ymax=100, 
			ymajorgrids, major grid style={dotted, gray}, 
			x tick label style={font={\footnotesize},yshift=1mm}, 
			y tick label style={font={\footnotesize},xshift=-1mm},
			ylabel={\emph{Efficiency(\%)}}, 
			xlabel={\emph{Entropy (\%)}}, 
			xlabel style={font={\footnotesize},xshift= 2mm}, 
			ylabel style={font={\footnotesize},yshift=-2mm},
			legend style={at={(0.5,-0.2)},legend columns=-1,anchor=north,nodes={scale=0.75, transform shape}}
			])ML";

	conf["purgeProbabilityThreshold"] = (0.5/4096);
	conf["O"] = 4;
	conf["K"] = 8;	
			

		res << "\\addplot+[line width=1pt, gray!50, mark=none] coordinates { ";
		for (size_t i=1; i<Dist.size()-1; i+=skip)
			res << "(" << double(i*100.)/Dist.size() << "," << (MarlinDict(Dist[i],conf)).efficiency*100. << ")";
		res << "};" << std::endl;

	conf["purgeProbabilityThreshold"] = (0.5/4096);
	conf["O"] = 4;
	conf["K"] = 12;	

		res << "\\addplot+[line width=1pt, gray!50, mark=none] coordinates { ";
		for (size_t i=1; i<Dist.size()-1; i+=skip)
			res << "(" << double(i*100.)/Dist.size() << "," << (MarlinDict(Dist[i],conf)).efficiency*100. << ")";
		res << "};" << std::endl;


		res << "\\legend{No Overlap, Victim only, Specialized only, Victim + Specialized}" << std::endl;
			
		res << R"ML(
			\end{axis} 
			\end{tikzpicture}
			\caption{}
			\label{fig:}
			\end{figure}
			)ML";
	}


	// All possible shift plotted
	if (run_analysis) {
	
		auto conf = baseConf;
		//tex << "\\input{results/ssse1.tex}\n";
		ofstream res("results/ssse1.tex");
//		auto &res = tex;

		for (auto &&distribution : distributions) {
			auto Dist = distribution.second;

			res << R"ML(
			\begin{figure}
			\centering
			\begin{tikzpicture} 
			\begin{axis}[
				title="Same Dictionary Size Efficiency", 
				title style={yshift=-1mm},
				height=3cm, width=6cm,
				scale only axis, 
				enlargelimits=false, 
				xmin=0, xmax=100, 
				ymin=80, ymax=100, 
				ymajorgrids, major grid style={dotted, gray}, 
				x tick label style={font={\footnotesize},yshift=1mm}, 
				y tick label style={font={\footnotesize},xshift=-1mm},
				ylabel={\emph{Efficiency(\%)}}, 
				xlabel={\emph{Entropy (\%)}}, 
				xlabel style={font={\footnotesize},xshift= 2mm}, 
				ylabel style={font={\footnotesize},yshift=-2mm},
				legend style={at={(0.5,-0.2)},legend columns=-1,anchor=north,nodes={scale=0.75, transform shape}}
				])ML";


		for (size_t shift=0; shift<7; shift++) {
			conf["purgeProbabilityThreshold"] = (0.5/4096);
			conf["O"] = 4;
			conf["K"] = 8;	
			conf["shift"] = shift;
				

				res << "\\addplot+[line width=1pt, gray!50, mark=none] coordinates { ";
				for (size_t i=1; i<Dist.size()-1; i+=skip)
					res << "(" << double(i*100.)/Dist.size() << "," << (MarlinDict(Dist[i],conf)).efficiency*100. << ")";
				res << "};" << std::endl;
		}



			res << "\\legend{No Overlap, Victim only, Specialized only, Victim + Specialized}" << std::endl;
				
			res << R"ML(
				\end{axis} 
				\end{tikzpicture}
				\caption{}
				\label{fig:}
				\end{figure}
				)ML";
		}
	}


	// Rice-Marlin vs Rice DONE!
	if (run_analysis) {
	
		
		//tex << "\\input{results/ssse1.tex}\n";
		ofstream res("results/ssse2.tex");
//		auto &res = tex;

		for (auto &&distribution : distributions) {
			auto Dist = distribution.second;

			res << R"ML(
			\begin{figure}
			\centering
			\begin{tikzpicture} 
			\begin{axis}[
				title="Same Dictionary Size Efficiency", 
				title style={yshift=-1mm},
				height=3cm, width=6cm,
				scale only axis, 
				enlargelimits=false, 
				xmin=0, xmax=100, 
				ymin=80, ymax=100, 
				ymajorgrids, major grid style={dotted, gray}, 
				x tick label style={font={\footnotesize},yshift=1mm}, 
				y tick label style={font={\footnotesize},xshift=-1mm},
				ylabel={\emph{Efficiency(\%)}}, 
				xlabel={\emph{Entropy (\%)}}, 
				xlabel style={font={\footnotesize},xshift= 2mm}, 
				ylabel style={font={\footnotesize},yshift=-2mm},
				legend style={at={(0.5,-0.2)},legend columns=-1,anchor=north,nodes={scale=0.75, transform shape}}
				])ML";

				// RICE Efficiency
				res << "\\addplot+[line width=2pt, gray!50, mark=none] coordinates { ";
				for (size_t i=1; i<Dist.size()-1; i+=skip) {

					for (size_t j=0; j<Dist[i].size(); j++) {
						if (Dist[i][j]>0 and Dist[i][j]<1) continue;
						std::cout << Dist[i][j] << std::endl;
					}
					
					double riceEfficiency = 0;
					for (size_t shift=0; shift<8; shift++) {
						double meansize = 0;
						
						for (size_t j=0; j<Dist[i].size(); j++)
							meansize += Dist[i][j] * (shift + 1 + (j>>shift));
						
						if (riceEfficiency < i/meansize)
							riceEfficiency = i/meansize;
						
					}

					for (size_t shift=0; shift<8; shift++) {
						double meansize = Dist[i][0] * (shift + 1);
						for (size_t j=1; j<Dist[i].size(); j++)
							meansize += Dist[i][j] * (shift + 2 + (int(j>127?256-j:j)>>shift));
						
						if (riceEfficiency < i/meansize)
							riceEfficiency = i/meansize;
						
					}
					
					riceEfficiency = 8*riceEfficiency/Dist.size();
					
					//std::cout << riceEfficiency << std::endl;
					
					
					
					res << "(" << double(i*100.)/Dist.size() << "," << riceEfficiency*100.<< ")";
				}
				res << "};" << std::endl;



				{
					auto conf = baseConf;
					conf["autoMaxWordSize"]=512;
					conf["purgeProbabilityThreshold"] = 0;
					conf["O"] = 4;
					conf["K"] = 8;	
					

					res << "\\addplot+[line width=2pt, blue!50, mark=none] coordinates { ";
					for (size_t i=1; i<Dist.size()-1; i+=skip) {
						double marlinEfficiency = MarlinDict(Dist[i],conf).efficiency*100.;
						res << "(" << double(i*100.)/Dist.size() << "," << marlinEfficiency << ")";
					}
					res << "};" << std::endl;
				}

				{
					auto conf = baseConf;
					conf["autoMaxWordSize"]=512;
					conf["purgeProbabilityThreshold"] = (0.5/4096);
					conf["O"] = 4;
					conf["K"] = 8;
					conf["shift"]=0;
					

					res << "\\addplot+[line width=2pt, red!50, mark=none] coordinates { ";
					for (size_t i=1; i<Dist.size()-1; i+=skip) {
						double marlinEfficiency = MarlinDict(Dist[i],conf).efficiency*100.;
						res << "(" << double(i*100.)/Dist.size() << "," << marlinEfficiency << ")";
					}
					res << "};" << std::endl;
				}

				{
					auto conf = baseConf;
					conf["autoMaxWordSize"]=512;
					conf["purgeProbabilityThreshold"] = (0.5/4096);
					conf["O"] = 4;
					conf["K"] = 8;	
					

					res << "\\addplot+[line width=1pt, green, mark=none] coordinates { ";
					for (size_t i=1; i<Dist.size()-1; i+=skip) {
						double marlinEfficiency = MarlinDict(Dist[i],conf).efficiency*100.;
						res << "(" << double(i*100.)/Dist.size() << "," << marlinEfficiency << ")";
					}
					res << "};" << std::endl;
				}


				

			res << "\\legend{No Overlap, Victim only, Specialized only, Victim + Specialized}" << std::endl;
				
			res << R"ML(
				\end{axis} 
				\end{tikzpicture}
				\caption{}
				\label{fig:}
				\end{figure}
				)ML";
		}
	}


	// Rice-Marlin vs Marlin
	if (run_analysis) {
	
		
		auto &res = tex;

		for (auto &&distribution : distributions) {
			auto Dist = distribution.second;

			res << R"ML(
			\begin{figure}
			\centering
			\begin{tikzpicture} 
			\begin{axis}[
				title="Same Dictionary Size Efficiency", 
				title style={yshift=-1mm},
				height=3cm, width=6cm,
				scale only axis, 
				enlargelimits=false, 
				xmin=0, xmax=100, 
				ymin=80, ymax=100, 
				ymajorgrids, major grid style={dotted, gray}, 
				x tick label style={font={\footnotesize},yshift=1mm}, 
				y tick label style={font={\footnotesize},xshift=-1mm},
				ylabel={\emph{Efficiency(\%)}}, 
				xlabel={\emph{Entropy (\%)}}, 
				xlabel style={font={\footnotesize},xshift= 2mm}, 
				ylabel style={font={\footnotesize},yshift=-2mm},
				legend style={at={(0.5,-0.2)},legend columns=-1,anchor=north,nodes={scale=0.75, transform shape}}
				])ML";

				{
					auto conf = baseConf;
					conf["autoMaxWordSize"]=512;
					conf["purgeProbabilityThreshold"] = (0.5/4096);
					conf["O"] = 4;
					conf["K"] = 8;	
					

					res << "\\addplot+[line width=2pt, blue!50, mark=none] coordinates { ";
					for (size_t i=1; i<Dist.size()-1; i+=skip) {
						double marlinEfficiency = MarlinDict(Dist[i],conf).efficiency*100.;
						res << "(" << double(i*100.)/Dist.size() << "," << marlinEfficiency << ")";
					}
					res << "};" << std::endl;
				}

				{
					auto conf = baseConf;
					conf["autoMaxWordSize"]=512;
					conf["purgeProbabilityThreshold"] = 0;
					conf["O"] = 4;
					conf["K"] = 12;
					conf["shift"]=0;
					

					res << "\\addplot+[line width=2pt, red!50, mark=none] coordinates { ";
					for (size_t i=1; i<Dist.size()-1; i+=skip) {
						double marlinEfficiency = MarlinDict(Dist[i],conf).efficiency*100.;
						res << "(" << double(i*100.)/Dist.size() << "," << marlinEfficiency << ")";
					}
					res << "};" << std::endl;
				}

				{
					auto conf = baseConf;
					conf["autoMaxWordSize"]=512;
					conf["purgeProbabilityThreshold"] = 0;
					conf["O"] = 0;
					conf["K"] = 12;
					conf["shift"]=0;
					

					res << "\\addplot+[line width=2pt, red!50, mark=none] coordinates { ";
					for (size_t i=1; i<Dist.size()-1; i+=skip) {
						double marlinEfficiency = MarlinDict(Dist[i],conf).efficiency*100.;
						res << "(" << double(i*100.)/Dist.size() << "," << marlinEfficiency << ")";
					}
					res << "};" << std::endl;
				}

				

			res << "\\legend{No Overlap, Victim only, Specialized only, Victim + Specialized}" << std::endl;
				
			res << R"ML(
				\end{axis} 
				\end{tikzpicture}
				\caption{}
				\label{fig:}
				\end{figure}
				)ML";
		}
	}


	// Dictionary efficiency vs size and shift
	if (run_analysis) {
		
		auto conf = baseConf;


		tex << R"ML(
		\begin{figure}
		\centering
		\begin{tikzpicture} 
		\begin{semilogxaxis}[
			title="Decoding Speed vs Efficiency", 
			title style={yshift=-1mm},
			height=3cm, width=5cm,
%			nodes near coords={(\coordindex)},
%			log origin=infty, 
%			log ticks with fixed point, 
			scale only axis, 
			enlargelimits=false, 
			xmin=8, xmax=16384, 
			ymin=50, ymax=100, 
			ymajorgrids, major grid style={dotted, gray}, 
%			xtick=data,
			xtick={16,64,256,1024,4096},
			xticklabels={$2^{4}$, $2^6$, $2^8$, $2^{10}$, $2^{12}$},
			x tick label style={font={\footnotesize},yshift=1mm}, 
			y tick label style={font={\footnotesize},xshift=-1mm},
%			ylabel style={font={\footnotesize},yshift=4mm}, 
%			xlabel style={font={\footnotesize},yshift=5.25mm, xshift=29mm},
%			axis y line=left,
			ylabel={\emph{Efficiency}}, 
			xlabel={\emph{H(\%)}}, 
			ylabel style={font={\footnotesize},yshift=4mm}, 
			xlabel style={font={\footnotesize},yshift=5.25mm, xshift=29mm}
			])ML";
			
			
		conf["purgeProbabilityThreshold"] = 0;
			
//		auto Dist = LaplacianPDF;
		auto Dist = distributions["Laplace"];
		for (size_t shift=0; shift<=5; shift++) {
			tex << R"ML( \addplot+[mark=none] coordinates { )ML" << std::endl;
			std::vector<std::pair<double, double>> values;
			for (size_t O=0; O<=0; O++) {
				for (double K= 1; K<=14; K+=0.125) {
					if (K+shift<8) continue;
					if (K+O>16) continue;
					conf["shift"] = shift;
					conf["O"] = O;
					conf["K"] = std::ceil(K);
					conf["numMarlinWords"] = std::pow(2,K+O);

					values.emplace_back(std::pow(2,K+O),MarlinDict(Dist[(Dist.size()-1)/2],conf).efficiency*100.);
					//std::cout << values.back().first << " " << values.back().second << std::endl;
					//tex << "(" << (1<<(O+K)) << "," << MarlinDict(Dist[(Dist.size()-1)/2],conf).efficiency*100. << ")";
					//tex << "% " << shift << " " << K << " " << O << std::endl;
				}
			}
			
			for (auto &&p : values)
				tex << "(" << p.first << "," << p.second << ")" << std::endl;

			tex << "};" << std::endl;
		}




		tex << R"ML(
			%\legend{Dedup, No Dedup, No Overlap}
			)ML";
			
			
		tex << R"ML(
			\end{semilogxaxis} 
			\end{tikzpicture}
			\caption{}
			\label{fig:}
			\end{figure}
			)ML";
	}


	// Dictionary efficiency vs size and shift (horizontal limits)
	if (run_analysis) {
		
		auto Dists = distributions["Laplace"];
		auto dist = Dists[(Dists.size()-1)/2];
		for (size_t shift=0; shift<=5; shift++) {
			std::map<int,double> pdf;
			for (size_t i=0; i<dist.size(); i++)
				pdf[i>>shift] += dist[i];
			
			std::cout << shift << " " << Distribution::entropy(dist) / (shift+Distribution::entropy(pdf)) << std::endl;
		}
	}

	

	tex << "\\end{document}" << endl;
	
	return 0;
}
