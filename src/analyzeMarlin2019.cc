#include <iostream>
#include <fstream>
#include <util/distribution.hpp>

#include <marlin/inc/marlin.h>

using namespace std;
int main() {
	
	int skip=2;
	
	std::vector<std::vector<double>> LaplacianPDF(101);
	for (size_t i=0; i<LaplacianPDF.size(); i++)
		LaplacianPDF[i] = Distribution::pdf(256,Distribution::Laplace,double(i)/double(LaplacianPDF.size()-1));

	std::vector<std::vector<double>> NormalPDF(101);
	for (size_t i=0; i<NormalPDF.size(); i++)
		NormalPDF[i] = Distribution::pdf(256,Distribution::Gaussian,double(i)/double(NormalPDF.size()-1));


	std::map<std::string, double> conf;
	conf["O"] = 4;
	conf["K"] = 8;	
	conf.emplace("iterations",2);

//	conf.emplace("autoMaxWordSize",7);
	conf.emplace("minMarlinSymbols",2);
	conf.emplace("purgeProbabilityThreshold",0.5/4096/128);

	
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
	
	// Unless stated differently: key=12, overlap=4


	// Same Dictionary Size, efficiency over H.
	if (true) {
	
		tex << "\\input{results/ssse1.tex}\n";
		ofstream res("results/ssse1.tex");
		
		auto Dist = LaplacianPDF;

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
		for (size_t i=1; i<LaplacianPDF.size()-1; i+=skip)
			res << "(" << double(i*100.)/Dist.size() << "," << (Marlin("",Dist[i],conf)).efficiency*100. << ")";
		res << "};" << std::endl;

	conf["purgeProbabilityThreshold"] = (0.5/4096);
	conf["O"] = 4;
	conf["K"] = 12;	

		res << "\\addplot+[line width=1pt, gray!50, mark=none] coordinates { ";
		for (size_t i=1; i<LaplacianPDF.size()-1; i+=skip)
			res << "(" << double(i*100.)/Dist.size() << "," << (Marlin("",Dist[i],conf)).efficiency*100. << ")";
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


	tex << "\\end{document}" << endl;
	
	return 0;
}
