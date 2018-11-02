#include <iostream>
#include <fstream>
#include <util/distribution.hpp>

#include <marlin/inc/marlin.h>


using Word = marlin::TMarlinDictionary<uint8_t,uint8_t>::Word;

static void printEncodingTable(const marlin::TMarlinDictionary<uint8_t,uint8_t> &dictionary) {
	
	
	const size_t FLAG_NEXT_WORD = 1<<31;
	const size_t NumChapters = 1<<dictionary.O;
	const size_t ChapterSize = 1<<dictionary.K;
	
	std::vector<std::vector<int>> table(dictionary.marlinAlphabet.size(),std::vector<int>(1<<(dictionary.O+dictionary.K),-1));
	
	std::vector<std::map<Word, size_t>> positions(NumChapters);

	// Init the mapping (to know where each word goes)
	for (size_t k=0; k<NumChapters; k++)
		for (size_t i=k*ChapterSize; i<(k+1)*ChapterSize; i++)
			positions[k][dictionary.words[i]] = i;
			
	// Link each possible word to its continuation
	for (size_t k=0; k<NumChapters; k++) {
		for (size_t i=k*ChapterSize; i<(k+1)*ChapterSize; i++) {
			auto word = dictionary.words[i];
			size_t wordIdx = i;
			while (word.size() > 1) {
				auto lastSymbol = word.back();						
				word.pop_back();
				if (not positions[k].count(word)) throw(std::runtime_error("This word has no parent. SHOULD NEVER HAPPEN!!!"));
				size_t parentIdx = positions[k][word];
				table[lastSymbol][parentIdx] = wordIdx;
				wordIdx = parentIdx;
			}
		}
	}
				
	//Link between inner dictionaries
	for (size_t k=0; k<NumChapters; k++)
		for (size_t i=k*ChapterSize; i<(k+1)*ChapterSize; i++)
			for (size_t j=0; j<dictionary.marlinAlphabet.size(); j++)
				if (table[j][i] == -1) // words that are not parent of anyone else.
					table[j][i] = positions[i%NumChapters][Word(1,j)] + FLAG_NEXT_WORD;
					
	std::cout << "\\begin{tabular}{ r *{8}{|c}||c*{7}{|c}}" << std::endl;
	std::cout << "    "; for (auto &&w : dictionary.words) { std::cout << " & "; for (auto &&c:w) std::cout << char('a'+c);} std::cout << " \\\\ \\hline" << std::endl;
	for (size_t i=0; i<table.size(); i++) {
		std::cout << "  " << char('a'+i) << " ";
		for (auto &&w: table[i]) { 
			double probability = 1000*dictionary.words[w&0xFFFF].p * dictionary.marlinAlphabet[i].p;
			std::cout << " & \\cellcolor{black!"<< int(probability) <<"}{"; if (probability>60) std::cout << "\\color{white}"; for (auto &&c:dictionary.words[w&0xFFFF]) std::cout << char('a'+c); if (w & FLAG_NEXT_WORD) std::cout << "!"; std::cout << "}"; }
			
		std::cout << " \\\\ \\hline" << std::endl;
	}
	std::cout << "\\end{tabular}" << std::endl;			

	std::cout << "\\begin{tabular}{ r *{8}{|c}||c*{7}{|c}}" << std::endl;
	std::cout << "    "; for (auto &&w : dictionary.words) { std::cout << " & "; for (auto &&c:w) std::cout << char('a'+c);} std::cout << " \\\\ \\hline" << std::endl;
	for (size_t i=0; i<table.size(); i++) {
		std::cout << "  " << char('a'+i) << " ";
		for (auto &&w: table[i]) { std::cout << " & "; for (auto &&c:dictionary.words[w&0xFFFF]) std::cout << char('a'+c); if (w & FLAG_NEXT_WORD) std::cout << "!"; }
			
		std::cout << " \\\\ \\hline" << std::endl;
	}
	std::cout << "\\end{tabular}" << std::endl;			
										
//	return ret;
}


using namespace std;
int main() {
	
	auto dist = Distribution::pdf(8,Distribution::Exponential,0.75);
	
	dist[0] = 1;
	for (size_t i=1; i<dist.size(); i++)
		dist[0] -= dist[i];

	std::map<std::string, double> conf;
	conf["O"] = 0;
	conf["K"] = 3;	
	conf["shift"] = 1;	
	conf["debug"] = 99;
	conf["iterations"] = 5;

	//conf.emplace("maxWordSize",7);
	conf.emplace("minMarlinSymbols",2);
	//conf.emplace("purgeProbabilityThreshold",0.5/4096/128);

	//conf["purgeProbabilityThreshold"] = 0.001;
	//conf.emplace("purgeProbabilityThreshold",0.06);

	
	marlin::TMarlinDictionary<uint8_t,uint8_t> dict(dist,conf);
	printEncodingTable(dict);
	
	
	
	
	for (size_t i=0; i<dist.size(); i++)
		std::cout << "L: " << char('a'+i) << " " << dist[i] << std::endl;
	
	return 0;
}
