#include <iostream>

#include <marlinlib/marlin.hpp>


using std::cin;
using std::cout;
using std::cerr;
using std::endl;

void usage() {
	
	cout << "Usage: testDictionary [options]" << endl;
	cout << "Options:" << endl;
	cout << "   -h or --help: show this help" << endl;
	cout << "   --custom  <N-1 probabilities>: use a custom distribution" << endl;
	cout << "   --exp  <size> <entropy>: use an exponential distribution with <size> alphabet and <entropy> entropy" << endl;
	cout << "   --lap  <size> <entropy>: use an Laplace distribution with <size> alphabet and <entropy> entropy" << endl;
	cout << "   --norm <size> <entropy>: use an normal distribution with <size> alphabet and <entropy> entropy" << endl;
	exit(-1);
}


int main(int argc, char **argv) {

	if (argc==1) usage();
	
	// Parse command line optins
	std::map<std::string,double> options;
	options["--keySize"]=12;
	options["--overlap"]=2;
	options["--maxWordSize"]=256;

	std::vector<double> P;
	for (int i=1; i<argc; i++) {
		if (argv[i][0]=='-') {
			std::string name;
			while (isalpha(*argv[i]) or *argv[i]=='-') name += *argv[i]++;
			options[name] = 1; 
			if (*argv[i] and *argv[i]=='=') 
				options[name] = atof(argv[i]+1);
		} else {
			P.push_back(atof(argv[i]));
		}
	}
	
	for (auto &op : options)
		cerr << op.first << " " << op.second << endl;
	
	if (options["-h"] or options["--help"] or options.empty())
		usage();
	
	//cerr << "P: "; for (auto p : P) cerr << p << " "; cerr  << endl;


	if (options["--custom"]) {
		// Adding probability for the last symbol, ensuring that all probabilities sum 1
		{
			double lp=1;
			for (auto p : P)
				lp -= p;
			P.push_back(lp);	
		}

		// Check that all probabilities are positive
		for (auto p : P)
			if (p<0)
				usage();
				
	} else if (options["--exp"]) {
		
		if (P.size()!=2 or P[0] < 1 or P[1]<0 or P[1]>1) usage(); 
		P = Distribution::pdf(P[0],Distribution::Exponential,P[1]);
	
	} else if (options["--lap"]) {
		
		if (P.size()!=2 or P[0] < 1 or P[1]<0 or P[1]>1) usage(); 
		P = Distribution::pdf(P[0],Distribution::Laplace,P[1]);
	} else {
		usage();
	}
		
	//cerr << "P: "; for (auto p : P) cerr << p << " "; cerr << Distribution::entropy(P) << endl;
		
	// Ensure that symbols are sorted in order of decreasing probabilities
	//std::sort(P.begin(), P.end(), std::greater<double>());
	
	
	std::cerr << "Marlin" << std::endl;
//	MarlinDictionary(P,options["--size"],options["--tries"]);
	std::cerr << "Tunstall" << std::endl;
	//TunstallDictionary(P,options["--size"]);
	std::cerr << "Marlin2" << std::endl;
//	Marlin2018Simple::setConfiguration("encoderFast",0);
	Marlin2018Simple::setConfiguration("debug",1.);
	Marlin2018Simple::setConfiguration("dedup",0.);
	Marlin2018Simple::theoreticalEfficiency(P,options["--keySize"],options["--overlap"],options["--maxWordSize"]);
	//Marlin2018Simple(P,options["--keySize"],options["--overlap"],options["--maxWordSize"]).benchmark(P,options["--testSize"]);
/*	Marlin2018Simple::setConfiguration("dedup",1.);
	Marlin2018Simple(P,options["--keySize"],options["--overlap"],options["--maxWordSize"]).test(P,options["--testSize"]);
	Marlin2018Simple::setConfiguration("dedup",0.);
	Marlin2018Simple(P,options["--keySize"],options["--overlap"],options["--maxWordSize"]).test(P,options["--testSize"]);
	Marlin2018Simple::setConfiguration("dedup",1.);
	Marlin2018Simple(P,options["--keySize"],options["--overlap"],options["--maxWordSize"]).test(P,options["--testSize"]);
	Marlin2018Simple::setConfiguration("dedup",0.);
	Marlin2018Simple(P,options["--keySize"],options["--overlap"],options["--maxWordSize"]).test(P,options["--testSize"]);
	Marlin2018Simple::setConfiguration("dedup",1.);
	Marlin2018Simple(P,options["--keySize"],options["--overlap"],options["--maxWordSize"]).test(P,options["--testSize"]);
*/
/*	Marlin2Dictionary(P,options["--size"],options["--tries"]);
	Marlin2Dictionary(P,options["--size"]*2,options["--tries"],1);
	Marlin2Dictionary(P,options["--size"]*4,options["--tries"],2);
	Marlin2Dictionary(P,options["--size"]*8,options["--tries"],3);
	Marlin2Dictionary(P,options["--size"]*16,options["--tries"],4);
	Marlin2Dictionary(P,options["--size"]*32,options["--tries"],5);
	Marlin2Dictionary(P,options["--size"]*512,options["--tries"],9);
	Marlin2Dictionary(P,options["--size"]*1024,options["--tries"],10);*/
		
	return 0;
}
