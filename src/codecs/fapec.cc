//#include <codecs/fapec.hpp>
//#include <thread>
//#include <iostream>
//#include <string.h>
//extern "C" {
//#include <fapec_opts.h>
//#include <fapec_comp.h>
//#include <fapec_decomp.h>
//}
//
//using namespace std;
//
////extern "C" int fapec_usropts_new(int verbLevel, int askOverwrite, int deleteInput, int enforcePriv,
////                                 int streamMode, int noAttr, int noCompHead,
////                                 int edacOpt, int cryptOpt, int threadPool, int decMode,
////                                 int noNames, int noFoot, int abortErr,
////                                 int noRecurseDir, int keepLinks);
//
//class FapecPimpl : public CODEC8AA {
//
//	std::string name() const { return std::string("Fapec"); }
//
//	void compress(const AlignedArray8 &in, AlignedArray8 &out) const {
//        int options = fapec_usropts_new(FAPEC_VERB_DEBUG, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
//
//        int64_t out_size;
//        unsigned char* out_data;
//
//        fapec_compress_buff2buff(in.data(), &out_data, in.size(), &out_size, options, NULL);
//        out.resize(out_size);
//        memcpy(out_data, out.data(), out_size);
//	}
//
//	void uncompress(const AlignedArray8 &in, AlignedArray8 &out) const {
//
//        int options = fapec_usropts_new(FAPEC_VERB_NONE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
//
//	    int64_t out_size;
//        unsigned char* out_data;
//        fapec_decomp_buff2buff(in.data(), &out_data, in.size(), &out_size, options, NULL);
//        out.resize(out_size);
//        memcpy(out_data, out.data(), out_size);
//	}
//
//public:
//	FapecPimpl()  {}
//};
//
//Fapec::Fapec() : CODEC8withPimpl(new FapecPimpl()) {}
