#include <codecs/lz4.hpp>
#include <lz4.h>


class Lz4Pimpl : public CODEC8AA {
	
	std::string name() const { return "Lz4"; };

	void   compress(const AlignedArray8 &in, AlignedArray8 &out) const {
		out.resize(LZ4_compress_default((const char *)in.begin(), (char *)out.begin(), in.size(), out.capacity()));
	}

	void uncompress(const AlignedArray8 &in, AlignedArray8 &out) const {
		//printf("IN: %ld %ld\n", in.size(), out.size()); fflush(stdout);
		//LZ4_decompress_fast((const char *)in.begin(), (char *)out.begin(), out.size());
		LZ4_decompress_safe((const char *)in.begin(), (char *)out.begin(), in.size(), out.size());
		//printf("OUT: %ld %ld\n", in.size(), out.size()); fflush(stdout);

	}
};


Lz4::Lz4() : CODEC8withPimpl(new Lz4Pimpl()) {}
