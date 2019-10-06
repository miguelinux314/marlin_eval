#include <codecs/zstd.hpp>
#include <thread>
#include <zstd.h>

using namespace std;

class ZstdPimpl : public CODEC8AA {
	
	int level;
	
	mutable std::map<std::thread::id,ZSTD_CCtx*> cctx;
	mutable std::map<std::thread::id,ZSTD_DCtx*> dctx;
	
	std::string name() const { return std::string("Zstd")+char('0'+level); }

	void   compress(const AlignedArray8 &in, AlignedArray8 &out) const {
		
		auto &&ctx = cctx[std::this_thread::get_id()];
		if (not ctx) 
			ctx = ZSTD_createCCtx();

		out.resize(ZSTD_compressCCtx(ctx, out.begin(), out.capacity(), in.begin(), in.size(), level));
	}

	void uncompress(const AlignedArray8 &in, AlignedArray8 &out) const {

		auto &&ctx = dctx[std::this_thread::get_id()];
		if (not ctx) 
			ctx = ZSTD_createDCtx();

		out.resize(ZSTD_decompressDCtx(ctx, out.begin(), out.capacity(), in.begin(), in.size()));
	}

public:
	ZstdPimpl(int level_) : level(level_) {}
	
	~ZstdPimpl() {
		for (auto &&c : cctx)
			ZSTD_freeCCtx(c.second);

		for (auto &&c : dctx)
			ZSTD_freeDCtx(c.second);
	}
};

Zstd::Zstd(int level) : CODEC8withPimpl(new ZstdPimpl(level)) {}
