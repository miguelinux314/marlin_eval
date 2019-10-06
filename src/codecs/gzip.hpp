#pragma once
#include <util/codec.hpp>

// Zlib notice
//Jean-loup Gailly        Mark Adler
//jloup@gzip.org          madler@alumni.caltech.edu
//
//        The deflate format used by zlib was defined by Phil Katz.  The  deflate  and  zlib  specifications  were
//        written  by  L.  Peter  Deutsch.   Thanks  to all the people who reported problems and suggested various
//improvements in zlib; who are too numerous to cite here.

struct Gzip : public CODEC8withPimpl { Gzip(int level=1); };
