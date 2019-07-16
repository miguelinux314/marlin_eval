#pragma once
#include <util/codec.hpp>
#include <util/distribution.hpp>

class MarlinBase : public CODEC8withPimpl {
public:
	enum Type { TUNSTALL, MARLIN };
    MarlinBase(Distribution::Type distType = Distribution::Laplace, Type dictType = MARLIN, size_t dictSize = 12, size_t numDict = 11);
};
