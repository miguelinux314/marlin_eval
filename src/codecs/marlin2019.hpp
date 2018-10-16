#pragma once
#include <util/codec.hpp>
#include <util/distribution.hpp>

struct Marlin2019 : public CODEC8withPimpl { 

	Marlin2019(
		Distribution::Type distType = Distribution::Laplace,
		std::map<std::string, double> conf = std::map<std::string, double>());
};
