#pragma once
#include <util/codec.hpp>
#include <util/distribution.hpp>

struct Marlin2019 : public CODEC8withPimpl { 
public:
    Marlin2019(Distribution::Type distType = Distribution::Laplace,
		std::map<std::string, double> conf = std::map<std::string, double>());
//    :
//            CODEC8withPimpl( new Marlin2019Pimpl(distType, conf) );
};
