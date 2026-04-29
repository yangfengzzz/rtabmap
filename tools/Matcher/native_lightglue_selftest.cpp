/*
Copyright (c) 2010-2016, Mathieu Labbe - IntRoLab - Universite de Sherbrooke
All rights reserved.
*/

#include <rtabmap/core/Parameters.h>
#include <rtabmap/utilite/ULogger.h>
#include <opencv2/core.hpp>

#ifdef RTABMAP_TORCH
#include <lightglue_torch/LightGlue.h>
#endif

#include <cstdlib>
#include <iostream>
#include <vector>

using namespace rtabmap;

int main(int argc, char * argv[])
{
#ifndef RTABMAP_TORCH
	std::cerr << "Torch support is required for native LightGlue self-test." << std::endl;
	return 2;
#else
	ULogger::setLevel(ULogger::kDebug);
	ULogger::setType(ULogger::kTypeConsole);

	if(argc < 2)
	{
		std::cerr << "Usage: native_lightglue_selftest <weights.pth>" << std::endl;
		return 2;
	}

	LGMatcher matcher(
			argv[1],
			Parameters::defaultLightGlueFilterThreshold(),
			Parameters::defaultLightGlueNLayers(),
			Parameters::defaultLightGlueDepthConfidence(),
			Parameters::defaultLightGlueWidthConfidence(),
			false);
	if(!matcher.isValid())
	{
		std::cerr << "Failed to initialize native LightGlue matcher." << std::endl;
		return 1;
	}

	std::vector<cv::KeyPoint> emptyKpts;
	cv::Mat emptyDesc;
	std::vector<cv::DMatch> matches = matcher.match(emptyDesc, emptyDesc, emptyKpts, emptyKpts, cv::Size(640, 480));
	if(!matches.empty())
	{
		std::cerr << "Expected empty matches for empty inputs." << std::endl;
		return 1;
	}

	std::cout << "native_lightglue_selftest OK" << std::endl;
	return 0;
#endif
}
