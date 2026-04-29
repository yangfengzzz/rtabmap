/*
Copyright (c) 2010-2016, Mathieu Labbe - IntRoLab - Universite de Sherbrooke
All rights reserved.
*/

#ifndef LIGHTGLUE_TORCH_H
#define LIGHTGLUE_TORCH_H

#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>

#include <memory>
#include <string>
#include <vector>

namespace rtabmap
{

class LGMatcher
{
public:
	LGMatcher(
			const std::string & weightsPath,
			float filterThreshold = 0.1f,
			int nLayers = 9,
			float depthConfidence = 0.95f,
			float widthConfidence = 0.99f,
			bool cuda = true);
	virtual ~LGMatcher();

	const std::string & path() const {return path_;}
	float filterThreshold() const {return filterThreshold_;}
	int nLayers() const {return nLayers_;}
	float depthConfidence() const {return depthConfidence_;}
	float widthConfidence() const {return widthConfidence_;}
	bool cuda() const {return cuda_;}
	bool isValid() const;

	std::vector<cv::DMatch> match(
			const cv::Mat & descriptorsQuery,
			const cv::Mat & descriptorsTrain,
			const std::vector<cv::KeyPoint> & keypointsQuery,
			const std::vector<cv::KeyPoint> & keypointsTrain,
			const cv::Size & imageSize);

private:
	class Impl;
	std::shared_ptr<Impl> impl_;
	std::string path_;
	float filterThreshold_;
	int nLayers_;
	float depthConfidence_;
	float widthConfidence_;
	bool cuda_;
};

}

#endif
