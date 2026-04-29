/*
Copyright (c) 2010-2016, Mathieu Labbe - IntRoLab - Universite de Sherbrooke
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Universite de Sherbrooke nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef SUPERGLUE_TORCH_H
#define SUPERGLUE_TORCH_H

#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>

#include <memory>
#include <string>
#include <vector>

namespace rtabmap
{

class SGMatcher
{
public:
	SGMatcher(
			const std::string & weightsPath,
			float matchThreshold = 0.2f,
			int iterations = 20,
			bool cuda = true);
	virtual ~SGMatcher();

	const std::string & path() const {return path_;}
	float matchThreshold() const {return matchThreshold_;}
	int iterations() const {return iterations_;}
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
	float matchThreshold_;
	int iterations_;
	bool cuda_;
};

}

#endif
