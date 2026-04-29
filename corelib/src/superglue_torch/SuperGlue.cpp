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

#include <superglue_torch/SuperGlue.h>
#include <rtabmap/utilite/ULogger.h>
#include <rtabmap/utilite/UDirectory.h>
#include <rtabmap/utilite/UFile.h>
#include <rtabmap/utilite/UConversion.h>

#include <torch/torch.h>
#include <torch/csrc/jit/serialization/import_read.h>
#include <caffe2/serialize/inline_container.h>

#include <cmath>
#include <fstream>

namespace rtabmap
{

namespace
{
using torch::indexing::Slice;

torch::nn::Sequential makeMLP(const std::vector<int64_t> & channels, bool doBn = true)
{
	torch::nn::Sequential seq;
	for(size_t i=1; i<channels.size(); ++i)
	{
		seq->push_back(torch::nn::Conv1d(torch::nn::Conv1dOptions(channels[i-1], channels[i], 1).bias(true)));
		if(i < channels.size()-1)
		{
			if(doBn)
			{
				seq->push_back(torch::nn::BatchNorm1d(channels[i]));
			}
			seq->push_back(torch::nn::ReLU());
		}
	}
	return seq;
}

torch::Tensor normalizeKeypoints(const torch::Tensor & keypoints, const cv::Size & imageSize)
{
	auto width = torch::tensor((float)imageSize.width, keypoints.options());
	auto height = torch::tensor((float)imageSize.height, keypoints.options());
	auto size = torch::stack({width, height}, 0).unsqueeze(0);
	auto center = size / 2.0f;
	auto scaling = std::get<0>(size.max(1, true)) * 0.7f;
	return (keypoints - center.unsqueeze(1)) / scaling.unsqueeze(1);
}

std::pair<torch::Tensor, torch::Tensor> attention(
		const torch::Tensor & query,
		const torch::Tensor & key,
		const torch::Tensor & value)
{
	const double dim = (double)query.size(1);
	auto scores = torch::einsum("bdhn,bdhm->bhnm", {query, key}) / std::sqrt(dim);
	auto prob = torch::softmax(scores, -1);
	return std::make_pair(torch::einsum("bhnm,bdhm->bdhn", {prob, value}), prob);
}

torch::Tensor logSinkhornIterations(torch::Tensor z, const torch::Tensor & logMu, const torch::Tensor & logNu, int iterations)
{
	auto u = torch::zeros_like(logMu);
	auto v = torch::zeros_like(logNu);
	for(int i=0; i<iterations; ++i)
	{
		u = logMu - torch::logsumexp(z + v.unsqueeze(1), 2);
		v = logNu - torch::logsumexp(z + u.unsqueeze(2), 1);
	}
	return z + u.unsqueeze(2) + v.unsqueeze(1);
}

torch::Tensor logOptimalTransport(const torch::Tensor & scores, const torch::Tensor & alpha, int iterations)
{
	const int64_t b = scores.size(0);
	const int64_t m = scores.size(1);
	const int64_t n = scores.size(2);

	auto bins0 = alpha.expand({b, m, 1});
	auto bins1 = alpha.expand({b, 1, n});
	auto alphaExpanded = alpha.expand({b, 1, 1});
	auto couplings = torch::cat(
			{
				torch::cat({scores, bins0}, -1),
				torch::cat({bins1, alphaExpanded}, -1)
			},
			1);

	auto ms = torch::full({}, (double)m, scores.options());
	auto ns = torch::full({}, (double)n, scores.options());
	auto norm = -(ms + ns).log();
	auto logMu = torch::cat({norm.expand({m}), ns.log().unsqueeze(0) + norm}, 0).unsqueeze(0).expand({b, m+1});
	auto logNu = torch::cat({norm.expand({n}), ms.log().unsqueeze(0) + norm}, 0).unsqueeze(0).expand({b, n+1});
	return logSinkhornIterations(couplings, logMu, logNu, iterations) - norm;
}

torch::Tensor arangeLike(const torch::Tensor & x, int dim)
{
	return torch::arange(x.size(dim), x.options().dtype(torch::kLong));
}

bool loadPickledStateDict(const std::shared_ptr<torch::nn::Module> & module, const std::string & path, std::string & error)
{
	std::ifstream file(path, std::ios::binary);
	if(!file.is_open())
	{
		error = "Failed to open weights file.";
		return false;
	}

	std::vector<char> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	if(data.empty())
	{
		error = "Weights file is empty.";
		return false;
	}

	torch::IValue ivalue;
	try
	{
		ivalue = torch::pickle_load(data);
	}
	catch(const c10::Error & e)
	{
		error = e.what_without_backtrace();
		return false;
	}

	if(ivalue.isNone())
	{
		error = "PyTorch C++ deserialization dropped an OrderedDict root. Convert the checkpoint with scripts/convert_superglue_weights.py and use the converted file.";
		return false;
	}
	if(!ivalue.isGenericDict())
	{
		error = "Pickle root is not a state dict.";
		return false;
	}

	std::map<std::string, torch::Tensor> stateDict;
	for(const auto & item : ivalue.toGenericDict())
	{
		if(item.key().isString() && item.value().isTensor())
		{
			stateDict.insert(std::make_pair(item.key().toStringRef(), item.value().toTensor()));
		}
	}

	torch::NoGradGuard noGradGuard;
	for(auto & item : module->named_parameters(true))
	{
		auto iter = stateDict.find(item.key());
		if(iter == stateDict.end())
		{
			error = uFormat("Missing parameter \"%s\" in state dict.", item.key().c_str());
			return false;
		}
		item.value().copy_(iter->second.to(item.value().device(), item.value().dtype()));
	}
	for(auto & item : module->named_buffers(true))
	{
		auto iter = stateDict.find(item.key());
		if(iter == stateDict.end())
		{
			error = uFormat("Missing buffer \"%s\" in state dict.", item.key().c_str());
			return false;
		}
		item.value().copy_(iter->second.to(item.value().device(), item.value().dtype()));
	}
	return true;
}

bool looksLikeLegacyPickle(const std::string & path)
{
	std::ifstream file(path, std::ios::binary);
	if(!file.is_open())
	{
		return false;
	}
	unsigned char magic[2] = {0, 0};
	file.read((char*)magic, 2);
	return file.gcount() == 2 && magic[0] == 0x80 && magic[1] == 0x02;
}

bool loadStateDictFromZip(const std::shared_ptr<torch::nn::Module> & module, const std::string & path, std::string & error)
{
	std::unique_ptr<caffe2::serialize::PyTorchStreamReader> reader;
	try
	{
		reader.reset(new caffe2::serialize::PyTorchStreamReader(path));
	}
	catch(const c10::Error & e)
	{
		error = e.what_without_backtrace();
		return false;
	}

	torch::IValue ivalue;
	try
	{
		ivalue = torch::jit::readArchiveAndTensors(
				"data",
				"",
				"data/",
				std::nullopt,
				std::nullopt,
				torch::kCPU,
				*reader);
	}
	catch(const c10::Error & e)
	{
		error = e.what_without_backtrace();
		return false;
	}

	if(ivalue.isNone())
	{
		error = "PyTorch C++ deserialization dropped an OrderedDict root. Convert the checkpoint with scripts/convert_superglue_weights.py and use the converted file.";
		return false;
	}
	if(!ivalue.isGenericDict())
	{
		error = "Archive root is not a state dict.";
		return false;
	}

	std::map<std::string, torch::Tensor> stateDict;
	for(const auto & item : ivalue.toGenericDict())
	{
		if(item.key().isString() && item.value().isTensor())
		{
			stateDict.insert(std::make_pair(item.key().toStringRef(), item.value().toTensor()));
		}
	}

	torch::NoGradGuard noGradGuard;
	for(auto & item : module->named_parameters(true))
	{
		auto iter = stateDict.find(item.key());
		if(iter == stateDict.end())
		{
			error = uFormat("Missing parameter \"%s\" in archive state dict.", item.key().c_str());
			return false;
		}
		item.value().copy_(iter->second.to(item.value().device(), item.value().dtype()));
	}
	for(auto & item : module->named_buffers(true))
	{
		auto iter = stateDict.find(item.key());
		if(iter == stateDict.end())
		{
			error = uFormat("Missing buffer \"%s\" in archive state dict.", item.key().c_str());
			return false;
		}
		item.value().copy_(iter->second.to(item.value().device(), item.value().dtype()));
	}
	return true;
}

bool loadArchivedStateDict(const std::shared_ptr<torch::nn::Module> & module, const std::string & path, std::string & error)
{
	torch::serialize::InputArchive archive;
	try
	{
		archive.load_from(path);
	}
	catch(const c10::Error & e)
	{
		error = e.what_without_backtrace();
		return false;
	}

	torch::NoGradGuard noGradGuard;
	for(auto & item : module->named_parameters(true))
	{
		torch::Tensor tensor;
		if(!archive.try_read(item.key(), tensor, false))
		{
			error = uFormat("Missing parameter \"%s\" in archived state dict.", item.key().c_str());
			return false;
		}
		item.value().copy_(tensor.to(item.value().device(), item.value().dtype()));
	}
	for(auto & item : module->named_buffers(true))
	{
		torch::Tensor tensor;
		if(!archive.try_read(item.key(), tensor, true))
		{
			error = uFormat("Missing buffer \"%s\" in archived state dict.", item.key().c_str());
			return false;
		}
		item.value().copy_(tensor.to(item.value().device(), item.value().dtype()));
	}
	return true;
}

struct KeypointEncoderImpl : torch::nn::Module
{
	KeypointEncoderImpl(int64_t featureDim, const std::vector<int64_t> & layers) :
			encoder(register_module("encoder", makeMLP({3, layers[0], layers[1], layers[2], featureDim})))
	{
		torch::NoGradGuard noGradGuard;
		auto lastConv = encoder->ptr<torch::nn::Conv1dImpl>(encoder->size() - 1);
		if(lastConv.get())
		{
			lastConv->bias.zero_();
		}
	}

	torch::Tensor forward(const torch::Tensor & keypoints, const torch::Tensor & scores)
	{
		return encoder->forward(torch::cat({keypoints.transpose(1, 2), scores.unsqueeze(1)}, 1));
	}

	torch::nn::Sequential encoder{nullptr};
};
TORCH_MODULE(KeypointEncoder);

struct MultiHeadedAttentionImpl : torch::nn::Module
{
	MultiHeadedAttentionImpl(int64_t numHeads, int64_t modelDim) :
			dim_(modelDim / numHeads),
			numHeads_(numHeads),
			merge(register_module("merge", torch::nn::Conv1d(torch::nn::Conv1dOptions(modelDim, modelDim, 1).bias(true)))),
			proj(register_module("proj", torch::nn::ModuleList()))
	{
		for(int i=0; i<3; ++i)
		{
			proj->push_back(torch::nn::Conv1d(torch::nn::Conv1dOptions(modelDim, modelDim, 1).bias(true)));
		}
	}

	torch::Tensor forward(const torch::Tensor & query, const torch::Tensor & key, const torch::Tensor & value)
	{
		const int64_t batchDim = query.size(0);
		auto queryProj = proj->ptr<torch::nn::Conv1dImpl>(0)->forward(query).view({batchDim, dim_, numHeads_, -1});
		auto keyProj = proj->ptr<torch::nn::Conv1dImpl>(1)->forward(key).view({batchDim, dim_, numHeads_, -1});
		auto valueProj = proj->ptr<torch::nn::Conv1dImpl>(2)->forward(value).view({batchDim, dim_, numHeads_, -1});
		auto x = attention(queryProj, keyProj, valueProj).first;
		return merge->forward(x.contiguous().view({batchDim, dim_ * numHeads_, -1}));
	}

	int64_t dim_;
	int64_t numHeads_;
	torch::nn::Conv1d merge{nullptr};
	torch::nn::ModuleList proj{nullptr};
};
TORCH_MODULE(MultiHeadedAttention);

struct AttentionalPropagationImpl : torch::nn::Module
{
	AttentionalPropagationImpl(int64_t featureDim, int64_t numHeads) :
			attn(register_module("attn", MultiHeadedAttention(numHeads, featureDim))),
			mlp(register_module("mlp", makeMLP({featureDim * 2, featureDim * 2, featureDim})))
	{
		torch::NoGradGuard noGradGuard;
		auto lastConv = mlp->ptr<torch::nn::Conv1dImpl>(mlp->size() - 1);
		if(lastConv.get())
		{
			lastConv->bias.zero_();
		}
	}

	torch::Tensor forward(const torch::Tensor & x, const torch::Tensor & source)
	{
		auto message = attn->forward(x, source, source);
		return mlp->forward(torch::cat({x, message}, 1));
	}

	MultiHeadedAttention attn{nullptr};
	torch::nn::Sequential mlp{nullptr};
};
TORCH_MODULE(AttentionalPropagation);

struct AttentionalGNNImpl : torch::nn::Module
{
	AttentionalGNNImpl(int64_t featureDim, const std::vector<std::string> & layerNames) :
			layers(register_module("layers", torch::nn::ModuleList())),
			names_(layerNames)
	{
		for(size_t i=0; i<layerNames.size(); ++i)
		{
			layers->push_back(AttentionalPropagation(featureDim, 4));
		}
	}

	std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor desc0, torch::Tensor desc1)
	{
		for(size_t i=0; i<names_.size(); ++i)
		{
			torch::Tensor src0 = names_[i] == "cross" ? desc1 : desc0;
			torch::Tensor src1 = names_[i] == "cross" ? desc0 : desc1;
			auto layer = layers->ptr<AttentionalPropagationImpl>(i);
			auto delta0 = layer->forward(desc0, src0);
			auto delta1 = layer->forward(desc1, src1);
			desc0 = desc0 + delta0;
			desc1 = desc1 + delta1;
		}
		return std::make_pair(desc0, desc1);
	}

	torch::nn::ModuleList layers{nullptr};
	std::vector<std::string> names_;
};
TORCH_MODULE(AttentionalGNN);

} // namespace

class SGMatcher::Impl : public torch::nn::Module
{
public:
	Impl(float matchThreshold, int iterations) :
			matchThreshold_(matchThreshold),
			iterations_(iterations),
			kenc(register_module("kenc", KeypointEncoder(std::make_shared<KeypointEncoderImpl>(256, std::vector<int64_t>{32, 64, 128})))),
			gnn(register_module("gnn", AttentionalGNN(std::make_shared<AttentionalGNNImpl>(256, std::vector<std::string>{"self", "cross", "self", "cross", "self", "cross", "self", "cross", "self", "cross", "self", "cross", "self", "cross", "self", "cross", "self", "cross"})))),
			finalProj(register_module("final_proj", torch::nn::Conv1d(torch::nn::Conv1dOptions(256, 256, 1).bias(true))))
	{
		binScore = register_parameter("bin_score", torch::ones({1}));
	}

	std::vector<cv::DMatch> match(
			const cv::Mat & descriptorsQuery,
			const cv::Mat & descriptorsTrain,
			const std::vector<cv::KeyPoint> & keypointsQuery,
			const std::vector<cv::KeyPoint> & keypointsTrain,
			const cv::Size & imageSize,
			const torch::Device & device)
	{
		std::vector<cv::DMatch> matches;
		if(descriptorsQuery.empty() || descriptorsTrain.empty() || keypointsQuery.empty() || keypointsTrain.empty())
		{
			return matches;
		}
		if(descriptorsQuery.type() != CV_32F || descriptorsTrain.type() != CV_32F)
		{
			UERROR("Native SuperGlue supports only float descriptors.");
			return matches;
		}
		if(descriptorsQuery.cols != 256 || descriptorsTrain.cols != 256)
		{
			UERROR("Native SuperGlue expects 256-dimensional descriptors (query=%d train=%d).", descriptorsQuery.cols, descriptorsTrain.cols);
			return matches;
		}
		if(descriptorsQuery.rows != (int)keypointsQuery.size() || descriptorsTrain.rows != (int)keypointsTrain.size())
		{
			UERROR("Descriptors and keypoints size mismatch for native SuperGlue.");
			return matches;
		}
		if(imageSize.width <= 0 || imageSize.height <= 0)
		{
			UERROR("Invalid image size for native SuperGlue.");
			return matches;
		}

		torch::NoGradGuard noGradGuard;

		auto queryTensor = torch::from_blob((void*)descriptorsQuery.data, {descriptorsQuery.rows, descriptorsQuery.cols}, torch::TensorOptions().dtype(torch::kFloat32)).transpose(0, 1).contiguous().unsqueeze(0).to(device);
		auto trainTensor = torch::from_blob((void*)descriptorsTrain.data, {descriptorsTrain.rows, descriptorsTrain.cols}, torch::TensorOptions().dtype(torch::kFloat32)).transpose(0, 1).contiguous().unsqueeze(0).to(device);

		std::vector<float> keypointsQueryVec(keypointsQuery.size() * 2);
		std::vector<float> keypointsTrainVec(keypointsTrain.size() * 2);
		std::vector<float> scoresQueryVec(keypointsQuery.size());
		std::vector<float> scoresTrainVec(keypointsTrain.size());
		for(size_t i=0; i<keypointsQuery.size(); ++i)
		{
			keypointsQueryVec[i * 2] = keypointsQuery[i].pt.x;
			keypointsQueryVec[i * 2 + 1] = keypointsQuery[i].pt.y;
			scoresQueryVec[i] = keypointsQuery[i].response;
		}
		for(size_t i=0; i<keypointsTrain.size(); ++i)
		{
			keypointsTrainVec[i * 2] = keypointsTrain[i].pt.x;
			keypointsTrainVec[i * 2 + 1] = keypointsTrain[i].pt.y;
			scoresTrainVec[i] = keypointsTrain[i].response;
		}

		auto kptsQuery = torch::from_blob(keypointsQueryVec.data(), {(long)keypointsQuery.size(), 2}, torch::TensorOptions().dtype(torch::kFloat32)).unsqueeze(0).to(device);
		auto kptsTrain = torch::from_blob(keypointsTrainVec.data(), {(long)keypointsTrain.size(), 2}, torch::TensorOptions().dtype(torch::kFloat32)).unsqueeze(0).to(device);
		auto scoresQuery = torch::from_blob(scoresQueryVec.data(), {(long)keypointsQuery.size()}, torch::TensorOptions().dtype(torch::kFloat32)).unsqueeze(0).to(device);
		auto scoresTrain = torch::from_blob(scoresTrainVec.data(), {(long)keypointsTrain.size()}, torch::TensorOptions().dtype(torch::kFloat32)).unsqueeze(0).to(device);

		auto normQuery = normalizeKeypoints(kptsQuery, imageSize);
		auto normTrain = normalizeKeypoints(kptsTrain, imageSize);

		auto desc0 = queryTensor + kenc->forward(normQuery, scoresQuery);
		auto desc1 = trainTensor + kenc->forward(normTrain, scoresTrain);
		auto gnnOutput = gnn->forward(desc0, desc1);
		auto proj0 = finalProj->forward(gnnOutput.first);
		auto proj1 = finalProj->forward(gnnOutput.second);
		auto scores = torch::einsum("bdn,bdm->bnm", {proj0, proj1}) / std::sqrt(256.0);
		scores = logOptimalTransport(scores, binScore, iterations_);

		auto scoreSlice = scores.index({Slice(), Slice(0, -1), Slice(0, -1)});
		auto max0 = scoreSlice.max(2);
		auto max1 = scoreSlice.max(1);
		auto indices0 = std::get<1>(max0);
		auto indices1 = std::get<1>(max1);
		auto mutual0 = arangeLike(indices0, 1).unsqueeze(0).to(device) == indices1.gather(1, indices0);
		auto mutual1 = arangeLike(indices1, 1).unsqueeze(0).to(device) == indices0.gather(1, indices1);
		auto zero = scores.new_zeros({});
		auto mscores0 = torch::where(mutual0, std::get<0>(max0).exp(), zero);
		auto valid0 = mutual0.logical_and(mscores0 > matchThreshold_);
		indices0 = torch::where(valid0, indices0, torch::full_like(indices0, -1));

		auto indices0Cpu = indices0.squeeze(0).to(torch::kCPU);
		auto scores0Cpu = mscores0.squeeze(0).to(torch::kCPU);
		for(int64_t i=0; i<indices0Cpu.size(0); ++i)
		{
			int trainIndex = (int)indices0Cpu[i].item<int64_t>();
			if(trainIndex >= 0)
			{
				matches.push_back(cv::DMatch((int)i, trainIndex, 1.0f - scores0Cpu[i].item<float>()));
			}
		}
		return matches;
	}

	float matchThreshold_;
	int iterations_;
	KeypointEncoder kenc{nullptr};
	AttentionalGNN gnn{nullptr};
	torch::nn::Conv1d finalProj{nullptr};
	torch::Tensor binScore;
};

SGMatcher::SGMatcher(const std::string & weightsPath, float matchThreshold, int iterations, bool cuda) :
		path_(uReplaceChar(weightsPath, '~', UDirectory::homeDir())),
		matchThreshold_(matchThreshold),
		iterations_(iterations),
		cuda_(cuda)
{
	if(path_.empty())
	{
		UERROR("Native SuperGlue weights path is empty!");
		return;
	}
	if(!UFile::exists(path_))
	{
		UERROR("Native SuperGlue weights path \"%s\" doesn't exist!", path_.c_str());
		return;
	}

	impl_ = std::make_shared<Impl>(matchThreshold_, iterations_);
	std::string loadError;
	const bool legacyPickle = looksLikeLegacyPickle(path_);
	bool loaded = false;
	if(legacyPickle)
	{
		loaded = loadPickledStateDict(impl_, path_, loadError);
	}
	else
	{
		loaded = loadStateDictFromZip(impl_, path_, loadError);
		if(!loaded)
		{
			loaded = loadArchivedStateDict(impl_, path_, loadError);
		}
		if(!loaded)
		{
			loaded = loadPickledStateDict(impl_, path_, loadError);
		}
	}
	if(!loaded)
	{
		if(legacyPickle)
		{
			loadError = "Legacy OrderedDict checkpoints should first be converted with scripts/convert_superglue_weights.py.";
		}
		UERROR("Failed to load native SuperGlue weights from \"%s\": %s", path_.c_str(), loadError.c_str());
		impl_.reset();
		return;
	}

	if(cuda_ && !torch::cuda::is_available())
	{
		UWARN("Cuda option is enabled but torch doesn't have cuda support on this platform, using CPU instead.");
	}
	cuda_ = cuda_ && torch::cuda::is_available();
	impl_->to(torch::Device(cuda_ ? torch::kCUDA : torch::kCPU));
	impl_->eval();
}

SGMatcher::~SGMatcher()
{
}

bool SGMatcher::isValid() const
{
	return impl_.get() != 0;
}

std::vector<cv::DMatch> SGMatcher::match(
		const cv::Mat & descriptorsQuery,
		const cv::Mat & descriptorsTrain,
		const std::vector<cv::KeyPoint> & keypointsQuery,
		const std::vector<cv::KeyPoint> & keypointsTrain,
		const cv::Size & imageSize)
{
	if(!impl_)
	{
		UERROR("Native SuperGlue matcher is not initialized.");
		return std::vector<cv::DMatch>();
	}
	return impl_->match(
			descriptorsQuery,
			descriptorsTrain,
			keypointsQuery,
			keypointsTrain,
			imageSize,
			torch::Device(cuda_ ? torch::kCUDA : torch::kCPU));
}

}
