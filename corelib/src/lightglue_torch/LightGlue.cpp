/*
Copyright (c) 2010-2016, Mathieu Labbe - IntRoLab - Universite de Sherbrooke
All rights reserved.
*/

#include <lightglue_torch/LightGlue.h>
#include <rtabmap/utilite/ULogger.h>
#include <rtabmap/utilite/UDirectory.h>
#include <rtabmap/utilite/UFile.h>
#include <rtabmap/utilite/UConversion.h>

#include <torch/torch.h>
#include <torch/csrc/jit/serialization/import_read.h>
#include <caffe2/serialize/inline_container.h>

#include <algorithm>
#include <cmath>
#include <fstream>

namespace rtabmap
{

namespace
{
using torch::indexing::Slice;

torch::Tensor normalizeKeypoints(const torch::Tensor & keypoints, const cv::Size & imageSize)
{
	auto width = torch::tensor((float)imageSize.width, keypoints.options());
	auto height = torch::tensor((float)imageSize.height, keypoints.options());
	auto size = torch::stack({width, height}, 0).unsqueeze(0);
	auto shift = size / 2.0f;
	auto scale = std::get<0>(size.max(-1)).unsqueeze(-1) / 2.0f;
	return (keypoints - shift.unsqueeze(1)) / scale.unsqueeze(1);
}

torch::Tensor rotateHalf(const torch::Tensor & x)
{
	auto xView = x.view({x.size(0), x.size(1), x.size(2), x.size(3) / 2, 2});
	auto x1 = xView.select(-1, 0);
	auto x2 = xView.select(-1, 1);
	return torch::stack({-x2, x1}, -1).flatten(-2);
}

torch::Tensor applyCachedRotaryEmb(const torch::Tensor & freqs, const torch::Tensor & t)
{
	return t * freqs.select(0, 0) + rotateHalf(t) * freqs.select(0, 1);
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

	try
	{
		if(ivalue.isNone())
		{
			error = "PyTorch C++ deserialization dropped an OrderedDict root. Convert the checkpoint with scripts/convert_lightglue_weights.py and use the converted file.";
			return false;
		}
		if(!ivalue.isGenericDict())
		{
			error = "Pickle root is not a state dict.";
			return false;
		}
	}
	catch(const c10::Error & e)
	{
		error = std::string("PyTorch C++ could not decode a dict-like state root. Convert the checkpoint with scripts/convert_lightglue_weights.py and use the converted file. Original error: ") + e.what_without_backtrace();
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

	auto getTensorKey = [&stateDict](const std::string & moduleKey) -> std::string {
		if(stateDict.find(moduleKey) != stateDict.end())
		{
			return moduleKey;
		}
		std::string upstreamKey = moduleKey;
		for(int i=0; i<32; ++i)
		{
			std::string selfPrefix = uFormat("transformers.%d.self_attn.", i);
			std::string selfUpstream = uFormat("self_attn.%d.", i);
			if(upstreamKey.find(selfPrefix) == 0)
			{
				upstreamKey.replace(0, selfPrefix.size(), selfUpstream);
				break;
			}
			std::string crossPrefix = uFormat("transformers.%d.cross_attn.", i);
			std::string crossUpstream = uFormat("cross_attn.%d.", i);
			if(upstreamKey.find(crossPrefix) == 0)
			{
				upstreamKey.replace(0, crossPrefix.size(), crossUpstream);
				break;
			}
		}
		if(stateDict.find(upstreamKey) != stateDict.end())
		{
			return upstreamKey;
		}
		return std::string();
	};

	torch::NoGradGuard noGradGuard;
	for(auto & item : module->named_parameters(true))
	{
		std::string tensorKey = getTensorKey(item.key());
		if(tensorKey.empty())
		{
			error = uFormat("Missing parameter \"%s\" in state dict.", item.key().c_str());
			return false;
		}
		item.value().copy_(stateDict.at(tensorKey).to(item.value().device(), item.value().dtype()));
	}
	for(auto & item : module->named_buffers(true))
	{
		std::string tensorKey = getTensorKey(item.key());
		if(tensorKey.empty())
		{
			if(item.key() == "confidence_thresholds")
			{
				continue;
			}
			error = uFormat("Missing buffer \"%s\" in state dict.", item.key().c_str());
			return false;
		}
		item.value().copy_(stateDict.at(tensorKey).to(item.value().device(), item.value().dtype()));
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

	try
	{
		if(ivalue.isNone())
		{
			error = "PyTorch C++ deserialization dropped an OrderedDict root. Convert the checkpoint with scripts/convert_lightglue_weights.py and use the converted file.";
			return false;
		}
		if(!ivalue.isGenericDict())
		{
			error = "Archive root is not a state dict.";
			return false;
		}
	}
	catch(const c10::Error & e)
	{
		error = std::string("PyTorch C++ could not decode a dict-like archive root. Convert the checkpoint with scripts/convert_lightglue_weights.py and use the converted file. Original error: ") + e.what_without_backtrace();
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

	auto getTensorKey = [&stateDict](const std::string & moduleKey) -> std::string {
		if(stateDict.find(moduleKey) != stateDict.end())
		{
			return moduleKey;
		}
		std::string upstreamKey = moduleKey;
		for(int i=0; i<32; ++i)
		{
			std::string selfPrefix = uFormat("transformers.%d.self_attn.", i);
			std::string selfUpstream = uFormat("self_attn.%d.", i);
			if(upstreamKey.find(selfPrefix) == 0)
			{
				upstreamKey.replace(0, selfPrefix.size(), selfUpstream);
				break;
			}
			std::string crossPrefix = uFormat("transformers.%d.cross_attn.", i);
			std::string crossUpstream = uFormat("cross_attn.%d.", i);
			if(upstreamKey.find(crossPrefix) == 0)
			{
				upstreamKey.replace(0, crossPrefix.size(), crossUpstream);
				break;
			}
		}
		if(stateDict.find(upstreamKey) != stateDict.end())
		{
			return upstreamKey;
		}
		return std::string();
	};

	torch::NoGradGuard noGradGuard;
	for(auto & item : module->named_parameters(true))
	{
		std::string tensorKey = getTensorKey(item.key());
		if(tensorKey.empty())
		{
			error = uFormat("Missing parameter \"%s\" in archive state dict.", item.key().c_str());
			return false;
		}
		item.value().copy_(stateDict.at(tensorKey).to(item.value().device(), item.value().dtype()));
	}
	for(auto & item : module->named_buffers(true))
	{
		std::string tensorKey = getTensorKey(item.key());
		if(tensorKey.empty())
		{
			if(item.key() == "confidence_thresholds")
			{
				continue;
			}
			error = uFormat("Missing buffer \"%s\" in archive state dict.", item.key().c_str());
			return false;
		}
		item.value().copy_(stateDict.at(tensorKey).to(item.value().device(), item.value().dtype()));
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

struct LearnableFourierPositionalEncodingImpl : torch::nn::Module
{
	LearnableFourierPositionalEncodingImpl(int64_t inputDim, int64_t dim) :
			linear(register_module("Wr", torch::nn::Linear(torch::nn::LinearOptions(inputDim, dim / 2).bias(false))))
	{
		torch::NoGradGuard noGradGuard;
		linear->weight.normal_(0.0, 1.0);
	}

	torch::Tensor forward(const torch::Tensor & x)
	{
		auto projected = linear->forward(x);
		auto cosines = torch::cos(projected);
		auto sines = torch::sin(projected);
		return torch::stack({cosines, sines}, 0).unsqueeze(-3).repeat_interleave(2, -1);
	}

	torch::nn::Linear linear{nullptr};
};
TORCH_MODULE(LearnableFourierPositionalEncoding);

struct TokenConfidenceImpl : torch::nn::Module
{
	TokenConfidenceImpl(int64_t dim) :
			token(register_module("token", torch::nn::Sequential(
					torch::nn::Linear(dim, 1),
					torch::nn::Sigmoid())))
	{
	}

	std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor & desc0, const torch::Tensor & desc1)
	{
		return std::make_pair(
				token->forward(desc0.detach()).squeeze(-1),
				token->forward(desc1.detach()).squeeze(-1));
	}

	torch::nn::Sequential token{nullptr};
};
TORCH_MODULE(TokenConfidence);

struct AttentionImpl : torch::nn::Module
{
	AttentionImpl()
	{
	}

	torch::Tensor forward(
			const torch::Tensor & q,
			const torch::Tensor & k,
			const torch::Tensor & v,
			const torch::Tensor & mask = torch::Tensor())
	{
		if(q.size(-2) == 0 || k.size(-2) == 0)
		{
			return q.new_zeros({q.size(0), q.size(1), q.size(2), v.size(-1)});
		}
		double scale = 1.0 / std::sqrt((double)q.size(-1));
		auto sim = torch::einsum("bhid,bhjd->bhij", {q, k}) * scale;
		if(mask.defined())
		{
			sim = sim.masked_fill(mask.logical_not(), -std::numeric_limits<float>::infinity());
		}
		auto attn = torch::softmax(sim, -1);
		auto out = torch::einsum("bhij,bhjd->bhid", {attn, v});
		return mask.defined() ? out.nan_to_num() : out;
	}
};
TORCH_MODULE(Attention);

struct SelfBlockImpl : torch::nn::Module
{
	SelfBlockImpl(int64_t embedDim, int64_t numHeads) :
			embedDim_(embedDim),
			numHeads_(numHeads),
			headDim_(embedDim / numHeads),
			wqkv(register_module("Wqkv", torch::nn::Linear(embedDim, 3 * embedDim))),
			innerAttn(register_module("inner_attn", Attention())),
			outProj(register_module("out_proj", torch::nn::Linear(embedDim, embedDim))),
			ffn(register_module("ffn", torch::nn::Sequential(
					torch::nn::Linear(2 * embedDim, 2 * embedDim),
					torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{2 * embedDim}).elementwise_affine(true)),
					torch::nn::GELU(),
					torch::nn::Linear(2 * embedDim, embedDim))))
	{
	}

	torch::Tensor forward(
			const torch::Tensor & x,
			const torch::Tensor & encoding,
			const torch::Tensor & mask = torch::Tensor())
	{
		auto qkv = wqkv->forward(x).view({x.size(0), x.size(1), numHeads_, headDim_, 3}).permute({0, 2, 1, 3, 4});
		auto q = applyCachedRotaryEmb(encoding, qkv.select(-1, 0));
		auto k = applyCachedRotaryEmb(encoding, qkv.select(-1, 1));
		auto v = qkv.select(-1, 2);
		auto context = innerAttn->forward(q, k, v, mask);
		auto message = outProj->forward(context.permute({0, 2, 1, 3}).contiguous().view({x.size(0), x.size(1), embedDim_}));
		return x + ffn->forward(torch::cat({x, message}, -1));
	}

	int64_t embedDim_;
	int64_t numHeads_;
	int64_t headDim_;
	torch::nn::Linear wqkv{nullptr};
	Attention innerAttn{nullptr};
	torch::nn::Linear outProj{nullptr};
	torch::nn::Sequential ffn{nullptr};
};
TORCH_MODULE(SelfBlock);

struct CrossBlockImpl : torch::nn::Module
{
	CrossBlockImpl(int64_t embedDim, int64_t numHeads) :
			heads_(numHeads),
			embedDim_(embedDim),
			headDim_(embedDim / numHeads),
			scale_(1.0 / std::sqrt((double)headDim_)),
			toQK(register_module("to_qk", torch::nn::Linear(embedDim, embedDim))),
			toV(register_module("to_v", torch::nn::Linear(embedDim, embedDim))),
			toOut(register_module("to_out", torch::nn::Linear(embedDim, embedDim))),
			ffn(register_module("ffn", torch::nn::Sequential(
					torch::nn::Linear(2 * embedDim, 2 * embedDim),
					torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{2 * embedDim}).elementwise_affine(true)),
					torch::nn::GELU(),
					torch::nn::Linear(2 * embedDim, embedDim))))
	{
	}

	std::pair<torch::Tensor, torch::Tensor> forward(
			const torch::Tensor & x0,
			const torch::Tensor & x1,
			const torch::Tensor & mask = torch::Tensor())
	{
		auto qk0 = toQK->forward(x0).view({x0.size(0), x0.size(1), heads_, headDim_}).permute({0, 2, 1, 3}) * std::sqrt(scale_);
		auto qk1 = toQK->forward(x1).view({x1.size(0), x1.size(1), heads_, headDim_}).permute({0, 2, 1, 3}) * std::sqrt(scale_);
		auto v0 = toV->forward(x0).view({x0.size(0), x0.size(1), heads_, headDim_}).permute({0, 2, 1, 3});
		auto v1 = toV->forward(x1).view({x1.size(0), x1.size(1), heads_, headDim_}).permute({0, 2, 1, 3});
		auto sim = torch::einsum("bhid,bhjd->bhij", {qk0, qk1});
		if(mask.defined())
		{
			sim = sim.masked_fill(mask.logical_not(), -std::numeric_limits<float>::infinity());
		}
		auto attn01 = torch::softmax(sim, -1);
		auto attn10 = torch::softmax(sim.transpose(-2, -1).contiguous(), -1);
		auto m0 = torch::einsum("bhij,bhjd->bhid", {attn01, v1});
		auto m1 = torch::einsum("bhji,bhid->bhjd", {attn10, v0});
		if(mask.defined())
		{
			m0 = m0.nan_to_num();
			m1 = m1.nan_to_num();
		}
		m0 = toOut->forward(m0.permute({0, 2, 1, 3}).contiguous().view({x0.size(0), x0.size(1), embedDim_}));
		m1 = toOut->forward(m1.permute({0, 2, 1, 3}).contiguous().view({x1.size(0), x1.size(1), embedDim_}));
		auto y0 = x0 + ffn->forward(torch::cat({x0, m0}, -1));
		auto y1 = x1 + ffn->forward(torch::cat({x1, m1}, -1));
		return std::make_pair(y0, y1);
	}

	int64_t heads_;
	int64_t embedDim_;
	int64_t headDim_;
	double scale_;
	torch::nn::Linear toQK{nullptr};
	torch::nn::Linear toV{nullptr};
	torch::nn::Linear toOut{nullptr};
	torch::nn::Sequential ffn{nullptr};
};
TORCH_MODULE(CrossBlock);

struct TransformerLayerImpl : torch::nn::Module
{
	TransformerLayerImpl(int64_t embedDim, int64_t numHeads) :
			selfAttn(register_module("self_attn", SelfBlock(embedDim, numHeads))),
			crossAttn(register_module("cross_attn", CrossBlock(embedDim, numHeads)))
	{
	}

	std::pair<torch::Tensor, torch::Tensor> forward(
			const torch::Tensor & desc0,
			const torch::Tensor & desc1,
			const torch::Tensor & encoding0,
			const torch::Tensor & encoding1)
	{
		auto out0 = selfAttn->forward(desc0, encoding0);
		auto out1 = selfAttn->forward(desc1, encoding1);
		return crossAttn->forward(out0, out1);
	}

	SelfBlock selfAttn{nullptr};
	CrossBlock crossAttn{nullptr};
};
TORCH_MODULE(TransformerLayer);

torch::Tensor sigmoidLogDoubleSoftmax(const torch::Tensor & sim, const torch::Tensor & z0, const torch::Tensor & z1)
{
	int64_t b = sim.size(0);
	int64_t m = sim.size(1);
	int64_t n = sim.size(2);
	auto certainties = torch::log_sigmoid(z0) + torch::log_sigmoid(z1).transpose(1, 2);
	auto scores0 = torch::log_softmax(sim, 2);
	auto scores1 = torch::log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2);
	auto scores = sim.new_full({b, m + 1, n + 1}, 0);
	scores.index_put_({Slice(), Slice(0, m), Slice(0, n)}, scores0 + scores1 + certainties);
	scores.index_put_({Slice(), Slice(0, m), -1}, torch::log_sigmoid(-z0.squeeze(-1)));
	scores.index_put_({Slice(), -1, Slice(0, n)}, torch::log_sigmoid(-z1.squeeze(-1)));
	return scores;
}

struct MatchAssignmentImpl : torch::nn::Module
{
	MatchAssignmentImpl(int64_t dim) :
			matchability(register_module("matchability", torch::nn::Linear(dim, 1))),
			finalProj(register_module("final_proj", torch::nn::Linear(dim, dim))),
			dim_(dim)
	{
	}

	std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor & desc0, const torch::Tensor & desc1)
	{
		auto mdesc0 = finalProj->forward(desc0);
		auto mdesc1 = finalProj->forward(desc1);
		double d = (double)dim_;
		mdesc0 = mdesc0 / std::pow(d, 0.25);
		mdesc1 = mdesc1 / std::pow(d, 0.25);
		auto sim = torch::einsum("bmd,bnd->bmn", {mdesc0, mdesc1});
		auto z0 = matchability->forward(desc0);
		auto z1 = matchability->forward(desc1);
		return std::make_pair(sigmoidLogDoubleSoftmax(sim, z0, z1), sim);
	}

	torch::Tensor getMatchability(const torch::Tensor & desc)
	{
		return torch::sigmoid(matchability->forward(desc)).squeeze(-1);
	}

	torch::nn::Linear matchability{nullptr};
	torch::nn::Linear finalProj{nullptr};
	int64_t dim_;
};
TORCH_MODULE(MatchAssignment);

void filterMatches(
		const torch::Tensor & scores,
		float threshold,
		torch::Tensor & m0,
		torch::Tensor & m1,
		torch::Tensor & mscores0,
		torch::Tensor & mscores1)
{
	auto scoreSlice = scores.index({Slice(), Slice(0, -1), Slice(0, -1)});
	auto max0 = scoreSlice.max(2);
	auto max1 = scoreSlice.max(1);
	m0 = std::get<1>(max0);
	m1 = std::get<1>(max1);
	auto indices0 = torch::arange(m0.size(1), m0.options()).unsqueeze(0);
	auto indices1 = torch::arange(m1.size(1), m1.options()).unsqueeze(0);
	auto mutual0 = indices0 == m1.gather(1, m0);
	auto mutual1 = indices1 == m0.gather(1, m1);
	auto max0Exp = std::get<0>(max0).exp();
	auto zero = torch::zeros({}, max0Exp.options());
	mscores0 = torch::where(mutual0, max0Exp, zero);
	mscores1 = torch::where(mutual1, mscores0.gather(1, m1), zero);
	auto valid0 = mutual0.logical_and(mscores0 > threshold);
	auto valid1 = mutual1.logical_and(valid0.gather(1, m1));
	m0 = torch::where(valid0, m0, torch::full_like(m0, -1));
	m1 = torch::where(valid1, m1, torch::full_like(m1, -1));
}

} // namespace

class LGMatcher::Impl : public torch::nn::Module
{
public:
	Impl(float filterThreshold, int nLayers, float depthConfidence, float widthConfidence) :
			filterThreshold_(filterThreshold),
			nLayers_(nLayers),
			depthConfidence_(depthConfidence),
			widthConfidence_(widthConfidence),
			inputDim_(256),
			descriptorDim_(256),
			numHeads_(4),
			inputProj(register_module("input_proj", torch::nn::Identity())),
			posenc(register_module("posenc", LearnableFourierPositionalEncoding(2, descriptorDim_ / numHeads_))),
			transformers(register_module("transformers", torch::nn::ModuleList())),
			logAssignment(register_module("log_assignment", torch::nn::ModuleList())),
			tokenConfidence(register_module("token_confidence", torch::nn::ModuleList()))
	{
		for(int i=0; i<nLayers_; ++i)
		{
			transformers->push_back(TransformerLayer(descriptorDim_, numHeads_));
			logAssignment->push_back(MatchAssignment(descriptorDim_));
			if(i < nLayers_ - 1)
			{
				tokenConfidence->push_back(TokenConfidence(descriptorDim_));
			}
		}
		auto confidenceThresholds = torch::empty({nLayers_}, torch::TensorOptions().dtype(torch::kFloat32));
		for(int i=0; i<nLayers_; ++i)
		{
			confidenceThresholds[i] = confidenceThreshold(i);
		}
		register_buffer("confidence_thresholds", confidenceThresholds);
	}

	float confidenceThreshold(int layerIndex) const
	{
		double threshold = 0.8 + 0.1 * std::exp(-4.0 * (double)layerIndex / (double)nLayers_);
		return (float)std::max(0.0, std::min(1.0, threshold));
	}

	torch::Tensor getPruningMask(
			const torch::Tensor & confidences,
			const torch::Tensor & scores,
			int layerIndex) const
	{
		auto keep = scores > (1.0f - widthConfidence_);
		if(confidences.defined())
		{
			keep = keep.logical_or(confidences <= this->named_buffers()["confidence_thresholds"][layerIndex]);
		}
		return keep;
	}

	bool checkIfStop(
			const torch::Tensor & confidences0,
			const torch::Tensor & confidences1,
			int layerIndex,
			int numPoints) const
	{
		auto confidences = torch::cat({confidences0, confidences1}, -1);
		auto threshold = this->named_buffers()["confidence_thresholds"][layerIndex];
		auto ratioConfident = 1.0 - (confidences < threshold).to(torch::kFloat32).sum().item<float>() / (float)numPoints;
		return ratioConfident > depthConfidence_;
	}

	int pruningMinKpts(const torch::Device & device) const
	{
		return device.is_cuda() ? 1024 : -1;
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
			UERROR("Native LightGlue supports only float descriptors.");
			return matches;
		}
		if(descriptorsQuery.cols != inputDim_ || descriptorsTrain.cols != inputDim_)
		{
			UERROR("Native LightGlue expects 256-dimensional descriptors (query=%d train=%d).", descriptorsQuery.cols, descriptorsTrain.cols);
			return matches;
		}
		if(descriptorsQuery.rows != (int)keypointsQuery.size() || descriptorsTrain.rows != (int)keypointsTrain.size())
		{
			UERROR("Descriptors and keypoints size mismatch for native LightGlue.");
			return matches;
		}
		if(imageSize.width <= 0 || imageSize.height <= 0)
		{
			UERROR("Invalid image size for native LightGlue.");
			return matches;
		}

		torch::NoGradGuard noGradGuard;

		auto desc0 = torch::from_blob((void*)descriptorsQuery.data, {1, descriptorsQuery.rows, descriptorsQuery.cols}, torch::TensorOptions().dtype(torch::kFloat32)).to(device).clone();
		auto desc1 = torch::from_blob((void*)descriptorsTrain.data, {1, descriptorsTrain.rows, descriptorsTrain.cols}, torch::TensorOptions().dtype(torch::kFloat32)).to(device).clone();

		std::vector<float> kpts0Vec(keypointsQuery.size() * 2);
		std::vector<float> kpts1Vec(keypointsTrain.size() * 2);
		for(size_t i=0; i<keypointsQuery.size(); ++i)
		{
			kpts0Vec[i * 2] = keypointsQuery[i].pt.x;
			kpts0Vec[i * 2 + 1] = keypointsQuery[i].pt.y;
		}
		for(size_t i=0; i<keypointsTrain.size(); ++i)
		{
			kpts1Vec[i * 2] = keypointsTrain[i].pt.x;
			kpts1Vec[i * 2 + 1] = keypointsTrain[i].pt.y;
		}
		auto kpts0 = torch::from_blob(kpts0Vec.data(), {1, (long)keypointsQuery.size(), 2}, torch::TensorOptions().dtype(torch::kFloat32)).to(device).clone();
		auto kpts1 = torch::from_blob(kpts1Vec.data(), {1, (long)keypointsTrain.size(), 2}, torch::TensorOptions().dtype(torch::kFloat32)).to(device).clone();
		kpts0 = normalizeKeypoints(kpts0, imageSize);
		kpts1 = normalizeKeypoints(kpts1, imageSize);

		desc0 = inputProj->forward(desc0);
		desc1 = inputProj->forward(desc1);
		auto encoding0 = posenc->forward(kpts0);
		auto encoding1 = posenc->forward(kpts1);

		const int m = descriptorsQuery.rows;
		const int n = descriptorsTrain.rows;
		const bool doEarlyStop = depthConfidence_ > 0.0f;
		const bool doPointPruning = widthConfidence_ > 0.0f;
		const int pruningTh = pruningMinKpts(device);
		auto ind0 = torch::arange(0, m, torch::TensorOptions().dtype(torch::kLong).device(device)).unsqueeze(0);
		auto ind1 = torch::arange(0, n, torch::TensorOptions().dtype(torch::kLong).device(device)).unsqueeze(0);

		torch::Tensor token0;
		torch::Tensor token1;
		int stopLayer = 0;
		for(int i=0; i<nLayers_; ++i)
		{
			stopLayer = i;
			if(desc0.size(1) == 0 || desc1.size(1) == 0)
			{
				break;
			}
			auto layer = transformers->ptr<TransformerLayerImpl>(i);
			auto out = layer->forward(desc0, desc1, encoding0, encoding1);
			desc0 = out.first;
			desc1 = out.second;
			if(i == nLayers_ - 1)
			{
				continue;
			}
			if(doEarlyStop)
			{
				auto confidenceLayer = tokenConfidence->ptr<TokenConfidenceImpl>(i);
				auto tokens = confidenceLayer->forward(desc0, desc1);
				token0 = tokens.first;
				token1 = tokens.second;
				if(checkIfStop(token0.index({Slice(), Slice(0, m)}), token1.index({Slice(), Slice(0, n)}), i, m + n))
				{
					break;
				}
			}
			if(doPointPruning && pruningTh >= 0 && desc0.size(1) > pruningTh)
			{
				auto scoreLayer = logAssignment->ptr<MatchAssignmentImpl>(i);
				auto scores0 = scoreLayer->getMatchability(desc0);
				auto pruneMask0 = getPruningMask(token0, scores0, i);
				auto keep0 = torch::nonzero(pruneMask0.squeeze(0)).squeeze(1);
				ind0 = ind0.index_select(1, keep0);
				desc0 = desc0.index_select(1, keep0);
				encoding0 = encoding0.index_select(-2, keep0);
			}
			if(doPointPruning && pruningTh >= 0 && desc1.size(1) > pruningTh)
			{
				auto scoreLayer = logAssignment->ptr<MatchAssignmentImpl>(i);
				auto scores1 = scoreLayer->getMatchability(desc1);
				auto pruneMask1 = getPruningMask(token1, scores1, i);
				auto keep1 = torch::nonzero(pruneMask1.squeeze(0)).squeeze(1);
				ind1 = ind1.index_select(1, keep1);
				desc1 = desc1.index_select(1, keep1);
				encoding1 = encoding1.index_select(-2, keep1);
			}
		}

		if(desc0.size(1) == 0 || desc1.size(1) == 0)
		{
			return matches;
		}

		auto assignment = logAssignment->ptr<MatchAssignmentImpl>(stopLayer);
		torch::Tensor scores;
		torch::Tensor sim;
		std::tie(scores, sim) = assignment->forward(desc0, desc1);
		torch::Tensor m0;
		torch::Tensor m1;
		torch::Tensor mscores0;
		torch::Tensor mscores1;
		filterMatches(scores, filterThreshold_, m0, m1, mscores0, mscores1);

		auto m0Cpu = m0.squeeze(0).to(torch::kCPU);
		auto scoresCpu = mscores0.squeeze(0).to(torch::kCPU);
		auto ind0Cpu = ind0.squeeze(0).to(torch::kCPU);
		auto ind1Cpu = ind1.squeeze(0).to(torch::kCPU);
		for(int64_t i=0; i<m0Cpu.size(0); ++i)
		{
			int64_t trainIndex = m0Cpu[i].item<int64_t>();
			if(trainIndex >= 0)
			{
				int queryIndex = (int)ind0Cpu[i].item<int64_t>();
				int mappedTrainIndex = (int)ind1Cpu[trainIndex].item<int64_t>();
				matches.push_back(cv::DMatch(queryIndex, mappedTrainIndex, 1.0f - scoresCpu[i].item<float>()));
			}
		}
		return matches;
	}

	float filterThreshold_;
	int nLayers_;
	float depthConfidence_;
	float widthConfidence_;
	int inputDim_;
	int descriptorDim_;
	int numHeads_;
	torch::nn::Identity inputProj{nullptr};
	LearnableFourierPositionalEncoding posenc{nullptr};
	torch::nn::ModuleList transformers{nullptr};
	torch::nn::ModuleList logAssignment{nullptr};
	torch::nn::ModuleList tokenConfidence{nullptr};
};

LGMatcher::LGMatcher(const std::string & weightsPath, float filterThreshold, int nLayers, float depthConfidence, float widthConfidence, bool cuda) :
		path_(uReplaceChar(weightsPath, '~', UDirectory::homeDir())),
		filterThreshold_(filterThreshold),
		nLayers_(nLayers),
		depthConfidence_(depthConfidence),
		widthConfidence_(widthConfidence),
		cuda_(cuda)
{
	if(path_.empty())
	{
		UERROR("Native LightGlue weights path is empty!");
		return;
	}
	if(!UFile::exists(path_))
	{
		UERROR("Native LightGlue weights path \"%s\" doesn't exist!", path_.c_str());
		return;
	}

	impl_ = std::make_shared<Impl>(filterThreshold_, nLayers_, depthConfidence_, widthConfidence_);
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
			loadError = "Legacy OrderedDict checkpoints should first be converted with scripts/convert_lightglue_weights.py.";
		}
		else if(loadError.find("isGenericDict()") != std::string::npos ||
				loadError.find("Expected GenericDict but got None") != std::string::npos)
		{
			loadError = "OrderedDict-style LightGlue checkpoints should first be converted with scripts/convert_lightglue_weights.py.";
		}
		UERROR("Failed to load native LightGlue weights from \"%s\": %s", path_.c_str(), loadError.c_str());
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

LGMatcher::~LGMatcher()
{
}

bool LGMatcher::isValid() const
{
	return impl_.get() != 0;
}

std::vector<cv::DMatch> LGMatcher::match(
		const cv::Mat & descriptorsQuery,
		const cv::Mat & descriptorsTrain,
		const std::vector<cv::KeyPoint> & keypointsQuery,
		const std::vector<cv::KeyPoint> & keypointsTrain,
		const cv::Size & imageSize)
{
	if(!impl_)
	{
		UERROR("Native LightGlue matcher is not initialized.");
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
