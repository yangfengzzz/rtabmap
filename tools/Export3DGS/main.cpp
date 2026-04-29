/*
Copyright (c) 2010-2016, Mathieu Labbe - IntRoLab - Universite de Sherbrooke
All rights reserved.
*/

#include <rtabmap/core/DBDriver.h>
#include <rtabmap/core/Optimizer.h>
#include <rtabmap/core/Rtabmap.h>
#include <rtabmap/core/util3d.h>
#include <rtabmap/utilite/UConversion.h>
#include <rtabmap/utilite/UDirectory.h>
#include <rtabmap/utilite/UFile.h>
#include <rtabmap/utilite/ULogger.h>
#include <rtabmap/utilite/UMath.h>
#include <rtabmap/utilite/UStl.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace rtabmap;

namespace {

struct Options
{
	std::string dbPath;
	std::string outputDir;
	std::string imageFormat = "png";
	double maxLinearSpeed = 0.0;
	double maxAngularSpeed = 0.0;
	double laplacianThreshold = 0.0;
	bool exportColmap = true;
	bool exportNerfstudio = true;
};

struct CameraInfo
{
	int id = 0;
	CameraModel model;
};

struct FrameInfo
{
	int imageId = 0;
	int nodeId = 0;
	int cameraId = 0;
	std::string fileName;
	std::string relativePath;
	CameraModel model;
	Transform cameraPoseOpenCV; // camera-to-world
	Transform worldToCameraOpenCV;
	cv::Mat rectifiedImage;
};

struct Observation
{
	double x = 0.0;
	double y = 0.0;
	int pointId = -1;
};

struct PointTrack
{
	cv::Point3f position;
	cv::Vec3b color = cv::Vec3b(255, 255, 255);
	double error = 0.0;
	std::vector<std::pair<int, int> > track; // image id, point2D idx
};

std::string joinPath(const std::string & a, const std::string & b)
{
	if(a.empty())
	{
		return b;
	}
	if(a.back() == '/' || a.back() == '\\')
	{
		return a + b;
	}
	return a + UDirectory::separator() + b;
}

std::string boolString(bool value)
{
	return value ? "true" : "false";
}

void showUsage()
{
	printf("\nUsage:\n"
		"rtabmap-export3dgs_dataset [options] --db database.db\n"
		"Options:\n"
		"    --db path                 Input RTAB-Map database.\n"
		"    --output_dir path         Output directory (default: <db>_3dgs).\n"
		"    --image_format png|jpg    Export image format (default: png).\n"
		"    --max_linear_speed value  Ignore frames over this linear speed in m/s (default: 0, disabled).\n"
		"    --max_angular_speed value Ignore frames over this angular speed in rad/s (default: 0, disabled).\n"
		"    --laplacian_threshold v   Ignore blurry frames below this Laplacian variance (default: 0, disabled).\n"
		"    --export_colmap bool      Write cameras.txt/images.txt/points3D.txt (default: true).\n"
		"    --export_nerfstudio bool  Write transforms.json (default: true).\n");
}

bool parseBool(const char * value, bool & output)
{
	std::string v = uToLowerCase(value);
	if(v == "1" || v == "true" || v == "yes" || v == "on")
	{
		output = true;
		return true;
	}
	if(v == "0" || v == "false" || v == "no" || v == "off")
	{
		output = false;
		return true;
	}
	return false;
}

bool parseArgs(int argc, char * argv[], Options & options)
{
	for(int i = 1; i < argc; ++i)
	{
		std::string arg = argv[i];
		if(arg == "--help" || arg == "-h")
		{
			showUsage();
			return false;
		}
		else if(arg == "--db" && i + 1 < argc)
		{
			options.dbPath = argv[++i];
		}
		else if(arg == "--output_dir" && i + 1 < argc)
		{
			options.outputDir = argv[++i];
		}
		else if(arg == "--image_format" && i + 1 < argc)
		{
			options.imageFormat = uToLowerCase(argv[++i]);
		}
		else if(arg == "--max_linear_speed" && i + 1 < argc)
		{
			options.maxLinearSpeed = std::atof(argv[++i]);
		}
		else if(arg == "--max_angular_speed" && i + 1 < argc)
		{
			options.maxAngularSpeed = std::atof(argv[++i]);
		}
		else if(arg == "--laplacian_threshold" && i + 1 < argc)
		{
			options.laplacianThreshold = std::atof(argv[++i]);
		}
		else if(arg == "--export_colmap" && i + 1 < argc)
		{
			if(!parseBool(argv[++i], options.exportColmap))
			{
				UERROR("Invalid boolean for --export_colmap: %s", argv[i]);
				return false;
			}
		}
		else if(arg == "--export_nerfstudio" && i + 1 < argc)
		{
			if(!parseBool(argv[++i], options.exportNerfstudio))
			{
				UERROR("Invalid boolean for --export_nerfstudio: %s", argv[i]);
				return false;
			}
		}
		else
		{
			UERROR("Unknown argument: %s", arg.c_str());
			showUsage();
			return false;
		}
	}
	if(options.dbPath.empty())
	{
		UERROR("--db is required.");
		showUsage();
		return false;
	}
	if(options.imageFormat != "png" && options.imageFormat != "jpg" && options.imageFormat != "jpeg")
	{
		UERROR("Unsupported image format: %s", options.imageFormat.c_str());
		return false;
	}
	if(!options.exportColmap && !options.exportNerfstudio)
	{
		UERROR("At least one of --export_colmap or --export_nerfstudio must be true.");
		return false;
	}
	return true;
}

double laplacianVariance(const cv::Mat & image)
{
	cv::Mat gray;
	if(image.channels() == 3)
	{
		cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
	}
	else
	{
		gray = image;
	}
	cv::Mat laplacian;
	cv::Laplacian(gray, laplacian, CV_16S);
	cv::Mat mean, stddev;
	cv::meanStdDev(laplacian, mean, stddev);
	double sigma = stddev.at<double>(0);
	return sigma * sigma;
}

cv::Point2f rectifyPoint(const cv::Point2f & pt, const CameraModel & model)
{
	if(!model.isValidForRectification())
	{
		return pt;
	}
	std::vector<cv::Point2f> src(1, pt);
	std::vector<cv::Point2f> dst;
	cv::undistortPoints(src, dst, model.K_raw(), model.D_raw(), model.R(), model.P());
	return dst.front();
}

cv::Point3f transformPointSafe(const cv::Point3f & p, const Transform & t)
{
	return util3d::transformPoint(p, t);
}

cv::Vec3b sampleColor(const cv::Mat & image, const cv::Point2f & pt)
{
	if(image.empty())
	{
		return cv::Vec3b(255, 255, 255);
	}
	int x = std::max(0, std::min((int)std::lround(pt.x), image.cols - 1));
	int y = std::max(0, std::min((int)std::lround(pt.y), image.rows - 1));
	if(image.channels() == 3)
	{
		return image.at<cv::Vec3b>(y, x);
	}
	unsigned char pixel = image.at<unsigned char>(y, x);
	return cv::Vec3b(pixel, pixel, pixel);
}

std::string cameraKey(const CameraModel & model)
{
	std::ostringstream oss;
	oss << std::fixed << std::setprecision(6)
		<< model.imageWidth() << "_"
		<< model.imageHeight() << "_"
		<< model.fx() << "_"
		<< model.fy() << "_"
		<< model.cx() << "_"
		<< model.cy();
	return oss.str();
}

bool createDirectory(const std::string & path)
{
	return UDirectory::exists(path) || UDirectory::makeDir(path);
}

cv::Mat to4x4(const Transform & t)
{
	cv::Mat m = cv::Mat::eye(4, 4, CV_64FC1);
	for(int r = 0; r < 3; ++r)
	{
		for(int c = 0; c < 4; ++c)
		{
			m.at<double>(r, c) = (double)t.data()[r * 4 + c];
		}
	}
	return m;
}

double computeReprojectionError(
		const cv::Point3f & pointWorld,
		const FrameInfo & frame,
		const cv::Point2f & observed)
{
	cv::Point3f pCam = transformPointSafe(pointWorld, frame.worldToCameraOpenCV);
	if(pCam.z <= 0.0f)
	{
		return 1e6;
	}
	float u = 0.0f;
	float v = 0.0f;
	frame.model.reproject(pCam.x, pCam.y, pCam.z, u, v);
	double dx = (double)u - observed.x;
	double dy = (double)v - observed.y;
	return std::sqrt(dx * dx + dy * dy);
}

bool writePly(const std::string & path, const std::map<int, PointTrack> & points)
{
	std::ofstream out(path.c_str());
	if(!out.is_open())
	{
		return false;
	}
	out << "ply\n";
	out << "format ascii 1.0\n";
	out << "element vertex " << points.size() << "\n";
	out << "property float x\n";
	out << "property float y\n";
	out << "property float z\n";
	out << "property uchar red\n";
	out << "property uchar green\n";
	out << "property uchar blue\n";
	out << "end_header\n";
	for(std::map<int, PointTrack>::const_iterator iter = points.begin(); iter != points.end(); ++iter)
	{
		const PointTrack & point = iter->second;
		out << point.position.x << " " << point.position.y << " " << point.position.z << " "
			<< (int)point.color[2] << " " << (int)point.color[1] << " " << (int)point.color[0] << "\n";
	}
	return true;
}

} // namespace

int main(int argc, char * argv[])
{
	ULogger::setType(ULogger::kTypeConsole);
	ULogger::setLevel(ULogger::kError);

	Options options;
	if(!parseArgs(argc, argv, options))
	{
		return 1;
	}

	options.dbPath = uReplaceChar(options.dbPath, '~', UDirectory::homeDir());
	if(!UFile::exists(options.dbPath))
	{
		UERROR("Database doesn't exist: %s", options.dbPath.c_str());
		return 1;
	}

	if(options.outputDir.empty())
	{
		std::string baseName = uSplit(UFile::getName(options.dbPath), '.').front();
		options.outputDir = joinPath(UDirectory::getDir(options.dbPath), baseName + "_3dgs");
	}
	options.outputDir = uReplaceChar(options.outputDir, '~', UDirectory::homeDir());

	const std::string imagesDir = joinPath(options.outputDir, "images");
	const std::string sparseDir = joinPath(joinPath(options.outputDir, "sparse"), "0");
	if(!createDirectory(options.outputDir) || !createDirectory(imagesDir) || (!createDirectory(joinPath(options.outputDir, "sparse")) || !createDirectory(sparseDir)))
	{
		UERROR("Failed to create output directories under %s", options.outputDir.c_str());
		return 1;
	}

	ParametersMap parameters;
	DBDriver * driver = DBDriver::create();
	if(!driver->openConnection(options.dbPath))
	{
		UERROR("Cannot open database %s", options.dbPath.c_str());
		delete driver;
		return 1;
	}
	parameters = driver->getLastParameters();
	driver->closeConnection(false);
	delete driver;

	Rtabmap rtabmap;
	rtabmap.init(parameters, options.dbPath);

	std::map<int, Transform> poses;
	std::multimap<int, Link> links;
	std::map<int, Signature> signatures;
	rtabmap.getGraph(poses, links, true, true, &signatures, true, false, false, false, true, false);
	rtabmap.close();

	if(poses.empty() || signatures.empty())
	{
		UERROR("No graph data was loaded from %s", options.dbPath.c_str());
		return 1;
	}

	std::map<int, Transform> retainedPoses;
	std::map<int, Signature> retainedSignatures;
	std::set<int> retainedIds;
	int ignoredFrames = 0;
	for(std::map<int, Transform>::const_iterator iter = poses.begin(); iter != poses.end(); ++iter)
	{
		if(iter->first <= 0)
		{
			continue;
		}
		if(!uContains(signatures, iter->first))
		{
			continue;
		}
		const Signature & s = signatures.at(iter->first);
		if(s.sensorData().cameraModels().size() != 1 || !s.sensorData().stereoCameraModels().empty())
		{
			UERROR("Node %d is not a single RGB-D camera frame. V1 only supports one camera model per node.", iter->first);
			return 1;
		}
		const CameraModel & model = s.sensorData().cameraModels()[0];
		if(!model.isValidForProjection())
		{
			UERROR("Node %d has missing or invalid calibration for PINHOLE export.", iter->first);
			return 1;
		}

		cv::Mat image = s.sensorData().imageRaw();
		if(image.empty())
		{
			s.sensorData().uncompressDataConst(&image, 0, 0, 0);
		}
		if(image.empty())
		{
			UERROR("Node %d has no RGB image data stored in the database.", iter->first);
			return 1;
		}

		bool ignore = false;
		const std::vector<float> & velocity = s.getVelocity();
		if((options.maxLinearSpeed > 0.0 || options.maxAngularSpeed > 0.0) && velocity.size() == 6)
		{
			float transVel = uMax3(std::fabs(velocity[0]), std::fabs(velocity[1]), std::fabs(velocity[2]));
			float rotVel = uMax3(std::fabs(velocity[3]), std::fabs(velocity[4]), std::fabs(velocity[5]));
			if(options.maxLinearSpeed > 0.0 && transVel > options.maxLinearSpeed)
			{
				ignore = true;
			}
			if(options.maxAngularSpeed > 0.0 && rotVel > options.maxAngularSpeed)
			{
				ignore = true;
			}
		}
		if(!ignore && options.laplacianThreshold > 0.0)
		{
			double variance = laplacianVariance(image);
			if(variance < options.laplacianThreshold)
			{
				ignore = true;
			}
		}
		if(ignore)
		{
			++ignoredFrames;
			continue;
		}
		retainedPoses.insert(*iter);
		retainedSignatures.insert(std::make_pair(iter->first, s));
		retainedIds.insert(iter->first);
	}

	if(retainedPoses.empty())
	{
		UERROR("No frames remained after filtering.");
		return 1;
	}

	std::multimap<int, Link> retainedLinks;
	for(std::multimap<int, Link>::const_iterator iter = links.begin(); iter != links.end(); ++iter)
	{
		if(retainedIds.find(iter->second.from()) != retainedIds.end() &&
		   retainedIds.find(iter->second.to()) != retainedIds.end())
		{
			retainedLinks.insert(*iter);
		}
	}

	Optimizer::Type optimizerType = Optimizer::kTypeUndef;
	if(Optimizer::isAvailable(Optimizer::kTypeG2O))
	{
		optimizerType = Optimizer::kTypeG2O;
	}
	else if(Optimizer::isAvailable(Optimizer::kTypeCVSBA))
	{
		optimizerType = Optimizer::kTypeCVSBA;
	}
	else if(Optimizer::isAvailable(Optimizer::kTypeCeres))
	{
		optimizerType = Optimizer::kTypeCeres;
	}
	else if(Optimizer::isAvailable(Optimizer::kTypeTORO))
	{
		optimizerType = Optimizer::kTypeTORO;
	}
	if(optimizerType == Optimizer::kTypeUndef)
	{
		UERROR("No BA-capable optimizer is available to compute sparse correspondences.");
		return 1;
	}

	std::shared_ptr<Optimizer> optimizer(Optimizer::create(optimizerType, parameters));
	std::map<int, cv::Point3f> points3DMap;
	std::map<int, std::map<int, FeatureBA> > wordReferences;
	optimizer->computeBACorrespondences(retainedPoses, retainedLinks, retainedSignatures, points3DMap, wordReferences, false, false, parameters);
	if(points3DMap.empty() || wordReferences.empty())
	{
		UERROR("No valid sparse landmark correspondences were computed for export.");
		return 1;
	}

	std::map<std::string, CameraInfo> camerasByKey;
	std::map<int, FrameInfo> framesByNodeId;
	std::map<int, FrameInfo> framesByImageId;
	int nextCameraId = 1;
	int nextImageId = 1;
	for(std::map<int, Signature>::const_iterator iter = retainedSignatures.begin(); iter != retainedSignatures.end(); ++iter)
	{
		const Signature & s = iter->second;
		const CameraModel & model = s.sensorData().cameraModels()[0];
		std::string key = cameraKey(model);
		if(!uContains(camerasByKey, key))
		{
			CameraInfo info;
			info.id = nextCameraId++;
			info.model = model;
			camerasByKey.insert(std::make_pair(key, info));
		}

		cv::Mat image = s.sensorData().imageRaw();
		if(image.empty())
		{
			s.sensorData().uncompressDataConst(&image, 0, 0, 0);
		}
		if(image.empty())
		{
			UERROR("Node %d is missing RGB image data during export.", iter->first);
			return 1;
		}
		cv::Mat rectified = model.isValidForRectification() ? model.rectifyImage(image) : image.clone();
		if(rectified.empty())
		{
			UERROR("Failed to rectify image for node %d.", iter->first);
			return 1;
		}

		FrameInfo frame;
		frame.imageId = nextImageId++;
		frame.nodeId = iter->first;
		frame.cameraId = camerasByKey.at(key).id;
		frame.fileName = uFormat("%06d.%s", frame.imageId, options.imageFormat.c_str());
		frame.relativePath = std::string("images") + UDirectory::separator() + frame.fileName;
		frame.model = model;
		frame.rectifiedImage = rectified;
		frame.cameraPoseOpenCV = retainedPoses.at(iter->first) * model.localTransform() * CameraModel::opticalRotation().inverse();
		frame.worldToCameraOpenCV = frame.cameraPoseOpenCV.inverse();

		std::string outputImagePath = joinPath(imagesDir, frame.fileName);
		if(!cv::imwrite(outputImagePath, rectified))
		{
			UERROR("Failed to save image %s", outputImagePath.c_str());
			return 1;
		}
		framesByNodeId.insert(std::make_pair(iter->first, frame));
		framesByImageId.insert(std::make_pair(frame.imageId, frame));
	}

	std::map<int, std::vector<Observation> > imageObservations;
	std::map<int, PointTrack> exportedPoints;
	int nextPointId = 1;
	for(std::map<int, cv::Point3f>::const_iterator pter = points3DMap.begin(); pter != points3DMap.end(); ++pter)
	{
		if(!uContains(wordReferences, pter->first))
		{
			continue;
		}
		const std::map<int, FeatureBA> & refs = wordReferences.at(pter->first);
		std::vector<std::pair<int, cv::Point2f> > pointRefs;
		pointRefs.reserve(refs.size());
		for(std::map<int, FeatureBA>::const_iterator rter = refs.begin(); rter != refs.end(); ++rter)
		{
			if(!uContains(framesByNodeId, rter->first))
			{
				continue;
			}
			const FrameInfo & frame = framesByNodeId.at(rter->first);
			cv::Point2f rectifiedPt = rectifyPoint(rter->second.kpt.pt, frame.model);
			if(rectifiedPt.x < 0.0f || rectifiedPt.y < 0.0f ||
			   rectifiedPt.x >= frame.rectifiedImage.cols || rectifiedPt.y >= frame.rectifiedImage.rows)
			{
				continue;
			}
			pointRefs.push_back(std::make_pair(rter->first, rectifiedPt));
		}
		if(pointRefs.size() < 2)
		{
			continue;
		}

		PointTrack point;
		point.position = pter->second;
		point.color = sampleColor(framesByNodeId.at(pointRefs.front().first).rectifiedImage, pointRefs.front().second);
		double totalError = 0.0;
		for(size_t i = 0; i < pointRefs.size(); ++i)
		{
			const FrameInfo & frame = framesByNodeId.at(pointRefs[i].first);
			Observation obs;
			obs.x = pointRefs[i].second.x;
			obs.y = pointRefs[i].second.y;
			obs.pointId = nextPointId;
			std::vector<Observation> & imagePoints = imageObservations[frame.imageId];
			int point2DIndex = (int)imagePoints.size();
			imagePoints.push_back(obs);
			point.track.push_back(std::make_pair(frame.imageId, point2DIndex));
			totalError += computeReprojectionError(point.position, frame, pointRefs[i].second);
		}
		point.error = totalError / double(pointRefs.size());
		exportedPoints.insert(std::make_pair(nextPointId++, point));
	}

	if(exportedPoints.empty())
	{
		UERROR("No sparse points survived frame filtering and track generation.");
		return 1;
	}

	if(options.exportColmap)
	{
		std::ofstream camerasFile(joinPath(sparseDir, "cameras.txt").c_str());
		std::ofstream imagesFile(joinPath(sparseDir, "images.txt").c_str());
		std::ofstream pointsFile(joinPath(sparseDir, "points3D.txt").c_str());
		if(!camerasFile.is_open() || !imagesFile.is_open() || !pointsFile.is_open())
		{
			UERROR("Failed to open COLMAP output files under %s", sparseDir.c_str());
			return 1;
		}

		camerasFile << "# Camera list with one line of data per camera:\n";
		camerasFile << "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n";
		for(std::map<std::string, CameraInfo>::const_iterator iter = camerasByKey.begin(); iter != camerasByKey.end(); ++iter)
		{
			const CameraModel & model = iter->second.model;
			camerasFile << iter->second.id << " PINHOLE "
				<< model.imageWidth() << " " << model.imageHeight() << " "
				<< std::setprecision(17) << model.fx() << " " << model.fy() << " "
				<< model.cx() << " " << model.cy() << "\n";
		}

		imagesFile << "# Image list with two lines of data per image:\n";
		imagesFile << "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n";
		imagesFile << "# POINTS2D[] as (X, Y, POINT3D_ID)\n";
		for(std::map<int, FrameInfo>::const_iterator iter = framesByImageId.begin(); iter != framesByImageId.end(); ++iter)
		{
			const FrameInfo & frame = iter->second;
			Eigen::Quaternionf q = frame.worldToCameraOpenCV.getQuaternionf();
			imagesFile << frame.imageId << " "
				<< std::setprecision(17)
				<< q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " "
				<< frame.worldToCameraOpenCV.x() << " "
				<< frame.worldToCameraOpenCV.y() << " "
				<< frame.worldToCameraOpenCV.z() << " "
				<< frame.cameraId << " " << frame.fileName << "\n";
			const std::vector<Observation> & observations = imageObservations[frame.imageId];
			for(size_t i = 0; i < observations.size(); ++i)
			{
				if(i > 0)
				{
					imagesFile << " ";
				}
				imagesFile << std::setprecision(17)
					<< observations[i].x << " "
					<< observations[i].y << " "
					<< observations[i].pointId;
			}
			imagesFile << "\n";
		}

		pointsFile << "# 3D point list with one line of data per point:\n";
		pointsFile << "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n";
		for(std::map<int, PointTrack>::const_iterator iter = exportedPoints.begin(); iter != exportedPoints.end(); ++iter)
		{
			const PointTrack & point = iter->second;
			pointsFile << iter->first << " "
				<< std::setprecision(17)
				<< point.position.x << " " << point.position.y << " " << point.position.z << " "
				<< (int)point.color[2] << " " << (int)point.color[1] << " " << (int)point.color[0] << " "
				<< point.error;
			for(size_t i = 0; i < point.track.size(); ++i)
			{
				pointsFile << " " << point.track[i].first << " " << point.track[i].second;
			}
			pointsFile << "\n";
		}

		writePly(joinPath(options.outputDir, "points3D.ply"), exportedPoints);
	}

	bool sharedIntrinsics = true;
	CameraModel sharedModel = framesByImageId.begin()->second.model;
	for(std::map<int, FrameInfo>::const_iterator iter = framesByImageId.begin(); iter != framesByImageId.end(); ++iter)
	{
		if(cameraKey(iter->second.model) != cameraKey(sharedModel))
		{
			sharedIntrinsics = false;
			break;
		}
	}

	if(options.exportNerfstudio)
	{
		std::ofstream jsonFile(joinPath(options.outputDir, "transforms.json").c_str());
		if(!jsonFile.is_open())
		{
			UERROR("Failed to open transforms.json for writing.");
			return 1;
		}
		jsonFile << "{\n";
		jsonFile << "  \"camera_model\": \"OPENCV\",\n";
		if(sharedIntrinsics)
		{
			jsonFile << "  \"fl_x\": " << std::setprecision(17) << sharedModel.fx() << ",\n";
			jsonFile << "  \"fl_y\": " << sharedModel.fy() << ",\n";
			jsonFile << "  \"cx\": " << sharedModel.cx() << ",\n";
			jsonFile << "  \"cy\": " << sharedModel.cy() << ",\n";
			jsonFile << "  \"w\": " << sharedModel.imageWidth() << ",\n";
			jsonFile << "  \"h\": " << sharedModel.imageHeight() << ",\n";
			jsonFile << "  \"k1\": 0.0,\n";
			jsonFile << "  \"k2\": 0.0,\n";
			jsonFile << "  \"p1\": 0.0,\n";
			jsonFile << "  \"p2\": 0.0,\n";
		}
		jsonFile << "  \"frames\": [\n";
		bool firstFrame = true;
		for(std::map<int, FrameInfo>::const_iterator iter = framesByImageId.begin(); iter != framesByImageId.end(); ++iter)
		{
			if(!firstFrame)
			{
				jsonFile << ",\n";
			}
			firstFrame = false;

			cv::Mat c2wOpenCV = to4x4(iter->second.cameraPoseOpenCV);
			c2wOpenCV.at<double>(0, 1) *= -1.0;
			c2wOpenCV.at<double>(1, 1) *= -1.0;
			c2wOpenCV.at<double>(2, 1) *= -1.0;
			c2wOpenCV.at<double>(0, 2) *= -1.0;
			c2wOpenCV.at<double>(1, 2) *= -1.0;
			c2wOpenCV.at<double>(2, 2) *= -1.0;

			jsonFile << "    {\n";
			jsonFile << "      \"file_path\": \"" << iter->second.relativePath << "\",\n";
			if(!sharedIntrinsics)
			{
				jsonFile << "      \"fl_x\": " << std::setprecision(17) << iter->second.model.fx() << ",\n";
				jsonFile << "      \"fl_y\": " << iter->second.model.fy() << ",\n";
				jsonFile << "      \"cx\": " << iter->second.model.cx() << ",\n";
				jsonFile << "      \"cy\": " << iter->second.model.cy() << ",\n";
				jsonFile << "      \"w\": " << iter->second.model.imageWidth() << ",\n";
				jsonFile << "      \"h\": " << iter->second.model.imageHeight() << ",\n";
				jsonFile << "      \"k1\": 0.0,\n";
				jsonFile << "      \"k2\": 0.0,\n";
				jsonFile << "      \"p1\": 0.0,\n";
				jsonFile << "      \"p2\": 0.0,\n";
			}
			jsonFile << "      \"transform_matrix\": [\n";
			for(int r = 0; r < 4; ++r)
			{
				jsonFile << "        [";
				for(int c = 0; c < 4; ++c)
				{
					if(c > 0)
					{
						jsonFile << ", ";
					}
					jsonFile << std::setprecision(17) << c2wOpenCV.at<double>(r, c);
				}
				jsonFile << "]";
				if(r < 3)
				{
					jsonFile << ",";
				}
				jsonFile << "\n";
			}
			jsonFile << "      ]\n";
			jsonFile << "    }";
		}
		jsonFile << "\n  ]\n";
		jsonFile << "}\n";
	}

	std::ofstream summary(joinPath(options.outputDir, "export_summary.txt").c_str());
	if(summary.is_open())
	{
		summary << "database=" << options.dbPath << "\n";
		summary << "output_dir=" << options.outputDir << "\n";
		summary << "frames_retained=" << framesByImageId.size() << "\n";
		summary << "frames_ignored=" << ignoredFrames << "\n";
		summary << "cameras=" << camerasByKey.size() << "\n";
		summary << "points3D=" << exportedPoints.size() << "\n";
		summary << "image_format=" << options.imageFormat << "\n";
		summary << "export_colmap=" << boolString(options.exportColmap) << "\n";
		summary << "export_nerfstudio=" << boolString(options.exportNerfstudio) << "\n";
	}

	printf("Exported 3DGS dataset to %s\n", options.outputDir.c_str());
	printf("Frames retained: %d, ignored: %d, sparse points: %d\n",
		(int)framesByImageId.size(), ignoredFrames, (int)exportedPoints.size());
	return 0;
}
