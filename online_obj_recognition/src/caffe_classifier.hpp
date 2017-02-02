#ifndef CAFFE_CLASSIFIER_HPP
#define CAFFE_CLASSIFIER_HPP

#define USE_OPENCV 1
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class PCL_Classifier
{
    public:
      PCL_Classifier(const string& model_file,
                     const string& trained_file,
                     const string& mean_file);
      void m_run(const pcl::PointCloud<pcl::PointXYZ>::Ptr pcloud_source);

    private:
      shared_ptr<Net<float> > m_net;
      pcl::PointCloud<pcl::PointXYZ>::Ptr m_pcloud;
      pcl::PointCloud<pcl::PointXYZ>::Ptr m_pcloud_grid;
      std::vector<string> m_labels;
      static const int m_blob_range = 50;
      int m_maxN = 1;
      float m_blob_array[m_blob_range][m_blob_range][m_blob_range];
      int m_num_elem3D;
      int m_num_elem2D;

      float m_scaler;
      pcl::PointXYZ m_minpt, m_maxpt, m_range;

      void m_get_pcl(const pcl::PointCloud<pcl::PointXYZ>::Ptr pcloud_source);
      void m_classify();
      void m_output_result();

};


#endif // CAFFE_CLASSIFIER_HPP
