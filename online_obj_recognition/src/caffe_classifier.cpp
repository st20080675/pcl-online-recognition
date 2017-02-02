#include "caffe_classifier.hpp"


PCL_Classifier::PCL_Classifier(const string& model_file,
                       const string& trained_file,
                       const string& label_file)
{
    #ifdef CPU_ONLY
      Caffe::set_mode(Caffe::CPU);
    #else
      Caffe::set_mode(Caffe::GPU);
    #endif

    /* Load the network. */
    m_net.reset(new Net<float>(model_file, TEST));
    m_net->CopyTrainedLayersFrom(trained_file);

    /* Load labels. */
    std::ifstream labels(label_file.c_str());
    CHECK(labels) << "Unable to open labels file " << label_file;
    string line;
    while (std::getline(labels, line))
        m_labels.push_back(string(line));

    Blob<float>* output_layer = m_net->output_blobs()[0];
    CHECK_EQ(m_labels.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";

    // save computation for later part
    m_num_elem3D = m_blob_range * m_blob_range * m_blob_range;
    m_num_elem2D = m_blob_range * m_blob_range;
}


void PCL_Classifier::m_get_pcl(const pcl::PointCloud<pcl::PointXYZ>::Ptr pcloud_source)
{
    /* get raw point cloud */
    m_pcloud = pcl::PointCloud<pcl::PointXYZ>::Ptr (new pcl::PointCloud<pcl::PointXYZ>);
    m_pcloud_grid = pcl::PointCloud<pcl::PointXYZ>::Ptr (new pcl::PointCloud<pcl::PointXYZ>);

    pcl::copyPointCloud(*pcloud_source, *m_pcloud);

    /* remove NAN points from the cloud */
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*m_pcloud, *m_pcloud, indices);
    // calculate raw input range/boundry
    pcl::getMinMax3D (*m_pcloud, m_minpt, m_maxpt);
    m_range.x = m_maxpt.x - m_minpt.x;
    m_range.y = m_maxpt.y - m_minpt.y;
    m_range.z = m_maxpt.z - m_minpt.z;
    m_scaler = (m_blob_range - 1)/(std::max(std::max(m_range.x, m_range.y), m_range.z));

    // shift and scale
    for (int i = 0; i < m_pcloud->points.size(); ++i)
    {
        m_pcloud->points[i].x = (m_pcloud->points[i].x - m_minpt.x)*m_scaler;
        m_pcloud->points[i].y = (m_pcloud->points[i].y - m_minpt.y)*m_scaler;
        m_pcloud->points[i].z = (m_pcloud->points[i].z - m_minpt.z)*m_scaler;
    }

    /* resampling the point blocb to blob_range * blob_range * blob_range */
    // Filter object.
    pcl::VoxelGrid<pcl::PointXYZ> filter;
    filter.setInputCloud(m_pcloud);
    // We set the size of every voxel to be 1x1x1cm is like: filter.setLeafSize(0.01f, 0.01f, 0.01f);
    // (only one point per every cubic centimeter will survive).
    filter.setLeafSize(0.5, 0.5, 0.5);
    filter.filter(*m_pcloud_grid);

    memset(m_blob_array, 0, sizeof(m_blob_array[0][0][0]) * m_num_elem3D);

    for (int i = 0; i < m_pcloud_grid->points.size(); ++i)
    {
        m_blob_array[(int)round(m_pcloud_grid->points[i].z)][(int)round(m_pcloud_grid->points[i].y)][(int)round(m_pcloud_grid->points[i].x)] = 1;
    }


}

void PCL_Classifier::m_classify()
{
    Blob<float>* input_blobs = m_net->input_blobs()[0];
    memcpy(input_blobs->mutable_cpu_data(), m_blob_array, sizeof(float) * input_blobs->count());
    m_net->ForwardPrefilled();
}

/* the below function is used in void PCL_Classifier::m_output_result() */
static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

void PCL_Classifier::m_output_result()
{
    /* Copy the output layer to a std::vector */
    Blob<float>* output_layer = m_net->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();
    std::vector<float>result = std::vector<float>(begin, end);

    m_maxN = std::min<int>(m_labels.size(), m_maxN);

    std::vector<std::pair<float, int> > pairs;
    for (size_t i = 0; i < result.size(); ++i)
    {
        pairs.push_back(std::make_pair(result[i], i));
    }
    std::partial_sort(pairs.begin(), pairs.begin() + m_maxN, pairs.end(), PairCompare);
    std::vector<int> maxN;
    for (int i = 0; i < m_maxN; ++i)
    {
        maxN.push_back(pairs[i].second);
    }

    std::vector<std::pair<string, float> > predictions;
    for (int i = 0; i < m_maxN; ++i)
    {
      int idx = maxN[i];
      predictions.push_back(std::make_pair(m_labels[idx], result[idx]));
    }

    /* Print the top N predictions. */
    for (size_t i = 0; i < predictions.size(); ++i)
    {
      std::pair<string, float> p = predictions[i];
      std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
                << p.first << "\"" << std::endl;
    }
}

void PCL_Classifier::m_run(const pcl::PointCloud<pcl::PointXYZ>::Ptr pcloud_source)
{
    m_get_pcl(pcloud_source);
    m_classify();
    m_output_result();
}
