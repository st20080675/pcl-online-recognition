// Original code by Geoffrey Biggs, taken from the PCL tutorial in
// http://pointclouds.org/documentation/tutorials/pcl_visualizer.php

// Simple OpenNI viewer that also allows to write the current scene to a .pcd
// when pressing SPACE.

#include <pcl/io/openni_grabber.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/console/parse.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/common.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/conditional_removal.h>

#include "caffe_classifier.hpp"

#include <iostream>

using namespace std;
using namespace pcl;

bool flag_busy = 0;
bool flag_init = 1;
PCL_Classifier *p_classifier;
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer4ports;  // Point cloud viewer object.
int v1(0), v2(0), v3(0), v4(0), v5(0), v6(0);
Grabber* openniGrabber;                                               // OpenNI grabber that takes data from the device.

boost::shared_ptr<pcl::visualization::PCLVisualizer> view4portsVisInit()
{
  // --------------------------------------------------------
  // -----Open 3D viewer and add point cloud and normals-----
  // --------------------------------------------------------
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->initCameraParameters ();

  viewer->createViewPort(0.0, 0.5, 0.3, 1.0, v1);
  viewer->setBackgroundColor (0.5, 0.5, 0.5, v1);
  viewer->addText("raw point cloud", 10, 10, "v1 text", v1);

  viewer->createViewPort(0.3, 0.5, 0.6, 1.0, v2);
  viewer->setBackgroundColor (0.3, 0.3, 0.3, v2);
  viewer->addText("point cloud in the selected range", 10, 10, "v2 text", v2);

  viewer->createViewPort(0.0, 0.0, 0.3, 0.5, v3);
  viewer->setBackgroundColor (0.3, 0.3, 0.3, v3);
  viewer->addText("segmented plane", 10, 10, "v3 text", v3);

  viewer->createViewPort(0.3, 0.0, 0.6, 0.5, v4);
  viewer->setBackgroundColor (0.5, 0.5, 0.5, v4);
  viewer->addText("segmented object", 10, 10, "v4 text", v4);

  // the XYZ(no RGB)
  viewer->createViewPort(0.6, 0.5, 1.0, 1.0, v5);
  viewer->setBackgroundColor (0.0, 0.0, 0.0, v5);
  viewer->addText("point cloud in the selected range (no color)", 10, 10, "v5 text", v5);

  viewer->createViewPort(0.6, 0.0, 1.0, 0.5, v6);
  viewer->setBackgroundColor (0.3, 0.3, 0.3, v6);
  viewer->addText("segmented object (no color)", 10, 10, "v6 text", v6);

  return (viewer);
}



// This function is called every time the device has new data.
void grabberCallback(const PointCloud<pcl::PointXYZRGBA>::ConstPtr& cloud_rgb)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr convexHull(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr objects(new pcl::PointCloud<pcl::PointXYZ>);
    // for colorful visualization

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_rgb_in_range(new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr plane_rgb(new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr objects_rgb(new pcl::PointCloud<pcl::PointXYZRGBA>);
    //pcl::copyPointCloud(*cloud_rgb, *cloud_rgb_in_range);

    /* select the points in certain by filtering */
    // And "And" condition requires all tests to check true. "Or" conditions also available.
    pcl::ConditionAnd<pcl::PointXYZRGBA>::Ptr condition(new pcl::ConditionAnd<pcl::PointXYZRGBA>);
    // First test, the point's Z value must be greater than (GT) 0.
    condition->addComparison(pcl::FieldComparison<pcl::PointXYZRGBA>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZRGBA>("z", pcl::ComparisonOps::GT, 0.0)));
    // Second test, the point's Z value must be less than (LT) 2.
    condition->addComparison(pcl::FieldComparison<pcl::PointXYZRGBA>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZRGBA>("z", pcl::ComparisonOps::LT, 1.6)));
    condition->addComparison(pcl::FieldComparison<pcl::PointXYZRGBA>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZRGBA>("x", pcl::ComparisonOps::GT, -1.0)));
    condition->addComparison(pcl::FieldComparison<pcl::PointXYZRGBA>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZRGBA>("x", pcl::ComparisonOps::LT, 0.4)));

    // Filter object.
    pcl::ConditionalRemoval<pcl::PointXYZRGBA> filter;
    filter.setCondition(condition);
    filter.setInputCloud(cloud_rgb);
    // If true, points that do not pass the filter will be set to a certain value (default NaN).
    // If false, they will be just removed, but that could break the structure of the cloud
    // (organized clouds are clouds taken from camera-like sensors that return a matrix-like image).
    filter.setKeepOrganized(false);
    // If keep organized was set true, points that failed the test will have their Z value set to this.
    //filter.setUserFilterValue(0.0);
    filter.filter(*cloud_rgb_in_range);

    //cloud->height = cloud_rgb->height;
    //cloud->width = cloud_rgb->width;
    //copy points
    for (int i = 0; i < cloud_rgb_in_range->points.size(); i++)
    {
        pcl::PointXYZ basic_point;
        basic_point.x = cloud_rgb_in_range->points[i].x;
        basic_point.y = cloud_rgb_in_range->points[i].y;
        basic_point.z = cloud_rgb_in_range->points[i].z;

        cloud->points.push_back(basic_point);
    }

    /* remove NAN points from the cloud */
    //std::vector<int> indices;
    //pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);

    // just to know the range, zmin~0.642; zmax~1.594
    /*
    pcl::PointXYZ minpt, maxpt;
    pcl::getMinMax3D (*cloud, minpt, maxpt);
    std::cout << "max z is: " << maxpt.z << "; min z is: "<< minpt.z<< std::endl;
    */

    // Get the plane model, if present.
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::SACSegmentation<pcl::PointXYZ> segmentation;
    segmentation.setInputCloud(cloud);
    segmentation.setModelType(pcl::SACMODEL_PLANE);
    segmentation.setMethodType(pcl::SAC_RANSAC);
    segmentation.setDistanceThreshold(0.03);
    segmentation.setOptimizeCoefficients(true);
    pcl::PointIndices::Ptr planeIndices(new pcl::PointIndices);
    segmentation.segment(*planeIndices, *coefficients);

    if (planeIndices->indices.size() == 0)
    {
        std::cout << "Could not find a plane in the scene." << std::endl;
    }
    else
    {
        // Copy the points of the plane to a new cloud.
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(planeIndices);
        extract.filter(*plane);

        // Retrieve the convex hull.
        pcl::ConvexHull<pcl::PointXYZ> hull;
        hull.setInputCloud(plane);
        // Make sure that the resulting hull is bidimensional.
        hull.setDimension(2);
        hull.reconstruct(*convexHull);

        // Redundant check.
        if (hull.getDimension() == 2)
        {
           // Prism object.
           pcl::ExtractPolygonalPrismData<pcl::PointXYZ> prism;
           prism.setInputCloud(cloud);
           prism.setInputPlanarHull(convexHull);
           // First parameter: minimum Z value. Set to 0, segments objects lying on the plane (can be negative).
           // Second parameter: maximum Z value, set to 10cm. Tune it according to the height of the objects you expect.
           prism.setHeightLimits(0.03f, 0.5f);
           pcl::PointIndices::Ptr objectIndices(new pcl::PointIndices);

           prism.segment(*objectIndices);

           // Get and show all points retrieved by the hull.
           extract.setIndices(objectIndices);
           extract.filter(*objects);

           pcl::ExtractIndices<pcl::PointXYZRGBA> extract_rgb;
           extract_rgb.setInputCloud(cloud_rgb_in_range);
           //extract_rgb.setIndices(indices);  // remove nan
           //extract_rgb.filter(*cloud_rgb_in_range);
           //extract_rgb.setInputCloud(cloud_rgb);
           extract_rgb.setIndices(planeIndices);
           extract_rgb.filter(*plane_rgb);
           extract_rgb.setIndices(objectIndices);
           extract_rgb.filter(*objects_rgb);

           p_classifier->m_run(objects);

           // for visualization
           pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb(cloud_rgb);
           pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb_in_range(cloud_rgb_in_range);
           pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb_plane(plane_rgb);
           pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb_obj(objects_rgb);

           if (flag_init)
           {
               flag_init = 0;
               viewer4ports->addPointCloud<pcl::PointXYZRGBA> (cloud_rgb, rgb, "raw cloud", v1);
               viewer4ports->addPointCloud<pcl::PointXYZRGBA> (cloud_rgb_in_range, rgb_in_range, "point cloud in range", v2);
               viewer4ports->addPointCloud<pcl::PointXYZRGBA> (plane_rgb, rgb_plane, "segmented plane", v3);
               viewer4ports->addPointCloud<pcl::PointXYZRGBA> (objects_rgb, rgb_obj, "segmented objects", v4);
               // no color
               viewer4ports->addPointCloud<pcl::PointXYZ> (convexHull, "point cloud in range (no color)", v5);
               viewer4ports->addPointCloud<pcl::PointXYZ> (objects, "segmented objects (no color)", v6);

               viewer4ports->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "raw cloud");
               viewer4ports->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "point cloud in range");
               viewer4ports->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "segmented plane");
               viewer4ports->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "segmented objects");

               viewer4ports->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "point cloud in range (no color)");
               viewer4ports->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "segmented objects (no color)");
               viewer4ports->addCoordinateSystem (0.5);
           }
           else
           {
               while (flag_busy)
               {
                   ;
               }
               flag_busy = 1;
               viewer4ports->updatePointCloud(cloud_rgb, "raw cloud");
               viewer4ports->updatePointCloud(cloud_rgb_in_range, "point cloud in range");
               viewer4ports->updatePointCloud(plane_rgb, "segmented plane");
               viewer4ports->updatePointCloud(objects_rgb, "segmented objects");

               viewer4ports->updatePointCloud(convexHull, "point cloud in range (no color)");
               viewer4ports->updatePointCloud(objects, "segmented objects (no color)");
               flag_busy = 0;
           }

         }
         else
        {
            std::cout << "The chosen hull is not planar." << std::endl;
        }
    }


}

int main(int argc, char** argv)
{
    // for caffe
    if (argc != 4)
    {
      std::cerr << "Usage: " << argv[0]
                << " deploy.prototxt network.caffemodel labels.txt"
                << std::endl;
      return 1;
    }

    ::google::InitGoogleLogging(argv[0]);

    string model_file   = argv[1];
    string trained_file = argv[2];
    string label_file   = argv[3];

    p_classifier = new PCL_Classifier(model_file, trained_file, label_file);


    // for openni
    openniGrabber = new OpenNIGrabber();
    if (openniGrabber == 0)
    {
        return -1;
    }

    boost::function<void (const PointCloud<PointXYZRGBA>::ConstPtr&)> f = boost::bind(&grabberCallback, _1);
    openniGrabber->registerCallback(f);
    viewer4ports = view4portsVisInit();

    openniGrabber->start();

    // Main loop.
    while (!viewer4ports->wasStopped())
    {
        while (flag_busy)
        {
            ;
        }
        flag_busy = 1;
        viewer4ports->spinOnce(500);
        flag_busy = 0;
        boost::this_thread::sleep(boost::posix_time::seconds(0.1));
    }

    openniGrabber->stop();

}
