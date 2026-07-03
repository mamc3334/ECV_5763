#include "count_finger.hpp"

using namespace cv;
using namespace std;

//Generated with claude and gemeni help

// Helper structure to compare cv::Point in std::set
struct PointCompare {
    bool operator()(const cv::Point& a, const cv::Point& b) const {
        return (a.x < b.x) || (a.x == b.x && a.y < b.y);
    }
};

/**
 * Traces a single finger branch from its tip down to the knuckle junction.
 * Returns a bounding rectangle wrapping the entire finger.
 */
cv::Rect traceWholeFinger(const cv::Mat& binarySkel, cv::Point tip)
{
    std::vector<cv::Point> fingerPoints;
    std::set<cv::Point, PointCompare> visited;
    
    std::queue<cv::Point> q;
    q.push(tip);
    visited.insert(tip);

    while (!q.empty())
    {
        cv::Point curr = q.front();
        q.pop();
        fingerPoints.push_back(curr);

        // Count total neighbors in the 3x3 to see if we hit the palm junction
        int totalNeighbors = 0;
        std::vector<cv::Point> unvisitedNeighbors;

        for (int ny = -1; ny <= 1; ++ny)
        {
            for (int nx = -1; nx <= 1; ++nx)
            {
                if (ny == 0 && nx == 0) continue;
                
                cv::Point neighbor(curr.x + nx, curr.y + ny);
                // Safe boundary check
                if (neighbor.x >= 0 && neighbor.x < binarySkel.cols &&
                    neighbor.y >= 0 && neighbor.y < binarySkel.rows)
                {
                    if (binarySkel.at<uchar>(neighbor.y, neighbor.x) == 255)
                    {
                        totalNeighbors++;
                        if (visited.find(neighbor) == visited.end())
                        {
                            unvisitedNeighbors.push_back(neighbor);
                        }
                    }
                }
            }
        }

        // Junction Detection: If a pixel has 3 or more total neighbors, 
        // it means we have reached the palm/knuckle area where fingers merge.
        if (totalNeighbors >= 3)
        {
            break; // Stop tracing this finger further into the palm
        }

        // Otherwise, continue down the 1-pixel-wide finger line
        for (const auto& nextPt : unvisitedNeighbors)
        {
            visited.insert(nextPt);
            q.push(nextPt);
        }
    }

    // Compute a true bounding box enclosing all collected points of this finger
    return cv::boundingRect(fingerPoints);
}

/**
 * Finds fingertips in the center 3x3 grid sector, traces their full length,
 * and draws bounding rectangles around each finger.
 */
int drawBoxesAroundFingers(const cv::Mat& skel, cv::Mat& canvas)
{
    cv::Mat binarySkel = skel;
    if (skel.channels() == 3) {
        cv::cvtColor(skel, binarySkel, cv::COLOR_BGR2GRAY);
    }

    if (canvas.empty() || canvas.size() != skel.size() || canvas.channels() != 3) {
        cv::cvtColor(binarySkel, canvas, cv::COLOR_GRAY2BGR);
    }

    // Center grid boundaries
    int startX = binarySkel.cols / 3;
    int endX   = (2 * binarySkel.cols) / 3;
    int startY = binarySkel.rows / 3;
    int endY   = (2 * binarySkel.rows) / 3;

    // Draw the green tracking zone
    cv::rectangle(canvas, cv::Point(startX, startY), cv::Point(endX, endY), cv::Scalar(0, 255, 0), 2);

    int fingerCount = 0;

    // 1. First Pass: Locate all fingertip endpoints within the target area
    for (int y = startY; y < endY; ++y)
    {
        for (int x = startX; x < endX; ++x)
        {
            if (binarySkel.at<uchar>(y, x) == 255)
            {
                int neighborCount = 0;
                for (int ny = -1; ny <= 1; ++ny)
                {
                    for (int nx = -1; nx <= 1; ++nx)
                    {
                        if (ny == 0 && nx == 0) continue;
                        if (binarySkel.at<uchar>(y + ny, x + nx) == 255) {
                            neighborCount++;
                        }
                    }
                }

                // Found a valid fingertip pixel
                if (neighborCount == 1)
                {
                    fingerCount++;
                    
                    // 2. Trace the line back to the hand mass and calculate its specific bounding box
                    cv::Rect fingerBox = traceWholeFinger(binarySkel, cv::Point(x, y));
                    
                    // 3. Draw a red rectangle wrapped around this full single finger
                    cv::rectangle(canvas, fingerBox, cv::Scalar(0, 0, 255), 2);
                }
            }
        }
    }

    // Print text readout
    std::string text = "Fingers: " + std::to_string(fingerCount);
    cv::putText(canvas, text, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);

    return fingerCount;
}

int highlightAndCountFingers(const cv::Mat& skel, cv::Mat& canvas)
{
    // Ensure our output canvas is 3-channel BGR so we can draw in color
    cvtColor(skel, canvas, cv::COLOR_GRAY2BGR);

    int endpointCount = 0;

    // Define the bounding box for the center section of a 3x3 grid
    int startX = skel.cols / 4;
    int endX   = (3 * skel.cols) / 4;
    int startY = skel.rows / 4;
    int endY   = (3 * skel.rows) / 4;

    // Draw the green center grid boundary on the canvas
    cv::rectangle(canvas, cv::Point(startX, startY), cv::Point(endX, endY), cv::Scalar(0, 255, 0), 2);

    // Vector to store the locations of our endpoints
    std::vector<cv::Point> fingertips;

    // Loop through ONLY the pixels inside the center grid sector
    for (int y = startY; y < endY; ++y)
    {
        for (int x = startX; x < endX; ++x)
        {
            if (skel.at<uchar>(y, x) == 255)
            {
                int neighborCount = 0;

                // Examine the 8-connected neighbors
                for (int ny = -2; ny <= 2; ++ny)
                {
                    for (int nx = -2; nx <= 2; ++nx)
                    {
                        if (ny == 0 && nx == 0) continue;

                        if (skel.at<uchar>(y + ny, x + nx) == 255)
                        {
                            neighborCount++;
                        }
                    }
                }

                // If it has exactly 1 neighbor, it's a fingertip endpoint
                if (neighborCount == 1)
                {
                    fingertips.push_back(cv::Point(x, y));
                    endpointCount++;
                }
            }
        }
    }

    // Draw a small red bounding box around each detected fingertip endpoint
    int boxSize = 10; // Dimensions of the bounding box around the tip
    for (const auto& pt : fingertips)
    {
        cv::Point topLeft(pt.x - boxSize / 2, pt.y - boxSize / 2);
        cv::Point bottomRight(pt.x + boxSize / 2, pt.y + boxSize / 2);
        
        // Draw red rectangle (BGR: Blue=0, Green=0, Red=255)
        cv::rectangle(canvas, topLeft, bottomRight, cv::Scalar(0, 0, 255), 2);
    }

    // Optional: Print the finger count text onto the top-left of the frame
    std::string countText = "Fingers: " + std::to_string(endpointCount);
    cv::putText(canvas, countText, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);

    return endpointCount;
}

// Count 8-connected white neighbors of pixel (x,y).
// Caller must ensure (x,y) is not on the image border.
static int neighborCount(const Mat &img, int x, int y)
{
    int n = 0;
    for (int dy = -2; dy <= 2; dy++)
        for (int dx = -2; dx <= 2; dx++)
            if (!(dx == 0 && dy == 0))
                n += (img.at<uchar>(y + dy, x + dx) > 0);
    return n;
}

// Returns finger count, draws green bounding boxes on debugOut (BGR).
int countFingers(const Mat &src, Mat &debugOut)
{
    CV_Assert(!src.empty() && src.type() == CV_8UC1);
    cvtColor(src, debugOut, COLOR_GRAY2BGR);

    // ── Step 1: Isolate the largest skeleton component (discard noise blobs) ──
    Mat labels, stats, centroids;
    int nLabels = connectedComponentsWithStats(src, labels, stats, centroids, 8, CV_32S);

    int handLabel = -1, maxArea = 0;
    for (int i = 1; i < nLabels; i++)
    {
        int area = stats.at<int>(i, CC_STAT_AREA);
        if (area > maxArea) { maxArea = area; handLabel = i; }
    }
    if (handLabel < 0) return 0;

    Mat handSkel;
    compare(labels, handLabel, handSkel, CMP_EQ);   // CV_8UC1, 255 where hand

    // ── Step 2: Find all skeleton endpoints (1-neighbor pixels = tips) ──
    vector<Point> endpoints;
    for (int y = 1; y < handSkel.rows - 1; y++)
        for (int x = 1; x < handSkel.cols - 1; x++)
            if (handSkel.at<uchar>(y, x) > 0 && neighborCount(handSkel, x, y) == 1)
                endpoints.push_back(Point(x, y));

    if (endpoints.empty()) return 0;

    // ── Step 3: Cluster nearby endpoints (merge spurious branch artifacts) ──
    const int CLUSTER_RADIUS = 25;
    vector<vector<Point>> clusters;
    vector<bool> used(endpoints.size(), false);

    for (size_t i = 0; i < endpoints.size(); i++)
    {
        if (used[i]) continue;
        vector<Point> grp = { endpoints[i] };
        used[i] = true;
        for (size_t j = i + 1; j < endpoints.size(); j++)
            if (!used[j] && norm(endpoints[i] - endpoints[j]) < CLUSTER_RADIUS)
            {
                grp.push_back(endpoints[j]);
                used[j] = true;
            }
        clusters.push_back(grp);
    }

    // ── Step 4: Discard the arm/wrist endpoint (spatial outlier filter) ──
    // Fingertips cluster together; the arm tip is a lone outlier far from the rest.
    vector<Point2f> cc;   // cluster centroids
    for (auto &grp : clusters)
    {
        Point2f c(0, 0);
        for (auto &p : grp) { c.x += p.x; c.y += p.y; }
        c *= 1.0f / grp.size();
        cc.push_back(c);
    }

    // Group centroid (mean of cluster centroids)
    Point2f groupCenter(0, 0);
    for (auto &c : cc) groupCenter += c;
    groupCenter *= 1.0f / cc.size();

    // Compute per-cluster distance to group center
    vector<float> dists;
    for (auto &c : cc) dists.push_back((float)norm(c - groupCenter));

    float meanDist = 0;
    for (float d : dists) meanDist += d;
    meanDist /= dists.size();

    // Remove any cluster that is >2× the mean distance (the arm outlier)
    vector<vector<Point>> fingerClusters;
    for (size_t i = 0; i < clusters.size(); i++)
        if (dists[i] <= 2.0f * meanDist)
            fingerClusters.push_back(clusters[i]);

    // ── Step 5: Draw padded bounding boxes, label each finger ──
    const int PAD = 15;
    for (size_t i = 0; i < fingerClusters.size(); i++)
    {
        Rect bbox = boundingRect(fingerClusters[i]);
        bbox.x      = max(0,            bbox.x      - PAD);
        bbox.y      = max(0,            bbox.y      - PAD);
        bbox.width  = min(src.cols - bbox.x, bbox.width  + 2 * PAD);
        bbox.height = min(src.rows - bbox.y, bbox.height + 2 * PAD);

        rectangle(debugOut, bbox, Scalar(0, 255, 0), 2);
        putText(debugOut, to_string(i + 1),
                Point(bbox.x + 3, bbox.y + 16),
                FONT_HERSHEY_SIMPLEX, 0.55, Scalar(0, 255, 0), 1);
    }

    // Optionally annotate total count
    putText(debugOut, "Fingers: " + to_string(fingerClusters.size()),
            Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 200, 255), 2);

    return (int)fingerClusters.size();
}


int contorFingers(const cv::Mat &src, cv::Mat &debugOut)
{
    Mat target;
    // Count the number of fingers in the skeletal image
    // This is a simple approach and may not be accurate for all cases
    int fingerCount = 0;

    cvtColor(src, debugOut, COLOR_GRAY2BGR);

    // Reduce the target area to the center of the image to avoid noise
    Rect targetArea(src.cols/4, src.rows/4, src.cols*3/4, src.rows*3/4);
    src(targetArea).copyTo(target);

    // Find contours in the skeletal image
    vector<vector<Point>> contours;
    findContours(target, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i < contours.size(); i++)
    {
        // Approximate the contour to reduce the number of points
        vector<Point> approx;
        approxPolyDP(contours[i], approx, 5, true);

        //draw rectangle around the contour for debugging
        Rect boundingBox = boundingRect(approx);
        rectangle(debugOut, boundingBox, Scalar(255, 255, 255), 1);

        // If the contour has more than 4 points, it may be a finger
        if (approx.size() > 4)
        {
            fingerCount++;
        }
    }

    return fingerCount;
}

int improvedCountFingers(const cv::Mat& skel, const cv::Mat& binaryMask, cv::Mat& canvas)
{
    // 1. Prepare canvas for color drawing
    if (skel.channels() == 1) {
        cv::cvtColor(skel, canvas, cv::COLOR_GRAY2BGR);
    } else {
        skel.copyTo(canvas);
    }

    if (skel.empty() || binaryMask.empty()) return 0;

    // 2. Find the Palm Center using Distance Transform on the binary mask
    cv::Mat dist;
    cv::distanceTransform(binaryMask, dist, cv::DIST_L2, 3);
    
    double maxVal;
    cv::Point palmCenter;
    cv::minMaxLoc(dist, nullptr, &maxVal, nullptr, &palmCenter);
    
    // Draw the palm center (Blue circle)
    cv::circle(canvas, palmCenter, 8, cv::Scalar(255, 0, 0), -1);

    // 3. Find true endpoints using contour approximation (handles imperfect thinning)
    // We find contours of the skeleton strokes themselves
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(skel, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Point> rawEndpoints;

    for (const auto& contour : contours)
    {
        // Smooth out the skeleton line path
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, 3.0, false);

        if (approx.size() < 2) continue;

        // The sharp turns/ends of a skeleton contour represent line endpoints
        // For an open path contour, the start and end points of the array are the tips
        rawEndpoints.push_back(approx.front());
        rawEndpoints.push_back(approx.back());
    }

    // 4. Filter and Cluster Endpoints
    // - Filter 1: Must be to the LEFT of the palm center (since hand points left)
    // - Filter 2: Must be far enough from the palm center (eliminates knuckles/wrist)
    double minDistanceFromPalm = maxVal * 1.8; // Adjust threshold scaling factor as needed
    
    std::vector<cv::Point> validTips;
    for (const auto& pt : rawEndpoints)
    {
        // Ensure it's pointing left relative to the palm
        if (pt.x >= palmCenter.x) continue; 

        // Ensure it's not a knuckle (too close to palm center)
        double distToPalm = cv::norm(pt - palmCenter);
        if (distToPalm < minDistanceFromPalm) continue;

        validTips.push_back(pt);
    }

    // Cluster nearby duplicate points (merges branch artifacts)
    const double CLUSTER_RADIUS = 20.0;
    std::vector<cv::Point> fingerTips;
    std::vector<bool> visited(validTips.size(), false);

    for (size_t i = 0; i < validTips.size(); ++i)
    {
        if (visited[i]) continue;

        cv::Point clusterMean = validTips[i];
        int count = 1;
        visited[i] = true;

        for (size_t j = i + 1; j < validTips.size(); ++j)
        {
            if (!visited[j] && cv::norm(validTips[i] - validTips[j]) < CLUSTER_RADIUS)
            {
                clusterMean += validTips[j];
                count++;
                visited[j] = true;
            }
        }
        
        // Calculate average point of the cluster
        clusterMean.x /= count;
        clusterMean.y /= count;
        fingerTips.push_back(clusterMean);
    }

    // 5. Draw bounding boxes and label the final detected fingertips
    int boxSize = 16;
    for (size_t i = 0; i < fingerTips.size(); ++i)
    {
        cv::Point topLeft(fingerTips[i].x - boxSize / 2, fingerTips[i].y - boxSize / 2);
        cv::Point bottomRight(fingerTips[i].x + boxSize / 2, fingerTips[i].y + boxSize / 2);
        
        // Draw Red bounding box around final fingertips
        cv::rectangle(canvas, topLeft, bottomRight, cv::Scalar(0, 0, 255), 2);
        
        // Label them 1, 2, 3...
        cv::putText(canvas, std::to_string(i + 1), cv::Point(topLeft.x, topLeft.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }

    // Print final count overlay
    std::string countText = "Fingers: " + std::to_string(fingerTips.size());
    cv::putText(canvas, countText, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);

    return static_cast<int>(fingerTips.size());
}

int countCircle(const cv::Mat& skel, const cv::Mat& binaryMask, cv::Mat& canvas)
{
    // 1. Prepare canvas for color drawing
    if (skel.channels() == 1) {
        cv::cvtColor(skel, canvas, cv::COLOR_GRAY2BGR);
    } else {
        skel.copyTo(canvas);
    }

    if (binaryMask.empty()) return 0;

    // 2. Find the Palm Center using Distance Transform
    cv::Mat dist;
    cv::distanceTransform(binaryMask, dist, cv::DIST_L2, 3);
    
    double maxVal;
    cv::Point palmCenter;
    cv::minMaxLoc(dist, nullptr, &maxVal, nullptr, &palmCenter);
    
    // Draw the palm center (Blue circle)
    cv::circle(canvas, palmCenter, 8, cv::Scalar(255, 0, 0), -1);

    // 3. Create a vertical sampling line to the left of the palm
    // Since maxVal is the radius of the palm, we step out past the palm into the fingers
    int sampleX = palmCenter.x - static_cast<int>(maxVal * 1.8); 
    
    // Safety check to keep the sampling column inside image bounds
    if (sampleX < 0) sampleX = 0;

    // Draw the sampling line (Green line) for debugging visual feedback
    cv::line(canvas, cv::Point(sampleX, 0), cv::Point(sampleX, binaryMask.rows), cv::Scalar(0, 255, 0), 2);

    // 4. Scan down the sample line and count transitions from Black to White
    int fingerCount = 0;
    bool insideFinger = false;

    for (int y = 0; y < binaryMask.rows; ++y)
    {
        // Check if the binary mask pixel is white (255) at our sampling X coordinate
        if (binaryMask.at<uchar>(y, sampleX) == 255)
        {
            if (!insideFinger)
            {
                // We just entered a finger!
                insideFinger = true;
                fingerCount++;
                
                // Draw a small red indicator box where it detected the finger crossing
                cv::rectangle(canvas, cv::Point(sampleX - 5, y - 5), cv::Point(sampleX + 5, y + 5), cv::Scalar(0, 0, 255), -1);
            }
        }
        else
        {
            // We hit a space between fingers
            insideFinger = false;
        }
    }

    // Print final count overlay
    std::string countText = "Fingers: " + std::to_string(fingerCount);
    cv::putText(canvas, countText, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);

    return fingerCount;
}