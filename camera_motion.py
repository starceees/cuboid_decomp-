import cv2
import numpy as np

class CameraMotionEstimator:
    def __init__(self, camera_matrix):
        self.prev_image = None
        self.prev_features = None
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.camera_matrix = camera_matrix
        self.dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    def process_frame(self, rgb_image, depth_map):
        curr_features = self.detect_features(rgb_image)

        motion = None
        if self.prev_image is not None:
            matches = self.match_features(self.prev_features, curr_features)
            if len(matches) > 4:  # Need at least 4 matches for a reliable estimate
                motion = self.estimate_motion(matches, self.prev_features, curr_features, depth_map)

        self.prev_image = rgb_image
        self.prev_features = curr_features

        return motion

    def detect_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_features(self, prev_features, curr_features):
        _, prev_descriptors = prev_features
        _, curr_descriptors = curr_features
        matches = self.bf.match(prev_descriptors, curr_descriptors)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches


    def estimate_motion(self, matches, prev_features, curr_features, depth_map):
        prev_keypoints, _ = prev_features
        curr_keypoints, _ = curr_features

        points_prev = np.float32([prev_keypoints[m.queryIdx].pt for m in matches])
        points_curr = np.float32([curr_keypoints[m.trainIdx].pt for m in matches])

        points_3d = [self.get_3d_point(pt, depth_map) for pt in points_prev]

        # Filter out None values and ensure each point is a valid 3D point
        valid_points_3d = [pt for pt in points_3d if pt is not None and len(pt) == 3]
        valid_indices = [i for i, pt in enumerate(points_3d) if pt is not None and len(pt) == 3]
        points_curr = np.float32([points_curr[i] for i in valid_indices])

        if len(valid_points_3d) >= 4 and len(valid_points_3d) == len(points_curr):
            points_3d_np = np.array(valid_points_3d, dtype=np.float32).reshape(-1, 3)
            points_curr_np = points_curr.reshape(-1, 2)
            _, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d_np, points_curr_np, self.camera_matrix, self.dist_coeffs)
            R, _ = cv2.Rodrigues(rvec)
            return R, tvec
        else:
            print("Not enough points for motion estimation")
            return None  # Not enough points for a reliable estimate



    def get_3d_point(self, pt, depth_map):
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
            depth = depth_map[y, x]
            if depth > 0:  # Check for valid depth
                return self.project_pixel_to_3d(pt, depth)
        return None

    def project_pixel_to_3d(self, pt, depth):
        x, y = pt
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        z = depth
        x = (x - cx) * z / fx
        y = (y - cy) * z / fy
        return np.array([x, y, z])


# # Camera intrinsic matrix
# camera_matrix = np.array([[92., 0, 160.], [0, 92., 120.], [0, 0, 1]])

# # Example usage
# motion_estimator = CameraMotionEstimator(camera_matrix)

# # Assume rgb_image and depth_map are obtained from your image processing pipeline
# rgb_image = ...  # Your current RGB frame
# depth_map = ...  # Your current depth map

# motion = motion_estimator.process_frame(rgb_image, depth_map)
# if motion:
#     R, tvec = motion
#     print(f"Rotation Matrix: {R}, Translation Vector: {tvec}")
