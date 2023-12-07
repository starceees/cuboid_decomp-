import cv2

# Reading the captured image
captured_image = cv2.imread('captured.png')
captured_gray = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)

# Reading the target image
target_image = cv2.imread('target_temp.png')
target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

# Applying ORB on both images
orb = cv2.ORB_create(nfeatures=2000)

# Detect and compute keypoints and descriptors for captured image
kp_captured, des_captured = orb.detectAndCompute(captured_gray, None)

# Detect and compute keypoints and descriptors for target image
kp_target, des_target = orb.detectAndCompute(target_gray, None)

# Create a BFMatcher (Brute Force Matcher) with Hamming distance
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors of captured image with descriptors of target image
matches = bf.match(des_target, des_captured)

# Sort matches based on their distances
matches = sorted(matches, key=lambda x: x.distance)

# Draw matches
matching_result = cv2.drawMatches(target_image, kp_target, captured_image, kp_captured, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matching result
cv2.imshow('Matching Result', matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
