import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
DEBUG = False
MATCHER = "knn" # available: "knn" - "bf"
FEATURE_EXTRACTOR = "sift" # available: "sift" - "orb"


def drawMatches(img1, kp1, img2, kp2, win_name=None):
    # get dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # create display
    view = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)

    view[:h1, :w1, :] = img1
    view[:h2, w2:, :] = img2

    color = (0, 0, 255)

    for idx in range(kp1.shape[0]):
        cv2.line(view, 
                 (int(kp1[idx,0,0]), int(kp1[idx,0,1])), 
                 (int(kp2[idx,0,0] + w1), int(kp2[idx,0,1])), 
                 color, 2)
    if not win_name is None:
        cv2.imshow(win_name, view)
    else:
        cv2.imshow("img", view)

    cv2.waitKey()
    cv2.destroyAllWindows()

def featureMatcher(img_left, img_right, n_pts = None):

    if FEATURE_EXTRACTOR == "sift":
        feature_extractor = cv2.xfeatures2d.SIFT_create()
    elif FEATURE_EXTRACTOR == "orb":
        feature_extractor = cv2.ORB_create(nfeatures=10000, edgeThreshold=10, fastThreshold=10)


    kp1, des1 = feature_extractor.detectAndCompute(img_left, None)
    kp2, des2 = feature_extractor.detectAndCompute(img_right, None)


    if MATCHER == "knn":
        des1 = des1.astype(np.float32)
        des2 = des2.astype(np.float32)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        
        p_left = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 1 ,2)
        p_right = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)

    elif MATCHER == "bf":
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        matches = sorted(matches, key=lambda x: x.distance)

        p_left = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        p_right = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    if DEBUG:
        drawMatches(img_left, p_left, img_right, p_right, "FM output")


    if not n_pts is None:
        return p_left[:n_pts], p_right[:n_pts]
    else:
        return p_left, p_right

def getCameraMatrix():
    p_2d = np.load("vr2d.npy")
    p_3d = np.load("vr3d.npy")

    ## get intrinsic matrix ##
    # aspect ratio fixed --> fx = fy
    f = 100
    # principal points are known and fixed
    cx, cy= 960, 540
    # zero skewness for initial guess
    c = 0
    intrinsic_mat = np.array([[f,  c,   cx],
                              [0,  f,   cy],
                              [0,  0,   1]], dtype=np.float32)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([p_3d[:,0,:]], [p_2d[:,0,:]], (1920, 1080), 
                                                       intrinsic_mat, np.zeros((14)), 
                                                       flags=(cv2.CALIB_USE_INTRINSIC_GUESS +\
                                                              cv2.CALIB_FIX_PRINCIPAL_POINT +\
                                                              cv2.CALIB_FIX_ASPECT_RATIO +\
                                                              cv2.CALIB_ZERO_TANGENT_DIST+\
                                                              cv2.CALIB_FIX_S1_S2_S3_S4 +\
                                                              cv2.CALIB_FIX_K1 +\
                                                              cv2.CALIB_FIX_K2 +\
                                                              cv2.CALIB_FIX_K3 ) )

    
    return mtx

def get_relative_pose(p_left, p_right, img_left, img_right, K):
    
    left_E_right, mask = cv2.findEssentialMat(p_right, p_left, K, cv2.RANSAC, 0.999, 1.0)
    # right_E_left, mask = cv2.findEssentialMat(p_left, p_right, K, cv2.RANSAC, 0.999, 1.0)


    p_left = p_left[mask.ravel() == 1]
    p_right = p_right[mask.ravel() == 1]


    if DEBUG:
        drawMatches(img_left, p_left, img_right, p_right, "Keypoint correspondences")

    # left_R_right: left to right rotation  
    # left_t_right: left to right translation
    points, left_R_right, left_t_right, mask2 = cv2.recoverPose(left_E_right, p_right, p_left, K) #cheirality check is done internally -> R,t are the correct ones
    # points, right_R_left, right_t_left, mask3 = cv2.recoverPose(right_E_left, p_left, p_right, K) #cheirality check is done internally -> R,t are the correct ones
    

    Mr = np.hstack((left_R_right, left_t_right))
    Ml = np.hstack((np.eye(3, 3), np.zeros((3, 1))))

    Pl = K @ Ml
    Pr = K @ Mr

    return Pl, Pr, Ml, Mr, left_R_right, left_t_right


if __name__ == "__main__":
    pth1 = "./inputs/img1.png"
    pth2 = "./inputs/img2.png"
    pth3 = "./inputs/img3.png"

    img1 = cv2.imread(pth1)
    img2 = cv2.imread(pth2)
    img3 = cv2.imread(pth3)

    R0 = np.array([[1,0,0],
                   [0,1,0],
                   [0,0,1]])

    t0 = np.array([[0],
                   [0],
                   [0]])

    K = getCameraMatrix()

    p1, p2 = featureMatcher(img1, img2)
    P0, P1, M0, M1, R1, t1 = get_relative_pose(p1, p2, img1, img2, K)
    r_temp = Rotation.from_matrix(R1)
    r1_euler = r_temp.as_euler('zyx', degrees=True) 
    # R1: relative rotation of the second camera position w.r.t the first camera position
    # t1: relative translation of the second camera position w.r.t the first camera position(up-to-scale)

    p1, p2 = featureMatcher(img1, img3)
    P0, P2, M0, M2, R2, t2 = get_relative_pose(p1, p2, img1, img3, K)
    r_temp = Rotation.from_matrix(R2)
    r2_euler = r_temp.as_euler('zyx', degrees=True) 
    # R2: relative rotation of the third camera position w.r.t the first camera position
    # t2: relative translation of the third camera position w.r.t the first camera position(up-to-scale)

    f = open("./outputs/results.txt", "w")

    str1 = "Rotation matrix of camera in the second image w.r.t camera in the first image: \n{}\n\n".format(R1)
    str2 = "Rotation of the second camera w.r.t the first camera in euler-anlge(zyx order):\n{}\n\n".format(r1_euler)
    str3 = "Translation vector of camera in the second image w.r.t camera in the first image: \n{}\n\n".format(t1)
    print(str1)
    print(str2)
    print(str3)
    f.write(str1)
    f.write(str2)
    f.write(str3)


    str4 = "Rotation matrix of camera in the third image w.r.t camera in the first image: \n{}\n\n".format(R2)
    str5 = "Rotation of the third camera w.r.t the first camera in euler-anlge(zyx order):\n{}\n\n".format(r2_euler)
    str6 = "Translation vector of camera in the third image w.r.t camera in the first image: \n{}\n\n".format(t2)
    print(str4)
    print(str5)
    print(str6)
    f.write(str4)
    f.write(str5)
    f.write(str6)



    plt.figure()
    plt.plot([0, t1[0], t2[0]], [0, t1[2], t2[2]]); 
    plt.title("Top View")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid()
    plt.savefig("./outputs/top_view.png")

    plt.figure()
    plt.plot([0, t1[0], t2[0]], [0, t1[1], t2[1]]); 
    plt.title("Front View")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.savefig("./outputs/front_view.png")

    plt.figure()
    plt.plot([0, t1[2], t2[2]], [0, t1[1], t2[1]]); 
    plt.title("Side View")
    plt.xlabel("Z")
    plt.ylabel("Y")
    plt.grid()
    plt.savefig("./outputs/side_view.png")

    plt.show()
    


