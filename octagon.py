# Importing libraries
import cv2
import numpy as np 
from matplotlib import pyplot as plt
import time

start = time.time()

# function to plot lines
def drawlines(img1, img2, lines, pts1, pts2):
	
	r, c  = img1.shape
	img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
	img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
	
	for r, pt1, pt2 in zip(lines, pts1, pts2):
		
		color = tuple(np.random.randint(0, 255,	3).tolist())
		
		x0, y0 = map(int, [0, -r[2] / r[1] ])
		x1, y1 = map(int,[c, -(r[2] + r[0] * c) / r[1] ])
		
		img1 = cv2.line(img1,(x0, y0), (x1, y1), color, 3)
		img1 = cv2.circle(img1,	tuple(pt1), 15, color, -1)
		img2 = cv2.circle(img2,	tuple(pt2), 15, color, -1)
	return img1, img2

# function to estimate fundamental matrix
def estimate_fundamental(k_A , k_B) :
    A = np.array([ [k_A[0][0]*k_B[0][0] , k_A[0][0]*k_B[0][1] , k_A[0][0] , k_A[0][1]*k_B[0][0] , k_A[0][1]*k_B[0][1], k_A[0][1] , k_B[0][0] , k_B[0][1] , 1 ] ,
                [k_A[1][0]*k_B[1][0] , k_A[1][0]*k_B[1][1] , k_A[1][0] , k_A[1][1]*k_B[1][0] , k_A[1][1]*k_B[1][1], k_A[1][1] , k_B[1][0] , k_B[1][1] , 1 ] ,
                [k_A[2][0]*k_B[2][0] , k_A[2][0]*k_B[2][1] , k_A[2][0] , k_A[2][1]*k_B[2][0] , k_A[2][1]*k_B[2][1], k_A[2][1] , k_B[2][0] , k_B[2][1] , 1 ] ,
                [k_A[3][0]*k_B[3][0] , k_A[3][0]*k_B[3][1] , k_A[3][0] , k_A[3][1]*k_B[3][0] , k_A[3][1]*k_B[3][1], k_A[3][1] , k_B[3][0] , k_B[3][1] , 1 ] ,
                [k_A[4][0]*k_B[4][0] , k_A[4][0]*k_B[4][1] , k_A[4][0] , k_A[4][1]*k_B[4][0] , k_A[4][1]*k_B[4][1], k_A[4][1] , k_B[4][0] , k_B[4][1] , 1 ] ,
                [k_A[5][0]*k_B[5][0] , k_A[5][0]*k_B[5][1] , k_A[5][0] , k_A[5][1]*k_B[5][0] , k_A[5][1]*k_B[5][1], k_A[5][1] , k_B[5][0] , k_B[5][1] , 1 ] ,
                [k_A[6][0]*k_B[6][0] , k_A[6][0]*k_B[6][1] , k_A[6][0] , k_A[6][1]*k_B[6][0] , k_A[6][1]*k_B[6][1], k_A[6][1] , k_B[6][0] , k_B[6][1] , 1 ] ,
                [k_A[7][0]*k_B[7][0] , k_A[7][0]*k_B[7][1] , k_A[7][0] , k_A[7][1]*k_B[7][0] , k_A[7][1]*k_B[7][1], k_A[7][1] , k_B[7][0] , k_B[7][1] , 1 ] 
    ] )
    U , S , V = np.linalg.svd(A)
    F = V[-1,:] 
    print(np.shape(V))    
    F = np.reshape(F , (3,3))
    u, s, vt = np.linalg.svd(F)
    s = np.diag(s)
    s[2, 2] = 0
    F = u @ (s @ vt)
    F = F / F[-1,-1]
    return np.round(F ,decimals =2) 

# Function to estimate Essential matrix
def estimate_essential(F):
    K = np.array([ [1742.11 , 0 , 804.9] ,
                    [0 , 1742.11 , 541.22] ,
                    [0 , 0 , 1]])
    K_t = K.transpose()  
    R = np.matmul(F,K)       
    E = np.matmul(K_t , R)
    print("Essential matrix")
    print(E)
    return(E)

# Function to determine all possible camera poses
def camera_pose(E):
    W = np.array([ [0 , -1 , 0] ,
                    [1 , 0 , 0] ,
                    [0 , 0 , 1]])

    U , D , V = np.linalg.svd(E) 
    C1 = np.reshape(U[:,-1] , (3,1))

    left1 = np.matmul(U , W)
    R1 = np.matmul(left1 , V)
    C2 = np.reshape(-U[:,-1] , (3,1))
    left2 = np.matmul(U , W.transpose())
    R3 = np.matmul(left2 , V)
    return C1 , R1 , C2 , R1 , C1 ,R3 , C2 , R3



# To resize frame
def resize(frame , x , y):
    dimensions = ( int(frame.shape[1]*x) , int(frame.shape[0]*y))
    return cv2.resize(frame, dimensions , interpolation = cv2.INTER_AREA)

# Function to calculate disparity and depth using SSD
def disparity(img1_rectified , img2_rectified):
    # Blank image
    disparity = np.zeros_like(img1_rectified , dtype=np.float64)

    # window traversing through image, with window size = 5X5 and stride = 1 pixel
    for h in range( img1_rectified.shape[0]-9 ) :
        for w in range( img1_rectified.shape[1]-9  ) :

            window_left = np.array(img2_rectified[h:h+5 , w:w+5])
            # 1D array to store SSD of each window of right image
            ssd_store = np.zeros((1 , 40 )) 
            index_ssd = 0

            # loop for doing search in other image, within a given search region
            for j in range( -20 , 20 ) :

                if  w>20 and (j+w)<951:
                    window_right = np.array(img1_rectified[h:h+5 , j+w:j+w+5])
                    result1 = np.subtract(window_left ,window_right) 
                    result1_sum = np.sum(np.square(result1))
                    ssd_store[0 , index_ssd] = result1_sum
                    index_ssd = index_ssd +1

                elif w<=20 :
                    window_right = np.array(img1_rectified[h:h+5 , j+20:j+25])
                    result1 = np.subtract(window_left ,window_right) 
                    result1_sum = np.sum(np.square(result1))
                    ssd_store[0 , index_ssd] = result1_sum
                    
                    index_ssd = index_ssd +1

            # Determining index of min SSD
            min_ssd_index = np.argmin(ssd_store)
            # Calculating disparity between images by taking difference between first column numbers of pixels of each window

            disparity[h , w] = min_ssd_index -20

    disparity = disparity+np.abs(np.min(disparity)) 
    depth_disparity = disparity + 1

    return np.uint8( (disparity/np.max(disparity))*255) ,depth_disparity



# Reading and resizing frames
img_A = cv2.imread("octagon/im0.png", cv2.IMREAD_COLOR)

# plt_A = cv2.cvtColor(img_A , cv2.COLOR_BGR2RGB)
resized_A = resize(img_A , 0.5 , 0.5)

img_B = cv2.imread("octagon/im1.png", cv2.IMREAD_COLOR)
# plt_B = cv2.cvtColor(img_B , cv2.COLOR_BGR2RGB)
resized_B = resize(img_B , 0.5 , 0.5)

# Convert to grayscale
gray_A= cv2.cvtColor(resized_A,cv2.COLOR_BGR2GRAY)
  
gray_B= cv2.cvtColor(resized_B,cv2.COLOR_BGR2GRAY)
# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints with ORB
kp_A = orb.detect(gray_A,None)
kp_B = orb.detect(gray_B,None)
# compute the descriptors with ORB
kp_A, des_A = orb.compute(gray_A, kp_A)
kp_B, des_B = orb.compute(gray_B, kp_B)

# Initialize BF matcher
bf = cv2.BFMatcher()
# Match descriptors.
matches = bf.knnMatch(des_A,des_B,k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.4*n.distance:
        good.append(m)  

final_img = cv2.drawMatches(resized_A, kp_A, resized_B, kp_B , good[:],None )
cv2.imshow("Matches", final_img)

list_kpA = []
list_kpB = []

list_kpA = [kp_A[mat.queryIdx].pt for mat in good] 
list_kpB = [kp_B[mat.trainIdx].pt for mat in good]

# Fundamental matrix
F_matrix = estimate_fundamental(list_kpA , list_kpB )

print("Rank of fundamental matrix")
print(np.linalg.matrix_rank(F_matrix))

# Determining essential matrix
E_matrix = estimate_essential(F_matrix)
C1, R1 , C2 , R2 , C3 , R3 , C4 , R4 = camera_pose(E_matrix)
print("Possible camera configs")
print(C1, R1)
print(C2, R2)
print(C3, R3)
print(C4, R4)
x_coord = list_kpA[0][0] 
y_coord = list_kpA[0][1] 
# correct_pose( C1, R1 , C2 , R2 , C3 , R3 , C4 , R4 , x_coord , y_coord)

list_kpA = np.int32(list_kpA)
list_kpB = np.int32(list_kpB)
# Find epilines corresponding to points in right image (second image) and drawing its lines on left image
linesLeft = cv2.computeCorrespondEpilines(list_kpB.reshape(-1, 1,2),2, F_matrix)
linesLeft = linesLeft.reshape(-1, 3)

img5, img6 = drawlines(gray_A, gray_B, linesLeft, list_kpA, list_kpB)
# Find epilines corresponding to points in left image (first image) and drawing its lines on right image
linesRight = cv2.computeCorrespondEpilines(list_kpA.reshape(-1, 1, 2), 1, F_matrix)
linesRight = linesRight.reshape(-1, 3)

img3, img4 = drawlines(gray_B, gray_A, linesRight, list_kpB,list_kpA)


h1, w1 ,_ = resized_A.shape
h2, w2, _ = resized_B.shape
_, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(list_kpA), np.float32(list_kpB), F_matrix, imgSize=(w1, h1))
print("H1: homography")
print(H1)
print("H2: homography")
print(H2)
# Image rectified
img1_rectified = cv2.warpPerspective(gray_A, H1, (w1, h1))
img2_rectified = cv2.warpPerspective(gray_B, H2, (w2, h2))


cv2.imshow("Image0 rectified", img1_rectified)
cv2.imshow("Image1 rectified", img2_rectified)

print("start")

# Disparity 
disparity , depth_dis = disparity(img1_rectified  , img2_rectified )
cv2.imshow("disparity" , disparity)
# Depth estimation
depth_dr = np.reciprocal(depth_dis+30 , dtype=np.float64)
depth = 221.76 * 1742.11 * depth_dr
depth = np.uint8((depth/np.max(depth))*255)
cv2.imshow("depth map" , depth)

#Heat map for disparity
heatmap_disparity = cv2.applyColorMap(disparity , cv2.COLORMAP_JET)
cv2.imshow("heat_map_disparity" , heatmap_disparity)

#Heat map for depth
heatmap_depth = cv2.applyColorMap(depth , cv2.COLORMAP_JET)
cv2.imshow("depth_heatmap" , heatmap_depth)

print("end")
end = time.time()
print("time taken to run : "+  str(end-start)+ " seconds ")  


cv2.waitKey(0)
plt.subplot(121), plt.imshow(img5)
plt.subplot(122), plt.imshow(img3)
plt.show() 
 
# It is for removing/deleting created GUI window from screen
# and memory
cv2.destroyAllWindows()