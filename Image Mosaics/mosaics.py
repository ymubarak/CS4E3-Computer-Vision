import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import cv2

# Manually identifies corresponding points from two views
def get_correspondences(image1, image2, num_of_Points=8, manual_selection_=True):
    if(num_of_Points < 8):
        print("Error: Num of paris must be greater or equal 4")
        return
    
    if(manual_selection_):
        # Display images, select matching points
        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)
        # Display the image
        ax1.imshow(image1)
        ax2.imshow(image2)
        plt.axis('image')
        pts = plt.ginput(n=num_of_Points)
        pts = np.reshape(pts, (2, int(num_of_Points/2), 2))
        # print(pts);
        return pts
    else:
        """ automatic selection process """

        img1 = cv2.imread('../images/mountain/image1.jpg',0)
        img2 = cv2.imread('../images/mountain/image2.jpg',0)

        # Initiate SIFT detector
        orb = cv2.ORB()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des1, des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        return matches
        # p to p corespondence points 
        pts = np.reshape(pts, (2, int(num_of_Points/2), 2))
        return pts


# Return x, y coordinates of point index from the selected points in image 1
src_ponit = lambda pts, index: (pts[0][index][0], pts[0][index][1])
# Return x, y coordinates of matched point index from image 2 
dest_ponit = lambda pts, index: (pts[1][index][0], pts[1][index][1])

# Compute sub matrix of A for a certain point and its matched one
def get_submatrix(x, y, x_, y_):
    submatrix = np.zeros((2,8))
    submatrix[0] = np.array([x, y,1,0,0,0, -x*x_, -y*x_])
    submatrix[1] = np.array([0,0,0,x, y,1, -x*y_, -y*y_])
    return submatrix

def to_homogenous_coords(h):
    # reshape h to matrix 3x3
    h_homo = np.zeros((3,3))
    for i in range(0,3):
        for j in range(0,3):
            if 3*i+j < 8:
                h_homo[i][j] = h[3*i+j]
    h_homo[2][2] = 1
    return h_homo

# compute x_ from x point using transform h
def transform(point, h):
    # compute x_
    x_ = np.dot(h, point)

    # normalize by dividing by w
    x_[0] = x_[0]/x_[2]
    x_[1] = x_[1]/x_[2]
    x_[2] = 1
    return x_

# compute x from x_ point using transform h
def inv_transform(point, h):
    # compute x
    h_inv = np.linalg.inv(h)
    ret = np.dot(h_inv, point)

    # normalize by dividing by w
    x[0] = x[0]/x[2]
    x[1] = x[1]/x[2]
    x[2] = 1
    return x

# calcuate H matrix using n matched points from get_correspondence
def calc_homograph(pts):
    # SOLVE A h = b
    n = int(pts.size/4) # find number of points
    
    # build A
    A = np.zeros((2*n, 8))
    for i in range(0, n):
        submatrix = get_submatrix(src_point(pts, i), dest_point(pts, i))
        A[2*i] = submatrix[0]
        A[2*i + 1] = submatrix[1]
    
    # build b
    b = np.zeros((2*n, 1))
    for i in range(0, n):
        b[2*i], b[2*i+1] = dest_point(pts, i)

    #solve equation
    h = np.linalg.lstsq(A, b)[0]
    print('A: {}\nh: {}\nb: {}'.format(A.shape, h.shape, b.shape))
    return h


def get_trans_info(source_image):
    # Get a new source_image with (2*width, 2*height), and get transform info to destination
    trans_info = {}
    trans_info['height'] = source_image.shape[0]
    trans_info['width'] = source_image.shape[1]

    min_mapped_i = min_mapped_j = int(100000)
    max_mapped_i = max_mapped_j = int(-100000)

    # calculate corners of transformed image
    print('Image A size ' + str(source_image.shape))

    # calcuate transformed corners positions
    corners = np.array([
        [0,0],
        [trans_info['height']-1, 0],
        [0, trans_info['width']-1],
        [trans_info['height']-1, trans_info['width']-1]
        ]);
    
    for k in range(0,4):
        i = corners[k][0]
        j = corners[k][1];

        mapped_positions = transform(np.array([[j],[i],[1]]), h);
        mapped_i = int(mapped_positions[1][0])
        mapped_j = int(mapped_positions[0][0])

        # update corners
        if mapped_i < min_mapped_i:
            min_mapped_i = mapped_i
        if mapped_i > max_mapped_i:
            max_mapped_i = mapped_i
        if mapped_j < min_mapped_j:
            min_mapped_j = mapped_j
        if mapped_j > max_mapped_j:
            max_mapped_j =mapped_j
    
    trans_info['new_height'] = (max_mapped_i-min_mapped_i+1);
    trans_info['new_width'] = (max_mapped_j-min_mapped_j+1);

    trans_info['shift_height'] = -min_mapped_i;
    trans_info['shift_width'] = -min_mapped_j;
    
    return trans_info


def wrap_image(source_image, h, allow_holes=True):
    trans_info = get_trans_info(source_image)
    output_image = np.zeros((trans_info['new_height'], trans_info['new_width'], 3),
                            dtype=np.uint8);
    print('Destination image size:', str(output_image.shape))
    if allow_holes:
        # method1: transform each pixels from dest_image to source_image,
        # which may cause black holes
        for i in range(0, trans_info['height']):
            for j in range(0, trans_info['width']):
                mapped_positions = transform(np.array([[j],[i],[1]]), h);
                mapped_j = int(mapped_positions[0][0])
                mapped_i = int(mapped_positions[1][0])
                x = mapped_i + trans_info['shift_height']
                y = mapped_j + trans_info['shift_width']
                output_image[x][y] = source_image[i][j];

        result = Image.fromarray(output_image)
        return result
    else:        
        # method2: calculate for each pixel in image1 the correspondence in dest_image
        # we need to interpolate, it removes the black holes
        for i in range(0, trans_info['new_height']):
            for j in range(0, trans_info['new_width']):
                # may be done in a more neat way!
                if int(output_image[i][j][0]) == 0 and int(output_image[i][j][1]) == 0 and int(output_image[i][j][2]) == 0:
                    # it's black let's get back to it's inverse!
                    invMappedPos = inv_transform(np.array([[(j - shift_width)], [(i - shift_height)],[1]]), h)
                    invMappedJ = invMappedPos[0][0]
                    invMappedI = invMappedPos[1][0]
                    if invMappedI <= height-1 and  invMappedI >= 0 and invMappedJ <= width-1 and invMappedJ >= 0:
                        # using bilinear interpolation
                        low_i = int(invMappedI);
                        low_j = int(invMappedJ);
                        dist_i = invMappedI - low_i;
                        dist_j = invMappedJ - low_j;
                        output_image[i][j] =                         (1-dist_i)*(1-dist_j)*source_image[low_i][low_j] +                         (1-dist_i)*(dist_j)*source_image[low_i][low_j+1] +                         (dist_i)*(1-dist_j)*source_image[low_i+1][low_j] +                         (dist_i)*(dist_j)*source_image[low_i+1][low_j+1]

        result = Image.fromarray(output_image)
        return result



def merge(warped_image, dest_image):
   trans_info = get_trans_info(source_image)
   # merge the warped image and the reference one
   dest_imageHeight = dest_image.shape[0]
   dest_imageWidth = dest_image.shape[1]

   # calculate merged width and height
   mergedImageHeight = dest_imageHeight + trans_info['shift_height']
   if new_height > mergedImageHeight:
       mergedImageHeight = new_height

   mergedImageWidth = dest_imageWidth + trans_info['shift_width']
   if new_width > mergedImageWidth:
       mergedImageWidth = new_width

   # make a new image of the new width and height
   mergedImage = np.zeros((mergedImageHeight, mergedImageWidth, 3), dtype=np.uint8);

   # sketch the reference image
   for i in range(0, dest_imageHeight):
       for j in range(0, dest_imageWidth):
           x = i + trans_info['shift_height']
           y = j + trans_info['shift_width']
           mergedImage[x][y] = dest_image[i][j]

   # sketch the destination image (warped image)
   for i in range(0, trans_info['new_height']):
       for j in range(0, trans_info['new_width']):
           if not( int(warped_image[i][j][0]) == 0 and                int(warped_image[i][j][0]) == 0 and                int(warped_image[i][j][2]) == 0 ):
               mergedImage[i][j] = warped_image[i][j]

   result = Image.fromarray(mergedImage)
   return result


def main():
    file1, file2 = 'images/mountain/image1.jpg', 'images/mountain/image2.jpg'

    image1, image2 = mpimg.imread(file1), mpimg.imread(file2)
        # get correspondences
    pts = get_correspondences(image1, image2, manual_selection_=True)
    # print(pts)

    # calculate h
    h = calc_homograph(pts)
    h_homo = to_homogenous_coords(h)

    # warp source image
    wraped = wrap_image(image1, h_homo, allow_holes=True)
    # merge results
    result = merge(wraped, image2, image1)

    # plot result
    plt.imshow(image1)
    plt.title('Source image')
    plt.show()

    plt.imshow(image2)
    plt.title('Dest image')
    plt.show()

    plt.imshow(result)
    plt.title('Result')
    plt.show()

