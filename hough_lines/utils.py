import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage.filters import generic_filter, maximum_filter, minimum_filter

def Sobel(img, threshold=50):
    def kernel(p):
        return (np.abs((p[0] + 2 * p[1] + p[2]) - (p[6] + 2 * p[7] + p[8])) +
                np.abs((p[2] + 2 * p[6] + p[7]) - (p[0] + 2 * p[3] + p[6])))
    
    res = generic_filter(img, kernel, (3, 3))
    res = np.array(list(map(lambda x: [255 if xx > threshold else 0 for xx in x], res)), dtype=res.dtype)
    return res

def HoughSpace(img_edges, rho=1, theta=np.pi/180/10):
    x_max, y_max = img_edges.shape
    theta_max = np.pi
    r_max = np.hypot(x_max, y_max)
    r_dim = int(r_max/rho)
    theta_dim = int(theta_max/theta)    
    hough_space = np.zeros((r_dim, theta_dim))
    for x in range(x_max):
        for y in range(y_max):
            if img_edges[x][y]:
                for theta_i in range(theta_dim):
                    theta = theta_i*theta_max/theta_dim
                    r = x*np.cos(theta) + y*np.sin(theta)
                    r_i = int(r_dim*r/r_max)
                    hough_space[r_i][theta_i] = hough_space[r_i][theta_i] + 1
    return hough_space  # thetas, rhos

def HoughLines(hough_space, threshold=100, neighborhood_size=20):    
    data_max = maximum_filter(hough_space, neighborhood_size)
    maxima = (hough_space == data_max)
    data_min = minimum_filter(hough_space, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    thetas, rhos = [], []
    for d_rho, d_theta in slices:
        theta_center = (d_theta.start + d_theta.stop - 1)/2
        thetas.append(theta_center)
        rho_center = (d_rho.start + d_rho.stop - 1)/2    
        rhos.append(rho_center)
    return thetas, rhos

def plot_hs(hs):
    plt.imshow(hs)
    r_dim, theta_dim = hs.shape
    plt.ylim(0, r_dim)
    tick_locs = [i for i in range(0, theta_dim, int(theta_dim/20))]
    tick_lbls = [round((i*np.pi)/theta_dim, 1) for i in range(0, theta_dim, int(theta_dim/20))]
    plt.xticks(tick_locs, tick_lbls)
    tick_locs = [i for i in range(0, r_dim, int(r_dim/10))]
    tick_lbls = [round(i, 1) for i in range(0, r_dim, int(r_dim/10))]
    plt.yticks(tick_locs, tick_lbls)
    plt.xlabel(r'Theta')
    plt.ylabel(r'rho')
    plt.title('Hough Space')
    plt.show()
