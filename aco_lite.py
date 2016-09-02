# imports
import numpy as np
import cv2

def distance_between_points(point_1, point_2):
    """ Find distance between points sqrt((x2 - x1)**2 + (y2 - y1)**2). """
    distance = int(np.sqrt((point_1[0] - point_2[0])**2 +
                           (point_1[1] - point_2[1])**2))
    return distance


def get_line(point_1, point_2):
    ''' Find the equation of the line between the two points. '''
    # find slope (y2 - y1) / (x2 - x1)
    m = (point_2[1] - point_1[1]) / float((point_2[0] - point_1[0]))
    # find intercept b = y - m*x
    b = point_1[1] - m * point_1[0]
    return m, b


def draw_bounding_rectangle(img, contours, color=(0, 0, 255), thickness=2):
    """ A function to draw the bounding rectangle. """
    x, y, w, h = cv2.boundingRect(contours)
    cv2.rectangle(img_color, (x, y), (x+w, y+h), color, thickness)
    return {'x': x, 'y': y, 'width': w, 'height': h}


def draw_min_bounding_box(contours):
    rect = cv2.minAreaRect(contours)
    box = cv2.boxPoints(rect)
    box = np.int0(box) # convert the coordinates to int
    cv2.drawContours(img_color, [box], 0, (0, 255, 0), 2)


def draw_path(img, rect, offset, length_offset=15, color=(0, 255, 0)):
    """ Function to draw the ACO paint path. """
    # setup variables
    points = []
    width = min([rect['width'], rect['height']])
    height = max([rect['width'], rect['height']])
    number_of_switch_backs = int(np.ceil(width / float(offset)))
    p1_offset = ((number_of_switch_backs * offset) - width) / 2
    current_point = (rect['x'] - length_offset, rect['y'] - p1_offset)
    cv2.circle(img, current_point, 4, color, -1)
    direction_controller = True
    points.append(current_point)

    # loop through points to draw path
    for point in range(1, 2 * number_of_switch_backs + 2):
        if point % 2 == 0:
            next_point = (current_point[0], current_point[1] + offset)
        else:
            if direction_controller:
                next_point = (
                    current_point[0] + height + (2*length_offset),
                    current_point[1])
                direction_controller = False
            else:
                next_point = (
                    current_point[0] - height - (2*length_offset),
                    current_point[1])
                direction_controller = True
        cv2.circle(img, next_point, 4, color, -1)
        cv2.line(img, current_point, next_point, color, 2)
        current_point = next_point
        points.append(next_point)

    return points


# read in color image and convert a copy to grayscale
img_color = cv2.imread('images/outsole_test.jpg')
# img_color = cv2.imread('images/outsole_rotated.png')
img_grayscale = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
cv2.imshow('Original', img_grayscale)
cv2.waitKey(0)
cv2.destroyAllWindows()

# threshold and invert the image
ret, thresh = cv2.threshold(img_grayscale, 250, 255, cv2.THRESH_BINARY)
thresh = (255-thresh) # invert the image so that the outsole is white
cv2.imshow('Threshold Image', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

# find contours in the image
im2, contours, hierarchy = cv2.findContours(thresh, 1, 2)

# Find the index of the largest contour
areas = [cv2.contourArea(c) for c in contours]
max_index = np.argmax(areas)
cnts=contours[max_index]

# find the moments of the contours
M = cv2.moments(cnts)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])
cv2.circle(img_color, (cX, cY), 7, (255, 0, 0), -1)

# draw the bounding rectangle
rect = draw_bounding_rectangle(img_color, cnts)
points = draw_path(img_color, rect, 30)
print points


# show final image with rectangle
cv2.imshow('Test', img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
