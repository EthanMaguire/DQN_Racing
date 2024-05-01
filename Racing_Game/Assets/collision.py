import math
import numpy as np
import pygame


def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def in_rect(rect, x1, y1):
    if rect[0] < x1 < rect[0] + rect[2] and rect[1] < y1 < rect[1] + rect[3]:
        return True
    else:
        return False


def define_rect(p1, p2):
    # Find the top left
    tl = [min(p1[0], p2[0]), min(p1[1], p2[1])]
    width = abs(p1[0] - p2[0])
    height = abs(p1[1] - p2[1])
    return [tl[0], tl[1], width, height]


def cart2pol(pt):
    r = np.sqrt(pt.x ** 2 + pt.y ** 2)
    theta = np.arctan2(pt.y, pt.x)
    return pygame.Vector2(r, theta)


def pol2cart(pt):
    x = pt.x * np.cos(pt.y)
    y = pt.x * np.sin(pt.y)
    return pygame.Vector2(x, y)


def check_poly_bezier(poly, bezier):
    # Checks if a polygon defined counterclockwise by its vertices is in contact with a 3-point Bézier curve
    # 1. Split the polygon into line segments going ccw
    lines = line_list_from_poly(poly)
    # 2. Check if any of these are inside the bezier P0, P1, P2 triangle
    # Using determinant method from https://math.stackexchange.com/questions/2378378/determine-if-a-line-passes-through-a-triangle
    # for triangle [v0, v1, v2], and line [p0, p1]
    # take, det(p0, p1, v0), det(p0, p1, v1), det(p0, p1, v2)
    # if one sign is different then the line passes through the triangle
    for line in lines:
        det1 = np.linalg.det([line[0], line[1], bezier[0]])
        det2 = np.linalg.det([line[0], line[1], bezier[1]])
        det3 = np.linalg.det([line[0], line[1], bezier[2]])
        print("Unfinished Code")
        raise SystemExit

    # 3. If so, subdivide the bezier into N line segments
    # 4. Check Line-Line collisions with each of these segments (self.check_poly_lines)
    # 5. Return True if a collision has occurred
    return False


def check_poly_lines(poly_points, check_points, closed_poly=True, full_search=False, self_check=False, self_skip=5):
    # Test collision between a polygon defined CW by its vertices and
    # a list of connect points (Not a closed loop)
    # 1. Check if a rectangular bounding box of the two sets of vertices is overlapping (Very large speedup)
    hit_lines = []  # set of lines that were hit
    # Bounding box is touching, do a line by line test
    poly_lines = line_list_from_poly(poly_points, closed=closed_poly)
    check_lines = line_list_from_poly(check_points, closed=False)
    for i, p_line in enumerate(poly_lines):
        for j, c_line in enumerate(check_lines):
            if self_check:  # Checking the for self intersection of a single polygon
                if j <= i + self_skip or j >= len(check_lines):
                    continue
            if do_intersect(p_line[0], p_line[1], c_line[0], c_line[1]):
                if full_search:
                    hit_lines.append(c_line)
                else:
                    return True, hit_lines

    if len(hit_lines) == 0:
        return False, hit_lines  # no collision detected
    else:
        return True, hit_lines


def line_list_from_poly(poly, closed=False):
    poly_lines = []  # list of pairs of points
    for i, pt in enumerate(poly):
        if i == 0:
            if closed:  # Include point back to start
                poly_lines.append([poly[-1], pt])  # Connect to starting point
            if not closed:  # skip return to start
                continue
        elif i == len(poly) - 1:  # dont overindex
            break
        else:  # Not on last point
            poly_lines.append([pt, poly[i + 1]])  # Connect to next point

    return poly_lines


def on_segment(p, q, r):
    # Given three collinear points p, q, r, the function checks if
    # point q lies on-line segment 'pr'
    if ((q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and
            (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):
        return True
    return False


def orientation(p, q, r):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Collinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise
    val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1]))
    if val > 0:
        # Clockwise orientation
        return 1

    elif val < 0:
        # Counterclockwise orientation
        return 2

    else:
        # Collinear orientation
        return 0


def do_intersect(p1, q1, p2, q2):
    # Test if two line segments intersect using the method from
    # https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
    # Find the 4 orientations required for
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if (o1 != o2) and (o3 != o4):
        return True

    # Special Cases

    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if (o1 == 0) and on_segment(p1, p2, q1):
        return True

    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if (o2 == 0) and on_segment(p1, q2, q1):
        return True

    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if (o3 == 0) and on_segment(p2, p1, q2):
        return True

    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if (o4 == 0) and on_segment(p2, q1, q2):
        return True

    # If none of the cases
    return False


# Returns true if two rectangles(r1, r1) (Top left and bottom right corners)
# and (l2, r2) overlap using the Top left and bottom right corners (r1[0], r1[1]), (r2[0], r2[1])
def check_rect_rect(r1, r2):
    # if rectangle has area 0, no overlap
    if r1[0][0] == r1[1][0] or r1[0][1] == r1[1][1] or r2[0][0] == r2[1][0] or r2[0][1] == r2[1][1]:
        return False

    # If one rectangle is on left side of other
    if r1[0][0] > r2[1][0] or r2[0][0] > r1[1][0]:
        return False

    # If one rectangle is above other
    if r1[1][1] < r2[0][1] or r2[1][1] < r1[0][1]:
        return False

    return True


def get_corners(points):  # gets the cartesian corners of a set of points
    return [pygame.Vector2(min([x[0] for x in points]), min([x[1] for x in points])),
            pygame.Vector2(max([x[0] for x in points]), max([x[1] for x in points]))]


def get_line_line_intersect(h1, h2, g1, g2):  # Get the intercept between lines h and g
    mh = (h2[1] - h1[1]) / (h2[0] - h1[0])
    mg = (g2[1] - g1[1]) / (g2[0] - g1[0])
    if mh == mg:  # check if they are parallel
        if h1[1] == g1[1]:  # Same line
            return 0
        else:
            return math.inf  # Never touch

    x = (g1[1] - h1[1] + mh * h1[0] - mg * g1[0]) / (mh - mg)
    y = mh * (x - h1[0]) + h1[1]
    return pygame.Vector2(x, y)


def check_pt_in_segment(pt, line):  # Checks if a point is within a line segments rectangular bounds
    corners = get_corners(line)
    if corners[0][0] < pt[0] < corners[1][0] and corners[0][1] < pt[1] < corners[1][1]:
        return True
    else:
        return False


def check_pt_corners(pt, corners):
    rect = define_rect(corners[0], corners[1])
    return in_rect(rect, pt[0], pt[1])



