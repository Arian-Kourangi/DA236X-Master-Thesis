from shapely.geometry import Point, Polygon, LineString
import numpy as np

def signed_distance_point_to_polygon(pt, lb,ub):
    """
    Compute the signed distance from a point to a rectangle defined by lb and ub.
    Negative distance indicates the point is inside the rectangle.
    Args:
        pt: numpy array of shape (2,), the point (x,y)
        lb: numpy array of shape (2,), the lower bound (x_min, y_min)
        ub: numpy array of shape (2,), the upper bound (x_max, y_max)
    Returns:
        signed_distance: float, the signed distance from the point to the rectangle
    """
    poly = Polygon([(lb[0], lb[1]), (ub[0], lb[1]), (ub[0], ub[1]), (lb[0], ub[1])])
    # pt: (x,y)
    p = Point(pt)
    if poly.contains(p):
        # distance to boundary, negative inside
        return -p.distance(poly.boundary)
    else:
        return p.distance(poly)


# pt= np.array([2.0, 2.0])
# lb = np.array([1.0, 1.0])
# ub = np.array([3.0, 3.0])
# print(signed_distance_point_to_polygon(pt, lb, ub))  # Expected output: -1.0