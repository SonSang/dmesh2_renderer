### Error codes for Anti-Aliasing implementation

* E00: Intersection point coincides with pixel vertex.
* E01: More than two intersection points are found.
* E02: There is one intersection point, but there is no end point of the triangle edge that resides in the pixel.
* E03: There was one end point of the triangle edge that resides in a pixel, while the other end point does not, there was no intersection point.
* E04: Subtriangle of polygon is not CCW order.
* E05: Number of polygon vertices exceeds 10, which is theoretically not possible.