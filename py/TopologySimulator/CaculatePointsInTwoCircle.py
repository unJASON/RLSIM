import sympy
import math
def caculatePointsInTwoCircle(xa,ya,ra,xb,yb,rb):
    xa = round(xa, 4)
    ya = round(ya, 4)
    ra = round(ra, 4)
    xb = round(xb, 4)
    yb = round(yb, 4)
    rb = round(rb, 4)
    distance = math.sqrt((xb - xa) ** 2 + (yb - ya) ** 2)
    min_rad = min(ra, rb)
    max_rad = max(ra, rb)
    if distance > ra + rb:
        return ((xa + (xb - xa) * ra / distance, ya + (yb - ya) * ra / distance), (xb + (xa - xb) * rb / distance, yb + (ya - yb) * rb / distance) )
    if distance + min_rad <= max_rad:
        if ra < rb:
            return ((xb + (xa - xb) * (distance+ra) / distance, yb + (ya - yb) * (distance+ra) / distance),(xb + (xa - xb) * (distance+ra) / distance, yb + (ya - yb) * (distance+ra) / distance))
        else:
            return ((xa + (xb - xa) * (distance+rb) / distance, ya + (yb - ya) *  (distance+rb) / distance), (xa + (xb - xa) * (distance+rb) / distance, ya + (yb - ya) *  (distance+rb) / distance))
    x, y = sympy.symbols('x y')
    f1 = (xa-x)**2 + (ya-y)**2 - ra**2
    f2 = (xb-x)**2 + (yb-y)**2 - rb**2
    result = sympy.solve([f1,f2],[x,y],quick=True)
    return (float(result[0][0]),float(result[0][1])),(float(result[1][0]),float(result[1][1]))

if __name__ == '__main__':
    # xa, ya, ra, xb, yb, rb = 0,0,1,1,0,1
    # xa, ya, ra, xb, yb, rb = 0, 0, 1, 2, 0, 0.9
    # xa, ya, ra, xb, yb, rb = 0, 0, 3.745750390290433, -1.287685900577545, -2.2034835007939715, 1.177420042080498
    xa, ya, ra, xb, yb, rb = 0.0, 0.0, 3.874443915924794, 3.9931837647806647, 0.0, 3.970127090124699
    # xa, ya, ra, xb, yb, rb = 0.0, 0.0, 3.87, 3.99, 0.0, 3.97
    print( caculatePointsInTwoCircle(xa, ya, ra, xb, yb, rb) )