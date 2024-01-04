from blueprints import *
from objects.nested_shape import NestedShape
import matplotlib.pyplot as plt

blueprint = generate_nested_regular_polygon_blueprint(3, size=2.0)

print("Blueprint:")
for item in blueprint:
    print(item)

nested_shape = NestedShape(blueprint)

fig, ax = plt.subplots()

ax = nested_shape.plot(ax)

# Make the aspect ratio equal
ax.set_aspect("equal")
plt.show()
