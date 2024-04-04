from blueprints import *
from objects.nested_shape import NestedShape
import matplotlib.pyplot as plt

blueprint = generate_nested_regular_polygon_blueprint(5, size=3.0)
# blueprint = strange_polygon_blueprint

print("Blueprint:")
for item in blueprint:
    print(item)

nested_shape = NestedShape(blueprint)

fig, ax = plt.subplots()

ax = nested_shape.plot(ax)

# Make the aspect ratio equal
ax.set_aspect("equal")
plt.show()
