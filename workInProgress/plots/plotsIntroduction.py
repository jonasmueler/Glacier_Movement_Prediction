#matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from itertools import chain


def draw_map(m, scale=0.2):
    # draw a shaded-relief image
    m.shadedrelief(scale=scale)

    # lats and longs are returned as a dictionary
    lats = m.drawparallels(np.linspace(-90, 90, 13))
    lons = m.drawmeridians(np.linspace(-180, 180, 13))

    # keys contain the plt.Line2D instances
    lat_lines = chain(*(tup[1][0] for tup in lats.items()))
    lon_lines = chain(*(tup[1][0] for tup in lons.items()))
    all_lines = chain(lat_lines, lon_lines)

    # cycle through these lines and set the desired style
    for line in all_lines:
        line.set(linestyle='-', alpha=0.3, color='w')

fig = plt.figure(figsize=(15, 10), edgecolor='w')
m = Basemap(projection='cyl', resolution=None,
            llcrnrlat=-90, urcrnrlat=90,
            llcrnrlon=-180, urcrnrlon=180, )
#m.scatter(-38, 66.4 ,30,marker='x',color='k')
#plt.text(-38, 68.4, "Helheim (66.4°N, -38°E)", fontsize = 16)
m.scatter(8.072999708, 46.438664912, 30,marker='x',color='k')
plt.text(-50, 50, "Jungfrau-Aletsch-Bietschhorn (46.44°N, 8.07°E)", fontsize = 16)
m.scatter( 77.781519, 31.869231, 30,marker='x',color='k')
plt.text(77.781519 -40, 31.869231 +3, "Parvati Glacier (31.87°N, 77.78°E)", fontsize = 16)
#m.scatter(-49.83333, 69.166666, 30,marker='x',color='k')
#plt.text(-155, 72.16, "Jakobshavn (69.16°N, -49.83°E)", fontsize = 16)
draw_map(m)
plt.savefig("worldMap.pdf", dpi = 1000)
plt.show()