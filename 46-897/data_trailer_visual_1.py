import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl

# Coordinates and reimbursement gap values
country_coords = {
    "United States": (-95.7129, 37.0902),
    "Canada": (-106.3468, 56.1304),
    "Germany": (10.4515, 51.1657),
    "India": (78.9629, 20.5937),
    "Brazil": (-51.9253, -14.2350),
    "United Kingdom": (-3.4360, 55.3781),
    "Japan": (138.2529, 36.2048)
}

gaps = {
    "United States": 3600,
    "Canada": 2400,
    "Germany": 1800,
    "India": 1700,
    "Brazil": 1100,
    "United Kingdom": 950,
    "Japan": 1800
}

gap_values = list(gaps.values())
alpha_map = {
    country: 0.4 + 0.6 * ((gap - min(gap_values)) / (max(gap_values) - min(gap_values)))
    for country, gap in gaps.items()
}

fig = plt.figure(figsize=(14, 8))
m = Basemap(projection='merc', llcrnrlat=-60, urcrnrlat=80,
            llcrnrlon=-180, urcrnrlon=180, lat_ts=20, resolution='l')

m.drawcoastlines(linewidth=0.3, color='gray')
m.drawcountries(linewidth=0.5, color='gray')
m.drawmapboundary(fill_color='#E6F2FA')
m.fillcontinents(color='white', lake_color='#E6F2FA')

# Color scale
vmin, vmax = 0, 5000
cmap = plt.cm.Reds
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

# Plot markers
for country, (lon, lat) in country_coords.items():
    x, y = m(lon, lat)
    gap = gaps[country]
    size = gap / 120
    alpha = alpha_map[country]
    color = cmap(norm(gap))
    m.plot(x, y, 'o', markersize=size, color=color, alpha=alpha)
    plt.text(x, y - 1e6, f"{country}\n${gap:,}", fontsize=9,
             ha='center', va='top', color='#1A1A1A', fontweight='bold')

# Add colorbar (bottom left)
ax = plt.gca()
cax = inset_axes(ax, width="2%", height="30%", loc='lower left', borderpad=4.5)
cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
cbar.set_ticks([0, 1000, 2000, 3000, 4000, 5000])
cbar.ax.tick_params(labelsize=8)

plt.tight_layout()
plt.show()