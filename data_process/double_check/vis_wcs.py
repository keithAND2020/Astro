import matplotlib.pyplot as plt

from shapely.wkt import loads
with open("fits_regions.txt", "r") as f:
    lines = f.readlines()

polygons = []
for line in lines:
    try:
        wkt_str = line.strip().split(":")[1]
        polygon = loads(wkt_str)
        polygons.append(polygon)
    except Exception as e:
        print(f"wrong: {line.strip()} - {e}")

plt.figure(figsize=(18, 12))

for polygon in polygons:
    ra, dec = polygon.exterior.xy
    plt.plot(ra, dec, color='blue', alpha=0.5)

plt.xlabel("(RA)")
plt.ylabel("(Dec)")
plt.xlim(0, 360) 
plt.ylim(-90, 90)  

plt.savefig('result.png')